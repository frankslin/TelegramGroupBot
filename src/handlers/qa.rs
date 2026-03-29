use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::Result;
use serde::Deserialize;
use serde_json::{json, Value};
use teloxide::prelude::*;
use teloxide::types::{
    ChatAction, InlineKeyboardButton, InlineKeyboardMarkup, MessageEntityKind, MessageEntityRef,
    MessageId, ParseMode, ReplyParameters,
};
use teloxide::RequestError;

use crate::config::{
    parse_third_party_model_id, ThirdPartyModelConfig, ThirdPartyProvider, CONFIG, Q_SYSTEM_PROMPT,
};
use crate::db::database::build_message_insert;
use crate::handlers::access::{check_access_control, is_rate_limited};
use crate::handlers::content::{
    download_telegraph_media, download_twitter_media, extract_telegraph_urls_and_content,
    extract_twitter_urls_and_content, extract_youtube_urls,
};
use crate::handlers::media::{
    collect_message_media, summarize_media_files, MediaCollectionOptions,
};
use crate::handlers::responses::send_response;
use crate::llm::media::MediaKind;
use crate::llm::runtime_models::{
    codex_selected_model_label, is_runtime_provider_ready, resolve_runtime_model_identifier,
    runtime_model_config, runtime_model_count, runtime_models, selected_codex_model_record,
};
use crate::llm::tool_runtime::ToolRuntime;
use crate::llm::{
    call_gemini, call_gemini_with_tool_runtime, call_third_party,
    call_third_party_with_tool_runtime,
};
use crate::state::{AppState, PendingQRequest, QaCommandMode};
use crate::utils::telegram::{build_message_link, start_chat_action_heartbeat};
use crate::utils::timing::{complete_command_timer, start_command_timer};
use tracing::{error, info, warn};

pub const MODEL_CALLBACK_PREFIX: &str = "model_select:";
pub const MODEL_GEMINI: &str = "gemini";
const SEND_MESSAGE_RETRY_ATTEMPTS: usize = 3;
const USER_ERROR_DETAIL_LIMIT: usize = 400;
const CHAT_SEARCH_RESULT_TARGET: usize = 15;
const CHAT_SEARCH_MESSAGE_LIMIT: usize = 3500;

const QC_SYSTEM_PROMPT: &str = "You are a helpful assistant in a Telegram group chat. You can use chat_context_query to retrieve messages from the current source chat only, and you must never assume access to any other chat. Use chat_context_query first when the user asks about prior discussion in this chat. Use web_search only for external or current facts that are not contained in the retrieved chat messages. Cite chat evidence with short snippets and the exact message link when chat history materially informs your answer. Cite web sources normally when web_search is used.\n\nGuidelines for your responses:\n1. Provide a direct, clear answer.\n2. Be concise but comprehensive.\n3. If you use retrieved chat messages, treat them as evidence from this chat only.\n4. If you use web search, cite the sources you relied on.\n5. IMPORTANT: The current UTC date and time is {current_datetime}.\n6. CRITICAL: You must decide the response language yourself using the same language policy as /q.\n7. Language policy:\n- Prefer the language of the user's actual question or request.\n- Ignore quoted text, links, usernames, slash commands, inline code, emojis, and other noise when deciding the response language.\n- If the replied-to content is in a different language from the user's current question, prioritize the current question unless the user explicitly asks you to answer in another language.\n- If the user's message is too short or ambiguous to infer reliably, use this Telegram user language hint: {telegram_user_language_hint}.\n- If that hint is missing, unknown, or still does not provide a reliable answer, default to Chinese.\n- When there is a clear instruction to answer in a specific language, follow that instruction.";

const CHAT_SEARCH_SYSTEM_PROMPT: &str = "You are helping search the current Telegram chat only. The search tool is keyword-based FTS retrieval, not semantic search. You must iteratively use chat_context_query to search this chat, inspect the returned messages, keep only clearly relevant messages, reformulate the query if needed, and continue until you have 15 relevant unique message IDs or you exhaust the 5 allowed chat_context_query calls. Never fabricate message IDs. Only choose message IDs that the tool actually returned. If fewer than 15 clearly relevant messages exist, return the best verified subset and explain that fewer relevant messages were found.";

fn now_unix_seconds() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64
}

fn message_entities_for_text(message: &Message) -> Option<Vec<MessageEntityRef<'_>>> {
    if message.text().is_some() {
        message.parse_entities()
    } else {
        message.parse_caption_entities()
    }
}

fn message_text_or_caption(message: &Message) -> Option<&str> {
    message.text().or_else(|| message.caption())
}

fn is_bot_mention_entity(
    entity: &MessageEntityRef<'_>,
    bot_user_id: i64,
    bot_username_lower: &str,
) -> bool {
    match entity.kind() {
        MessageEntityKind::Mention => {
            if bot_username_lower.is_empty() {
                return false;
            }
            entity
                .text()
                .trim_start_matches('@')
                .eq_ignore_ascii_case(bot_username_lower)
        }
        MessageEntityKind::TextMention { user } => {
            i64::try_from(user.id.0).ok() == Some(bot_user_id)
        }
        _ => false,
    }
}

fn strip_bot_mentions_from_query(
    text: &str,
    entities: Option<&[MessageEntityRef<'_>]>,
    bot_user_id: i64,
    bot_username_lower: &str,
) -> String {
    let Some(entities) = entities else {
        return text.trim().to_string();
    };

    let mut ranges = entities
        .iter()
        .filter(|entity| is_bot_mention_entity(entity, bot_user_id, bot_username_lower))
        .map(|entity| entity.start()..entity.end())
        .collect::<Vec<_>>();
    if ranges.is_empty() {
        return text.trim().to_string();
    }

    ranges.sort_by_key(|range| range.start);
    let mut stripped = text.to_string();
    for range in ranges.into_iter().rev() {
        stripped.replace_range(range, " ");
    }

    stripped.split_whitespace().collect::<Vec<_>>().join(" ")
}

pub fn is_reply_to_this_bot(message: &Message, bot_user_id: i64) -> bool {
    let Some(reply) = message.reply_to_message() else {
        return false;
    };
    let Some(reply_from) = reply.from.as_ref() else {
        return false;
    };
    if !reply_from.is_bot {
        return false;
    }
    i64::try_from(reply_from.id.0).ok() == Some(bot_user_id)
}

pub fn is_mentioning_this_bot(
    message: &Message,
    bot_user_id: i64,
    bot_username_lower: &str,
) -> bool {
    let Some(entities) = message_entities_for_text(message) else {
        return false;
    };

    entities
        .iter()
        .any(|entity| is_bot_mention_entity(entity, bot_user_id, bot_username_lower))
}

pub fn should_auto_q_trigger(
    message: &Message,
    bot_user_id: i64,
    bot_username_lower: &str,
) -> bool {
    if message
        .from
        .as_ref()
        .map(|user| user.is_bot)
        .unwrap_or(false)
    {
        return false;
    }

    let Some(text) = message_text_or_caption(message) else {
        return false;
    };
    if text.trim_start().starts_with('/') {
        return false;
    }

    is_mentioning_this_bot(message, bot_user_id, bot_username_lower)
        || is_reply_to_this_bot(message, bot_user_id)
}

pub fn build_auto_q_query(
    message: &Message,
    bot_user_id: i64,
    bot_username_lower: &str,
) -> Option<String> {
    let text = message_text_or_caption(message)?;
    if text.trim().is_empty() {
        return None;
    }

    let entities = message_entities_for_text(message);
    let stripped =
        strip_bot_mentions_from_query(text, entities.as_deref(), bot_user_id, bot_username_lower);
    if stripped.is_empty() {
        None
    } else {
        Some(stripped)
    }
}

fn truncate_for_user(text: &str, limit: usize) -> String {
    if text.chars().count() <= limit {
        return text.to_string();
    }
    let truncated: String = text.chars().take(limit).collect();
    format!("{truncated}...")
}

fn third_party_provider_label(provider: ThirdPartyProvider) -> &'static str {
    match provider {
        ThirdPartyProvider::OpenRouter => "OpenRouter",
        ThirdPartyProvider::Nvidia => "NVIDIA",
        ThirdPartyProvider::OpenAI => "OpenAI",
        ThirdPartyProvider::OpenAICodex => "OpenAI Codex",
    }
}

fn configured_model_display_name(model_name: &str) -> String {
    if model_name == MODEL_GEMINI {
        "Gemini".to_string()
    } else {
        runtime_model_config(model_name)
            .map(|config| {
                if config.provider == ThirdPartyProvider::OpenAICodex {
                    if let Some(record) = selected_codex_model_record() {
                        if record.slug == config.model {
                            return codex_selected_model_label(&record);
                        }
                    }
                }
                config.name.clone()
            })
            .unwrap_or_else(|| model_name.to_string())
    }
}

fn result_model_display_name(model_name: &str, gemini_model_used: Option<&str>) -> String {
    if model_name == MODEL_GEMINI {
        gemini_model_used
            .unwrap_or(CONFIG.gemini_model.as_str())
            .to_string()
    } else if let Some(config) = runtime_model_config(model_name) {
        if config.provider == ThirdPartyProvider::OpenAICodex {
            if let Some(record) = selected_codex_model_record() {
                if record.slug == config.model {
                    return codex_selected_model_label(&record);
                }
            }
        }
        config.model.clone()
    } else {
        model_name.to_string()
    }
}

fn qa_mode_label(mode: QaCommandMode) -> &'static str {
    match mode {
        QaCommandMode::Standard => "standard",
        QaCommandMode::ChatContext => "chat_context",
    }
}

fn format_llm_error_message(model_name: &str, err: &anyhow::Error) -> String {
    let display_model = configured_model_display_name(model_name);
    let provider =
        runtime_model_config(model_name).map(|config| third_party_provider_label(config.provider));
    let err_text = err.to_string();

    let friendly = match provider {
        Some("OpenRouter") if err_text.contains("OpenRouter request failed") => {
            if err_text.contains("status 404") || err_text.contains("404 Not Found") {
                format!(
                    "Sorry, {display_model} is unavailable on OpenRouter right now. Please pick another model or try again later."
                )
            } else {
                format!(
                    "Sorry, {display_model} returned an OpenRouter error. Please try again later or choose another model."
                )
            }
        }
        Some("NVIDIA") if err_text.contains("NVIDIA request failed") => {
            if err_text.contains("status 404") || err_text.contains("404 Not Found") {
                format!(
                    "Sorry, {display_model} is unavailable on NVIDIA right now. Please pick another model or try again later."
                )
            } else {
                format!(
                    "Sorry, {display_model} returned an NVIDIA error. Please try again later or choose another model."
                )
            }
        }
        _ => format!(
            "Sorry, I couldn't process your request with {display_model}. Please try again later."
        ),
    };

    let detail = truncate_for_user(&err_text, USER_ERROR_DETAIL_LIMIT);
    format!("{friendly}\n\nError: {detail}")
}

async fn send_message_with_retry(
    bot: &Bot,
    chat_id: ChatId,
    text: &str,
    reply_to: Option<MessageId>,
    parse_mode: Option<ParseMode>,
    reply_markup: Option<InlineKeyboardMarkup>,
) -> Result<Message> {
    let text = text.to_string();
    let mut delay = Duration::from_secs_f32(1.5);
    let mut last_err: Option<RequestError> = None;

    for attempt in 0..SEND_MESSAGE_RETRY_ATTEMPTS {
        let mut request = bot.send_message(chat_id, text.clone());
        if let Some(reply_to) = reply_to {
            request = request.reply_parameters(ReplyParameters::new(reply_to));
        }
        if let Some(mode) = parse_mode {
            request = request.parse_mode(mode);
        }
        if let Some(markup) = reply_markup.clone() {
            request = request.reply_markup(markup);
        }

        match request.await {
            Ok(message) => return Ok(message),
            Err(err) => {
                let retryable = matches!(
                    err,
                    RequestError::Network(_) | RequestError::RetryAfter(_) | RequestError::Io(_)
                );
                if !retryable || attempt + 1 == SEND_MESSAGE_RETRY_ATTEMPTS {
                    return Err(err.into());
                }

                warn!("send_message attempt {} failed: {err}", attempt + 1);
                if let RequestError::RetryAfter(wait) = err {
                    tokio::time::sleep(wait.duration()).await;
                } else {
                    tokio::time::sleep(delay).await;
                    delay *= 2;
                }
                last_err = Some(err);
            }
        }
    }

    Err(last_err.expect("send_message retry exhausted").into())
}

fn resolve_exact_model_identifier_with_models(
    identifier: &str,
    models: &[ThirdPartyModelConfig],
) -> Option<String> {
    let trimmed = identifier.trim();
    if trimmed.is_empty() {
        return None;
    }

    if trimmed.eq_ignore_ascii_case(MODEL_GEMINI) {
        return Some(MODEL_GEMINI.to_string());
    }

    if let Some((provider, model)) = parse_third_party_model_id(trimmed) {
        let qualified = format!("{}:{}", provider.as_str(), model);
        return models
            .iter()
            .any(|config| config.id == qualified)
            .then_some(qualified);
    }

    let exact_matches = models
        .iter()
        .filter(|config| config.model == trimmed)
        .collect::<Vec<_>>();
    if exact_matches.len() == 1 {
        return Some(exact_matches[0].id.clone());
    }

    None
}

fn resolve_alias_to_model_id_with_models(
    identifier: &str,
    models: &[ThirdPartyModelConfig],
    alias_map: &[(&str, &str)],
) -> Option<String> {
    if let Some(exact) = resolve_exact_model_identifier_with_models(identifier, models) {
        return Some(exact);
    }

    let alias = identifier.trim().to_lowercase();
    if alias.is_empty() {
        return None;
    }
    if alias == MODEL_GEMINI {
        return Some(MODEL_GEMINI.to_string());
    }

    for (token, model) in alias_map {
        if alias == *token && !model.trim().is_empty() {
            return Some((*model).to_string());
        }
    }

    let fuzzy_matches = models
        .iter()
        .filter(|config| {
            let haystack = format!(
                "{} {} {}",
                config.provider.as_str(),
                config.name,
                config.model
            )
            .to_lowercase();
            haystack.contains(&alias)
        })
        .collect::<Vec<_>>();
    if fuzzy_matches.len() == 1 {
        return Some(fuzzy_matches[0].id.clone());
    }

    None
}

fn normalize_model_identifier_with_models(
    identifier: &str,
    models: &[ThirdPartyModelConfig],
    alias_map: &[(&str, &str)],
) -> String {
    let stripped = identifier.trim();
    if stripped.is_empty() {
        return MODEL_GEMINI.to_string();
    }
    if stripped.eq_ignore_ascii_case(MODEL_GEMINI) {
        return MODEL_GEMINI.to_string();
    }

    resolve_alias_to_model_id_with_models(stripped, models, alias_map)
        .unwrap_or_else(|| stripped.to_string())
}

fn normalize_model_identifier(identifier: &str) -> String {
    if let Some(resolved) = resolve_runtime_model_identifier(identifier) {
        return resolved;
    }

    let mapping = [
        ("llama", CONFIG.llama_model.as_str()),
        ("grok", CONFIG.grok_model.as_str()),
        ("qwen", CONFIG.qwen_model.as_str()),
        ("deepseek", CONFIG.deepseek_model.as_str()),
        ("gpt", CONFIG.gpt_model.as_str()),
    ];
    let models = runtime_models();
    normalize_model_identifier_with_models(identifier, &models, &mapping)
}

fn is_third_party_model_available(config: &ThirdPartyModelConfig) -> bool {
    is_runtime_provider_ready(config.provider)
}

fn has_available_third_party_models_for_request(
    has_images: bool,
    has_video: bool,
    has_audio: bool,
    has_documents: bool,
    require_tools: bool,
) -> bool {
    let models = runtime_models();
    !available_third_party_models_for_request(
        &models,
        has_images,
        has_video,
        has_audio,
        has_documents,
        require_tools,
    )
    .is_empty()
}

fn is_model_configured(model_key: &str) -> bool {
    let normalized = normalize_model_identifier(model_key);
    if normalized == MODEL_GEMINI {
        return true;
    }
    runtime_model_config(&normalized).is_some()
}

fn is_model_configured_for_request(
    model_key: &str,
    has_images: bool,
    has_video: bool,
    has_audio: bool,
    has_documents: bool,
    require_tools: bool,
) -> bool {
    let normalized = normalize_model_identifier(model_key);
    model_supports_media_for_request(
        &normalized,
        has_images,
        has_video,
        has_audio,
        has_documents,
        require_tools,
    )
}

fn model_supports_media_for_request(
    model_name: &str,
    has_images: bool,
    has_video: bool,
    has_audio: bool,
    has_documents: bool,
    require_tools: bool,
) -> bool {
    if model_name == MODEL_GEMINI {
        return true;
    }
    if has_documents {
        return false;
    }

    let Some(config) = runtime_model_config(model_name) else {
        return false;
    };
    if !is_third_party_model_available(&config) {
        return false;
    }
    third_party_model_matches_request_capabilities(
        &config,
        has_images,
        has_video,
        has_audio,
        has_documents,
        require_tools,
    )
}

fn third_party_model_matches_request_capabilities(
    config: &ThirdPartyModelConfig,
    has_images: bool,
    has_video: bool,
    has_audio: bool,
    has_documents: bool,
    require_tools: bool,
) -> bool {
    if has_documents {
        return false;
    }
    if require_tools && !config.tools {
        return false;
    }
    if has_images && !config.image {
        return false;
    }
    if has_video && !config.video {
        return false;
    }
    if has_audio && !config.audio {
        return false;
    }
    true
}

fn available_third_party_models_for_request<'a>(
    models: &'a [ThirdPartyModelConfig],
    has_images: bool,
    has_video: bool,
    has_audio: bool,
    has_documents: bool,
    require_tools: bool,
) -> Vec<&'a ThirdPartyModelConfig> {
    models
        .iter()
        .filter(|config| is_third_party_model_available(config))
        .filter(|config| {
            third_party_model_matches_request_capabilities(
                config,
                has_images,
                has_video,
                has_audio,
                has_documents,
                require_tools,
            )
        })
        .collect()
}

pub fn create_model_selection_keyboard(
    has_images: bool,
    has_video: bool,
    has_audio: bool,
    has_documents: bool,
    require_tools: bool,
) -> InlineKeyboardMarkup {
    let mut keyboard: Vec<Vec<InlineKeyboardButton>> = Vec::new();
    let gemini_button = InlineKeyboardButton::callback(
        "Gemini 3",
        format!("{}{}", MODEL_CALLBACK_PREFIX, MODEL_GEMINI),
    );

    let mut first_row = vec![gemini_button];
    let mut third_party_buttons = Vec::new();
    let models = runtime_models();

    for config in available_third_party_models_for_request(
        &models,
        has_images,
        has_video,
        has_audio,
        has_documents,
        require_tools,
    ) {
        let model_identifier = config.id.trim();
        if model_identifier.is_empty() {
            continue;
        }
        third_party_buttons.push(InlineKeyboardButton::callback(
            config.name.clone(),
            format!("{}{}", MODEL_CALLBACK_PREFIX, model_identifier),
        ));
    }

    if !third_party_buttons.is_empty() {
        first_row.push(third_party_buttons.remove(0));
    }
    keyboard.push(first_row);

    for chunk in third_party_buttons.chunks(2) {
        keyboard.push(chunk.to_vec());
    }

    InlineKeyboardMarkup::new(keyboard)
}

fn build_prompt_from_template(template: &str, telegram_user_language_hint: Option<&str>) -> String {
    let now = chrono::Utc::now().format("%Y-%m-%d %H:%M:%S").to_string();
    template.replace("{current_datetime}", &now).replace(
        "{telegram_user_language_hint}",
        telegram_user_language_hint.unwrap_or("unknown"),
    )
}

fn build_system_prompt(telegram_user_language_hint: Option<&str>) -> String {
    build_prompt_from_template(Q_SYSTEM_PROMPT, telegram_user_language_hint)
}

fn build_chat_context_system_prompt(telegram_user_language_hint: Option<&str>) -> String {
    build_prompt_from_template(QC_SYSTEM_PROMPT, telegram_user_language_hint)
}

fn chat_search_rebuilding_message(command_name: &str) -> String {
    format!(
        "The chat search index is rebuilding right now. Please try /{} again in a few minutes.",
        command_name
    )
}

fn escape_html(text: &str) -> String {
    let mut escaped = String::with_capacity(text.len());
    for ch in text.chars() {
        match ch {
            '&' => escaped.push_str("&amp;"),
            '<' => escaped.push_str("&lt;"),
            '>' => escaped.push_str("&gt;"),
            '"' => escaped.push_str("&quot;"),
            '\'' => escaped.push_str("&#39;"),
            _ => escaped.push(ch),
        }
    }
    escaped
}

fn truncate_for_display(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        return text.to_string();
    }

    let mut truncated: String = text.chars().take(max_chars).collect();
    truncated.push_str("...");
    truncated
}

#[derive(Debug, Deserialize)]
struct ChatSearchSelection {
    selected_message_ids: Vec<i64>,
    note: Option<String>,
}

fn chat_search_response_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "selected_message_ids": {
                "type": "array",
                "items": { "type": "integer" },
                "description": "Up to 15 relevant unique message IDs returned by the search tool, ordered most relevant first."
            },
            "note": {
                "type": "string",
                "description": "Short explanation if fewer than 15 relevant messages were found."
            }
        },
        "required": ["selected_message_ids"],
        "additionalProperties": false
    })
}

fn parse_chat_search_selection(text: &str) -> Option<ChatSearchSelection> {
    serde_json::from_str::<ChatSearchSelection>(text).ok()
}

fn format_chat_search_results_html(
    query: &str,
    hits: &[crate::db::models::ChatSearchHit],
    note: Option<&str>,
    model_name: &str,
) -> String {
    let mut lines = vec![format!("<b>Chat search:</b> {}", escape_html(query.trim()))];

    if hits.is_empty() {
        lines.push("No clearly relevant messages were found.".to_string());
    } else {
        for (index, hit) in hits.iter().enumerate() {
            let username = escape_html(hit.username.as_deref().unwrap_or("Anonymous"));
            let timestamp = escape_html(&hit.date.format("%Y-%m-%d %H:%M:%S UTC").to_string());
            let snippet = escape_html(&truncate_for_display(&hit.snippet, 120));
            let provenance_prefix = if hit.asks_ai {
                let command = hit.ai_command.as_deref().unwrap_or("q");
                format!("[AI ask /{}] ", escape_html(command))
            } else {
                String::new()
            };
            let link = hit
                .link
                .clone()
                .or_else(|| build_message_link(hit.chat_id, hit.message_id));
            let link_html = match link {
                Some(url) => format!("<a href=\"{}\">message link</a>", escape_html(&url)),
                None => "link unavailable".to_string(),
            };
            lines.push(format!(
                "{}. {} [{}]: {}{} {}",
                index + 1,
                username,
                timestamp,
                provenance_prefix,
                snippet,
                link_html
            ));
        }
    }

    if let Some(note) = note.filter(|value| !value.trim().is_empty()) {
        lines.push(format!("<i>{}</i>", escape_html(note.trim())));
    }
    lines.push(format!("<i>Model: {}</i>", escape_html(model_name)));
    lines.join("\n")
}

fn split_html_for_telegram(text: &str, max_chars: usize) -> Vec<String> {
    if text.chars().count() <= max_chars {
        return vec![text.to_string()];
    }

    let mut parts = Vec::new();
    let mut current = String::new();

    for line in text.lines() {
        let line = if current.is_empty() {
            line.to_string()
        } else {
            format!("\n{line}")
        };
        if current.chars().count() + line.chars().count() > max_chars && !current.is_empty() {
            parts.push(current);
            current = line.trim_start_matches('\n').to_string();
        } else {
            current.push_str(&line);
        }
    }

    if !current.is_empty() {
        parts.push(current);
    }

    if parts.is_empty() {
        vec![text.to_string()]
    } else {
        parts
    }
}

async fn send_chat_search_response(
    bot: &Bot,
    chat_id: ChatId,
    message_id: MessageId,
    response_html: &str,
) -> Result<()> {
    let chunks = split_html_for_telegram(response_html, CHAT_SEARCH_MESSAGE_LIMIT);
    let mut chunks_iter = chunks.into_iter();
    let first_chunk = chunks_iter.next().unwrap_or_default();

    bot.edit_message_text(chat_id, message_id, first_chunk)
        .parse_mode(ParseMode::Html)
        .await?;

    for chunk in chunks_iter {
        bot.send_message(chat_id, chunk)
            .parse_mode(ParseMode::Html)
            .await?;
    }

    Ok(())
}

#[allow(deprecated)]
async fn process_request(
    bot: &Bot,
    state: &AppState,
    request: PendingQRequest,
    model_name: &str,
) -> Result<()> {
    if request.mode == QaCommandMode::ChatContext && !state.db.is_search_ready() {
        bot.edit_message_text(
            ChatId(request.chat_id),
            MessageId(request.selection_message_id as i32),
            chat_search_rebuilding_message("qc"),
        )
        .await?;
        return Ok(());
    }

    let system_prompt = match request.mode {
        QaCommandMode::Standard => build_system_prompt(request.telegram_language_code.as_deref()),
        QaCommandMode::ChatContext => {
            build_chat_context_system_prompt(request.telegram_language_code.as_deref())
        }
    };

    let mut query = request.query.clone();
    for content in &request.telegraph_contents {
        query.push_str("\n\n");
        query.push_str(content);
    }
    for content in &request.twitter_contents {
        query.push_str("\n\n");
        query.push_str(content);
    }

    let supports_tools = if model_name == MODEL_GEMINI {
        true
    } else {
        runtime_model_config(model_name)
            .map(|config| config.tools)
            .unwrap_or(false)
    };
    let media_summary = summarize_media_files(&request.media_files);
    let provider_label = if model_name == MODEL_GEMINI {
        "Gemini".to_string()
    } else {
        runtime_model_config(model_name)
            .map(|config| third_party_provider_label(config.provider).to_string())
            .unwrap_or_else(|| "Unknown".to_string())
    };
    let logged_model_name = configured_model_display_name(model_name);

    info!(
        "Processing QA request: mode={}, provider={}, model={}, chat_id={}, user_id={}, message_id={}, selection_message_id={}, tools_enabled={}, images={}, videos={}, audios={}, documents={}, youtube_urls={}, query_len={}",
        qa_mode_label(request.mode),
        provider_label,
        logged_model_name,
        request.chat_id,
        request.user_id,
        request.message_id,
        request.selection_message_id,
        supports_tools,
        media_summary.images,
        media_summary.videos,
        media_summary.audios,
        media_summary.documents,
        request.youtube_urls.len(),
        query.chars().count()
    );

    let _chat_action =
        start_chat_action_heartbeat(bot.clone(), ChatId(request.chat_id), ChatAction::Typing);

    let response = match request.mode {
        QaCommandMode::Standard => {
            if model_name == MODEL_GEMINI {
                let use_pro = !request.media_files.is_empty() || !request.youtube_urls.is_empty();
                call_gemini(
                    &system_prompt,
                    &query,
                    true,
                    false,
                    Some(&CONFIG.gemini_thinking_level),
                    None,
                    use_pro,
                    Some(request.media_files.clone()),
                    Some(request.youtube_urls.clone()),
                    Some("Q_SYSTEM_PROMPT"),
                )
                .await
                .map(|result| (result.text, Some(result.model_used)))
            } else {
                let image_data_list: Vec<Vec<u8>> = request
                    .media_files
                    .iter()
                    .filter(|file| file.kind == MediaKind::Image)
                    .map(|file| file.bytes.clone())
                    .collect();
                call_third_party(
                    &system_prompt,
                    &query,
                    model_name,
                    "Answer to Your Question",
                    &image_data_list,
                    supports_tools,
                )
                .await
                .map(|result| (result, None))
            }
        }
        QaCommandMode::ChatContext => {
            let mut runtime = ToolRuntime::for_qc(state.db.clone(), request.chat_id);
            if model_name == MODEL_GEMINI {
                let use_pro = !request.media_files.is_empty() || !request.youtube_urls.is_empty();
                call_gemini_with_tool_runtime(
                    &format!("{}\n\n{}", system_prompt, runtime.tool_limit_guidance()),
                    &query,
                    &mut runtime,
                    use_pro,
                    Some(request.media_files.clone()),
                    Some(request.youtube_urls.clone()),
                    Some("QC_SYSTEM_PROMPT"),
                    None,
                )
                .await
                .map(|result| (result.text, Some(result.model_used)))
            } else {
                let image_data_list: Vec<Vec<u8>> = request
                    .media_files
                    .iter()
                    .filter(|file| file.kind == MediaKind::Image)
                    .map(|file| file.bytes.clone())
                    .collect();
                call_third_party_with_tool_runtime(
                    &system_prompt,
                    &query,
                    model_name,
                    "Answer about Chat",
                    &image_data_list,
                    &mut runtime,
                )
                .await
                .map(|result| (result, None))
            }
        }
    };
    let (response, gemini_model_used) = match response {
        Ok(response) => response,
        Err(err) => {
            error!(
                "QA request failed: mode={}, provider={}, model={}, chat_id={}, user_id={}, message_id={}, selection_message_id={}, tools_enabled={}, images={}, videos={}, audios={}, documents={}, youtube_urls={}, query_len={}, error={:#}",
                qa_mode_label(request.mode),
                provider_label,
                logged_model_name,
                request.chat_id,
                request.user_id,
                request.message_id,
                request.selection_message_id,
                supports_tools,
                media_summary.images,
                media_summary.videos,
                media_summary.audios,
                media_summary.documents,
                request.youtube_urls.len(),
                query.chars().count(),
                err
            );
            let message = format_llm_error_message(model_name, &err);
            bot.edit_message_text(
                ChatId(request.chat_id),
                MessageId(request.selection_message_id as i32),
                message,
            )
            .await?;
            return Err(err);
        }
    };

    if response.trim().is_empty() {
        bot.edit_message_text(ChatId(request.chat_id), MessageId(request.selection_message_id as i32), "I couldn't find an answer to your question. Please try rephrasing or asking something else.")
            .await?;
        return Ok(());
    }

    let mut response_text = response;
    if !model_name.is_empty() {
        let display_model = result_model_display_name(model_name, gemini_model_used.as_deref());
        response_text.push_str(&format!("\n\nModel: {}", display_model));
    }

    send_response(
        bot,
        ChatId(request.chat_id),
        MessageId(request.selection_message_id as i32),
        &response_text,
        if request.mode == QaCommandMode::ChatContext {
            "Answer about Chat"
        } else {
            "Answer to Your Question"
        },
        ParseMode::Markdown,
    )
    .await?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model(provider: ThirdPartyProvider, name: &str, raw_model: &str) -> ThirdPartyModelConfig {
        ThirdPartyModelConfig {
            id: format!("{}:{}", provider.as_str(), raw_model),
            provider,
            name: name.to_string(),
            model: raw_model.to_string(),
            image: false,
            video: false,
            audio: false,
            tools: true,
        }
    }

    #[test]
    fn available_third_party_models_can_require_tools() {
        let mut without_tools = model(
            ThirdPartyProvider::OpenRouter,
            "No Tools",
            "openrouter/no-tools",
        );
        without_tools.tools = false;
        let models = vec![
            model(
                ThirdPartyProvider::OpenRouter,
                "With Tools",
                "openrouter/with-tools",
            ),
            without_tools,
        ];

        assert!(third_party_model_matches_request_capabilities(
            &models[0], false, false, false, false, true,
        ));
        assert!(!third_party_model_matches_request_capabilities(
            &models[1], false, false, false, false, true,
        ));
    }

    #[test]
    fn normalize_model_identifier_prefers_alias_mapping() {
        let models = vec![
            model(
                ThirdPartyProvider::OpenRouter,
                "Qwen 3",
                "qwen/qwen3-next-80b-a3b-instruct:free",
            ),
            model(
                ThirdPartyProvider::Nvidia,
                "Gemma 3n",
                "google/gemma-3n-e4b-it",
            ),
        ];
        let aliases = [
            ("llama", ""),
            ("grok", ""),
            ("qwen", "openrouter:qwen/qwen3-next-80b-a3b-instruct:free"),
            ("deepseek", ""),
            ("gpt", ""),
        ];

        assert_eq!(
            normalize_model_identifier_with_models("qwen", &models, &aliases),
            "openrouter:qwen/qwen3-next-80b-a3b-instruct:free"
        );
        assert_eq!(
            normalize_model_identifier_with_models("google/gemma-3n-e4b-it", &models, &aliases),
            "nvidia:google/gemma-3n-e4b-it"
        );
    }

    #[test]
    fn normalize_model_identifier_keeps_ambiguous_raw_model_ids_unqualified() {
        let models = vec![
            model(ThirdPartyProvider::OpenRouter, "Shared OR", "shared/model"),
            model(ThirdPartyProvider::Nvidia, "Shared NV", "shared/model"),
        ];
        let aliases = [
            ("llama", ""),
            ("grok", ""),
            ("qwen", ""),
            ("deepseek", ""),
            ("gpt", ""),
        ];

        assert_eq!(
            normalize_model_identifier_with_models("shared/model", &models, &aliases),
            "shared/model"
        );
        assert_eq!(
            normalize_model_identifier_with_models("nvidia:shared/model", &models, &aliases),
            "nvidia:shared/model"
        );
        assert_eq!(
            normalize_model_identifier_with_models("openrouter:shared/model", &models, &aliases),
            "openrouter:shared/model"
        );
    }

    #[test]
    fn codex_selected_model_label_prefers_selected_reasoning_level() {
        let record = crate::llm::runtime_models::CodexSelectedModelRecord {
            slug: "gpt-5.4".to_string(),
            display_name: "GPT-5.4".to_string(),
            description: None,
            input_modalities: vec!["text".to_string()],
            priority: 1,
            etag: None,
            default_reasoning_level: Some("medium".to_string()),
            supported_reasoning_levels: vec![],
            selected_reasoning_level: Some("high".to_string()),
            web_search_tool_type: crate::llm::openai_codex::CodexWebSearchToolType::Text,
            supports_search_tool: false,
            fetched_at: chrono::Utc::now(),
        };

        assert_eq!(codex_selected_model_label(&record), "gpt-5.4 high");
    }

    #[test]
    fn codex_selected_model_label_falls_back_to_default_reasoning_level() {
        let mut record = crate::llm::runtime_models::CodexSelectedModelRecord {
            slug: "gpt-5.4".to_string(),
            display_name: "GPT-5.4".to_string(),
            description: None,
            input_modalities: vec!["text".to_string()],
            priority: 1,
            etag: None,
            default_reasoning_level: Some("medium".to_string()),
            supported_reasoning_levels: vec![],
            selected_reasoning_level: None,
            web_search_tool_type: crate::llm::openai_codex::CodexWebSearchToolType::Text,
            supports_search_tool: false,
            fetched_at: chrono::Utc::now(),
        };

        assert_eq!(codex_selected_model_label(&record), "gpt-5.4 medium");
        record.selected_reasoning_level = Some(String::new());
        assert_eq!(codex_selected_model_label(&record), "gpt-5.4 medium");
    }
}

#[allow(deprecated)]
async fn q_handler_internal(
    bot: Bot,
    state: AppState,
    message: Message,
    query: Option<String>,
    force_gemini: bool,
    command_name: &str,
    mode: QaCommandMode,
) -> Result<()> {
    if !check_access_control(&bot, &message, command_name).await {
        return Ok(());
    }

    let user_id = message
        .from
        .as_ref()
        .and_then(|user| i64::try_from(user.id.0).ok())
        .unwrap_or_default();
    if is_rate_limited(user_id) {
        send_message_with_retry(
            &bot,
            message.chat.id,
            "You're sending commands too quickly. Please wait a moment before trying again.",
            Some(message.id),
            None,
            None,
        )
        .await?;
        return Ok(());
    }

    let query_text_raw = query.unwrap_or_default();
    let query_entities = message_entities_for_text(&message);
    let reply_message = message.reply_to_message();
    let mut reply_text_raw = String::new();
    let mut reply_text = String::new();
    let mut telegraph_contents = Vec::new();
    let mut twitter_contents = Vec::new();

    if let Some(reply) = reply_message {
        reply_text_raw = reply
            .text()
            .map(|value| value.to_string())
            .or_else(|| reply.caption().map(|value| value.to_string()))
            .unwrap_or_default();
        if !reply_text_raw.trim().is_empty() {
            let reply_entities = message_entities_for_text(reply);
            let (reply_text_processed, reply_telegraph) =
                extract_telegraph_urls_and_content(&reply_text_raw, reply_entities.as_deref(), 5)
                    .await;
            let (reply_text_processed, reply_twitter) = extract_twitter_urls_and_content(
                &reply_text_processed,
                reply_entities.as_deref(),
                5,
            )
            .await;
            telegraph_contents.extend(reply_telegraph);
            twitter_contents.extend(reply_twitter);
            reply_text = reply_text_processed;
        }
    }

    let original_query = if query_text_raw.trim().is_empty() {
        reply_text_raw.clone()
    } else {
        query_text_raw.clone()
    };

    if original_query.trim().is_empty() {
        send_message_with_retry(
            &bot,
            message.chat.id,
            &format!(
                "Please provide a question or reply to a message with /{}.",
                command_name
            ),
            Some(message.id),
            None,
            None,
        )
        .await?;
        return Ok(());
    }

    if mode == QaCommandMode::ChatContext && !state.db.is_search_ready() {
        send_message_with_retry(
            &bot,
            message.chat.id,
            &chat_search_rebuilding_message("qc"),
            Some(message.id),
            None,
            None,
        )
        .await?;
        return Ok(());
    }

    let mut query_text = query_text_raw.clone();
    if !query_text.trim().is_empty() {
        let (query_text_processed, query_telegraph) =
            extract_telegraph_urls_and_content(&query_text, query_entities.as_deref(), 5).await;
        let (query_text_processed, query_twitter) =
            extract_twitter_urls_and_content(&query_text_processed, query_entities.as_deref(), 5)
                .await;
        telegraph_contents.extend(query_telegraph);
        twitter_contents.extend(query_twitter);
        query_text = query_text_processed;
    }

    let query_base = if query_text.trim().is_empty() {
        reply_text.clone()
    } else if reply_text.trim().is_empty() {
        query_text.clone()
    } else {
        format!(
            "Context from replied message: \"{}\"\n\nQuestion: {}",
            reply_text, query_text
        )
    };

    let (query_text, youtube_urls) = extract_youtube_urls(&query_base, 10);

    let user_language_code = message
        .from
        .as_ref()
        .and_then(|user| user.language_code.as_deref());

    let username = message
        .from
        .as_ref()
        .map(|user| user.full_name())
        .unwrap_or_else(|| "Anonymous".to_string());

    let db_query_text = if let Some(reply) = message.reply_to_message() {
        let replied_text = reply
            .text()
            .map(|value| value.to_string())
            .or_else(|| reply.caption().map(|value| value.to_string()))
            .unwrap_or_default();
        if replied_text.is_empty() {
            query_text.clone()
        } else {
            format!(
                "Context from replied message: \"{}\"\n\nQuestion: {}",
                replied_text, query_text
            )
        }
    } else {
        query_text.clone()
    };

    let media_options = MediaCollectionOptions::for_qa();
    let max_files = media_options.max_files;
    let media = collect_message_media(&bot, &state, &message, media_options).await;
    let mut media_files = media.files;

    let mut remaining = max_files.saturating_sub(media_files.len());
    if remaining > 0 {
        let telegraph_files = download_telegraph_media(&telegraph_contents, remaining).await;
        remaining = remaining.saturating_sub(telegraph_files.len());
        media_files.extend(telegraph_files);
    }

    if remaining > 0 {
        let twitter_files = download_twitter_media(&twitter_contents, remaining).await;
        media_files.extend(twitter_files);
    }

    let media_summary = summarize_media_files(&media_files);
    let has_images = media_summary.images > 0;
    let has_video = media_summary.videos > 0;
    let has_audio = media_summary.audios > 0;
    let has_documents = media_summary.documents > 0;

    let require_tools = mode.requires_custom_tools();
    let must_use_gemini = force_gemini
        || has_video
        || has_audio
        || has_documents
        || !youtube_urls.is_empty()
        || !has_available_third_party_models_for_request(
            has_images,
            has_video,
            has_audio,
            has_documents,
            require_tools,
        )
        || runtime_model_count() == 0;
    if must_use_gemini {
        let processing_message_text = if has_video {
            "Analyzing video and processing your question...".to_string()
        } else if has_audio {
            "Analyzing audio and processing your question...".to_string()
        } else if has_images {
            format!(
                "Analyzing {} image(s) and processing your question...",
                media_summary.images
            )
        } else if has_documents {
            format!(
                "Analyzing {} document(s) and processing your question...",
                media_summary.documents
            )
        } else if !twitter_contents.is_empty() {
            format!(
                "Analyzing {} Twitter post(s) and processing your question...",
                twitter_contents.len()
            )
        } else if !youtube_urls.is_empty() {
            format!(
                "Analyzing {} YouTube video(s) and processing your question...",
                youtube_urls.len()
            )
        } else {
            "Processing your question...".to_string()
        };
        let processing_message = send_message_with_retry(
            &bot,
            message.chat.id,
            &processing_message_text,
            Some(message.id),
            None,
            None,
        )
        .await?;
        let mut timer = start_command_timer(command_name, &message);
        let _chat_action =
            start_chat_action_heartbeat(bot.clone(), message.chat.id, ChatAction::Typing);
        let use_pro = if command_name == "qq" {
            false
        } else {
            media_summary.total > 0 || !youtube_urls.is_empty()
        };
        let response = if mode == QaCommandMode::ChatContext {
            let mut runtime = ToolRuntime::for_qc(state.db.clone(), message.chat.id.0);
            call_gemini_with_tool_runtime(
                &format!(
                    "{}\n\n{}",
                    build_chat_context_system_prompt(user_language_code),
                    runtime.tool_limit_guidance()
                ),
                &query_text,
                &mut runtime,
                use_pro,
                Some(media_files.clone()),
                Some(youtube_urls.clone()),
                Some("QC_SYSTEM_PROMPT"),
                None,
            )
            .await
        } else {
            call_gemini(
                &build_system_prompt(user_language_code),
                &query_text,
                true,
                false,
                Some(&CONFIG.gemini_thinking_level),
                None,
                use_pro,
                Some(media_files.clone()),
                Some(youtube_urls.clone()),
                Some("Q_SYSTEM_PROMPT"),
            )
            .await
        };
        let response = match response {
            Ok(response) => response.text,
            Err(err) => {
                let message = format_llm_error_message(MODEL_GEMINI, &err);
                bot.edit_message_text(processing_message.chat.id, processing_message.id, message)
                    .await?;
                return Err(err);
            }
        };

        send_response(
            &bot,
            processing_message.chat.id,
            processing_message.id,
            &response,
            if mode == QaCommandMode::ChatContext {
                "Answer about Chat"
            } else {
                "Answer to Your Question"
            },
            ParseMode::Markdown,
        )
        .await?;
        complete_command_timer(&mut timer, "success", None);
        return Ok(());
    }

    let has_media = has_images || has_video || has_audio || has_documents;
    let mut selection_text = "Please select which AI model to use for your question:".to_string();
    if has_media {
        selection_text.push_str("\n\n*Note: Only models that support media are shown.*");
    }

    let keyboard = create_model_selection_keyboard(
        has_images,
        has_video,
        has_audio,
        has_documents,
        require_tools,
    );
    let selection_message = send_message_with_retry(
        &bot,
        message.chat.id,
        &selection_text,
        Some(message.id),
        Some(ParseMode::Markdown),
        Some(keyboard),
    )
    .await?;

    let request_key = format!("{}_{}", message.chat.id.0, selection_message.id.0);
    let timer = start_command_timer(command_name, &message);

    let pending_request = PendingQRequest {
        user_id,
        username: username.clone(),
        query: query_text.clone(),
        original_query: original_query.clone(),
        db_query_text: db_query_text.clone(),
        telegram_language_code: user_language_code.map(str::to_string),
        media_files,
        youtube_urls,
        telegraph_contents: telegraph_contents
            .iter()
            .map(|c| c.text_content.clone())
            .collect(),
        twitter_contents: twitter_contents
            .iter()
            .map(|c| c.text_content.clone())
            .collect(),
        chat_id: message.chat.id.0,
        message_id: message.id.0 as i64,
        selection_message_id: selection_message.id.0 as i64,
        original_user_id: user_id,
        reply_to_message_id: message.reply_to_message().map(|msg| msg.id.0 as i64),
        timestamp: now_unix_seconds(),
        command_timer: Some(timer),
        mode,
    };

    state
        .pending_q_requests
        .lock()
        .insert(request_key.clone(), pending_request);

    let bot_clone = bot.clone();
    let state_clone = state.clone();
    tokio::spawn(async move {
        handle_model_timeout(bot_clone, state_clone, request_key).await;
    });

    let db_insert = build_message_insert(
        Some(user_id),
        Some(username),
        message
            .text()
            .map(|value| value.to_string())
            .or_else(|| message.caption().map(|value| value.to_string()))
            .or_else(|| Some(original_query.clone())),
        None,
        message.date,
        message.reply_to_message().map(|msg| msg.id.0 as i64),
        Some(message.chat.id.0),
        Some(message.id.0 as i64),
        Some(db_query_text.clone()),
        true,
        Some(command_name.to_string()),
        true,
        true,
    );
    let _ = state.db.queue_message_insert(db_insert).await;

    Ok(())
}

pub async fn q_handler(
    bot: Bot,
    state: AppState,
    message: Message,
    query: Option<String>,
    force_gemini: bool,
    command_name: &str,
) -> Result<()> {
    q_handler_internal(
        bot,
        state,
        message,
        query,
        force_gemini,
        command_name,
        QaCommandMode::Standard,
    )
    .await
}

pub async fn qc_handler(
    bot: Bot,
    state: AppState,
    message: Message,
    query: Option<String>,
) -> Result<()> {
    q_handler_internal(
        bot,
        state,
        message,
        query,
        false,
        "qc",
        QaCommandMode::ChatContext,
    )
    .await
}

pub async fn s_handler(
    bot: Bot,
    state: AppState,
    message: Message,
    query: Option<String>,
) -> Result<()> {
    if !check_access_control(&bot, &message, "s").await {
        return Ok(());
    }

    let user_id = message
        .from
        .as_ref()
        .and_then(|user| i64::try_from(user.id.0).ok())
        .unwrap_or_default();
    if is_rate_limited(user_id) {
        send_message_with_retry(
            &bot,
            message.chat.id,
            "You're sending commands too quickly. Please wait a moment before trying again.",
            Some(message.id),
            None,
            None,
        )
        .await?;
        return Ok(());
    }

    let query_text = query
        .filter(|value| !value.trim().is_empty())
        .or_else(|| {
            message.reply_to_message().and_then(|reply| {
                reply
                    .text()
                    .map(|value| value.to_string())
                    .or_else(|| reply.caption().map(|value| value.to_string()))
            })
        })
        .unwrap_or_default();
    if query_text.trim().is_empty() {
        send_message_with_retry(
            &bot,
            message.chat.id,
            "Please provide a search query or reply to a message with /s.",
            Some(message.id),
            None,
            None,
        )
        .await?;
        return Ok(());
    }

    if !state.db.is_search_ready() {
        send_message_with_retry(
            &bot,
            message.chat.id,
            &chat_search_rebuilding_message("s"),
            Some(message.id),
            None,
            None,
        )
        .await?;
        return Ok(());
    }

    let mut timer = start_command_timer("s", &message);
    let processing_message = send_message_with_retry(
        &bot,
        message.chat.id,
        "Searching this chat...",
        Some(message.id),
        None,
        None,
    )
    .await?;
    let _chat_action =
        start_chat_action_heartbeat(bot.clone(), message.chat.id, ChatAction::Typing);

    let mut runtime = ToolRuntime::for_search(state.db.clone(), message.chat.id.0);
    let response = call_gemini_with_tool_runtime(
        &format!(
            "{}\n\n{}",
            CHAT_SEARCH_SYSTEM_PROMPT,
            runtime.tool_limit_guidance()
        ),
        &query_text,
        &mut runtime,
        false,
        None,
        None,
        Some("CHAT_SEARCH_SYSTEM_PROMPT"),
        Some(chat_search_response_schema()),
    )
    .await;
    let response = match response {
        Ok(response) => response,
        Err(err) => {
            let message = format_llm_error_message(MODEL_GEMINI, &err);
            bot.edit_message_text(processing_message.chat.id, processing_message.id, message)
                .await?;
            complete_command_timer(
                &mut timer,
                "error",
                Some("gemini_search_failed".to_string()),
            );
            return Err(err);
        }
    };

    let selection = parse_chat_search_selection(&response.text);
    let mut selected_hits = selection
        .as_ref()
        .map(|selection| {
            runtime.select_hits_by_message_ids(
                &selection.selected_message_ids,
                CHAT_SEARCH_RESULT_TARGET,
            )
        })
        .unwrap_or_default();
    if selected_hits.len() > CHAT_SEARCH_RESULT_TARGET {
        selected_hits.truncate(CHAT_SEARCH_RESULT_TARGET);
    }

    let note = selection.as_ref().and_then(|value| value.note.as_deref()).or_else(|| {
        (selected_hits.len() < CHAT_SEARCH_RESULT_TARGET).then_some(
            "Fewer than 15 clearly relevant messages were found within the 5 allowed search attempts.",
        )
    });
    let response_html =
        format_chat_search_results_html(&query_text, &selected_hits, note, &response.model_used);

    send_chat_search_response(
        &bot,
        processing_message.chat.id,
        processing_message.id,
        &response_html,
    )
    .await?;
    complete_command_timer(&mut timer, "success", None);

    Ok(())
}

pub async fn qq_handler(
    bot: Bot,
    state: AppState,
    message: Message,
    query: Option<String>,
) -> Result<()> {
    q_handler(bot, state, message, query, true, "qq").await
}

pub async fn handle_model_timeout(bot: Bot, state: AppState, request_key: String) {
    tokio::time::sleep(Duration::from_secs(CONFIG.model_selection_timeout)).await;
    let request = state.pending_q_requests.lock().remove(&request_key);
    let Some(mut request) = request else {
        return;
    };

    let mut default_model = normalize_model_identifier(&CONFIG.default_q_model);
    if default_model != MODEL_GEMINI && !is_model_configured(&default_model) {
        default_model = MODEL_GEMINI.to_string();
    }

    let summary = summarize_media_files(&request.media_files);
    let has_images = summary.images > 0;
    let has_video = summary.videos > 0;
    let has_audio = summary.audios > 0;
    let has_documents = summary.documents > 0;
    if !is_model_configured_for_request(
        &default_model,
        has_images,
        has_video,
        has_audio,
        has_documents,
        request.mode.requires_custom_tools(),
    ) {
        default_model = MODEL_GEMINI.to_string();
    }

    let _ = bot
        .edit_message_text(
            ChatId(request.chat_id),
            MessageId(request.selection_message_id as i32),
            "No model selected in time. Using default model...",
        )
        .reply_markup(InlineKeyboardMarkup::new(
            Vec::<Vec<InlineKeyboardButton>>::new(),
        ))
        .await;

    let command_timer = request.command_timer.take();
    let result = process_request(&bot, &state, request, &default_model).await;
    if let Some(mut timer) = command_timer {
        let status = if result.is_ok() { "success" } else { "error" };
        complete_command_timer(
            &mut timer,
            status,
            Some("timeout_default_model".to_string()),
        );
    }
    if let Err(err) = result {
        error!(
            "Timed-out QA request failed after default-model fallback: model={}, error={:#}",
            configured_model_display_name(&default_model),
            err
        );
    }
}

pub async fn model_selection_callback(
    bot: Bot,
    state: AppState,
    query: CallbackQuery,
) -> Result<()> {
    bot.answer_callback_query(query.id.clone()).await?;

    let Some(data) = &query.data else {
        return Ok(());
    };
    if !data.starts_with(MODEL_CALLBACK_PREFIX) {
        return Ok(());
    }

    let selected_token = data.trim_start_matches(MODEL_CALLBACK_PREFIX);
    let selected_model = normalize_model_identifier(selected_token);

    let message = match query.message.clone() {
        Some(msg) => msg,
        None => return Ok(()),
    };

    let request_key = format!("{}_{}", message.chat().id.0, message.id().0);
    let request = {
        let mut pending = state.pending_q_requests.lock();
        pending.remove(&request_key)
    };
    let mut request = match request {
        Some(req) => req,
        None => {
            bot.edit_message_text(message.chat().id, message.id(), "This request has expired.")
                .reply_markup(InlineKeyboardMarkup::new(
                    Vec::<Vec<InlineKeyboardButton>>::new(),
                ))
                .await?;
            return Ok(());
        }
    };

    let summary = summarize_media_files(&request.media_files);
    let has_images = summary.images > 0;
    let has_video = summary.videos > 0;
    let has_audio = summary.audios > 0;
    let has_documents = summary.documents > 0;
    if !is_model_configured_for_request(
        selected_token,
        has_images,
        has_video,
        has_audio,
        has_documents,
        request.mode.requires_custom_tools(),
    ) {
        bot.answer_callback_query(query.id.clone()).await?;
        return Ok(());
    }

    let query_user_id = i64::try_from(query.from.id.0).unwrap_or_default();
    if request.original_user_id != query_user_id {
        bot.answer_callback_query(query.id.clone()).await?;
        return Ok(());
    }

    if now_unix_seconds() - request.timestamp > CONFIG.model_selection_timeout as i64 {
        if let Some(mut timer) = request.command_timer.take() {
            complete_command_timer(&mut timer, "expired", Some("selection_timeout".to_string()));
        }
        bot.edit_message_text(
            message.chat().id,
            message.id(),
            "Selection timed out. Please try again.",
        )
        .reply_markup(InlineKeyboardMarkup::new(
            Vec::<Vec<InlineKeyboardButton>>::new(),
        ))
        .await?;
        return Ok(());
    }

    if !model_supports_media_for_request(
        &selected_model,
        has_images,
        has_video,
        has_audio,
        has_documents,
        request.mode.requires_custom_tools(),
    ) {
        bot.edit_message_text(
            message.chat().id,
            message.id(),
            if request.mode == QaCommandMode::ChatContext {
                "Selected model cannot use the required tool set for this request. Please choose Gemini or a tool-capable model."
            } else {
                "Selected model does not support the attached media. Please choose Gemini."
            },
        )
        .reply_markup(InlineKeyboardMarkup::new(
            Vec::<Vec<InlineKeyboardButton>>::new(),
        ))
        .await?;
        return Ok(());
    }

    let display_name = configured_model_display_name(&selected_model);

    let processing_text = if summary.videos > 0 {
        format!(
            "Analyzing video and processing your question with {}...",
            display_name
        )
    } else if summary.audios > 0 {
        format!(
            "Analyzing audio and processing your question with {}...",
            display_name
        )
    } else if summary.images > 0 {
        format!(
            "Analyzing {} image(s) and processing your question with {}...",
            summary.images, display_name
        )
    } else if summary.documents > 0 {
        format!(
            "Analyzing {} document(s) and processing your question with {}...",
            summary.documents, display_name
        )
    } else {
        format!("Processing your question with {}...", display_name)
    };

    bot.edit_message_text(message.chat().id, message.id(), processing_text)
        .reply_markup(InlineKeyboardMarkup::new(
            Vec::<Vec<InlineKeyboardButton>>::new(),
        ))
        .await?;

    let command_timer = request.command_timer.take();
    let result = process_request(&bot, &state, request, &selected_model).await;
    if let Some(mut timer) = command_timer {
        let status = if result.is_ok() { "success" } else { "error" };
        complete_command_timer(&mut timer, status, None);
    }

    if let Err(err) = result {
        return Err(err);
    }

    Ok(())
}
