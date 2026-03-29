use std::collections::HashSet;
use std::future::Future;
use std::path::Path;
use std::time::Duration;

use anyhow::Result;
use chrono::Utc;
use teloxide::prelude::*;
use teloxide::types::{
    ChatAction, FileId, InlineKeyboardButton, InlineKeyboardMarkup, InputFile, InputMedia,
    InputMediaPhoto, MessageEntityRef, MessageId, ParseMode, ReplyParameters,
};
use teloxide::RequestError;

use crate::config::{
    CONFIG, FACTCHECK_SYSTEM_PROMPT, PAINTME_SYSTEM_PROMPT, PORTRAIT_SYSTEM_PROMPT,
    PROFILEME_SYSTEM_PROMPT, TLDR_SYSTEM_PROMPT,
};
use crate::handlers::access::{check_access_control, check_admin_access, is_rate_limited};
use crate::handlers::content::{
    create_telegraph_page, extract_telegraph_urls_and_content, extract_twitter_urls_and_content,
};
use crate::handlers::media::{
    collect_message_media, get_file_url, summarize_media_files, MediaCollectionOptions,
    MediaSummary,
};
use crate::handlers::responses::send_response;
use crate::llm::media::detect_mime_type;
use crate::llm::openai_codex;
use crate::llm::runtime_models::{runtime_model_count, selected_codex_model_record};
use crate::llm::web_search::is_search_enabled;
use crate::llm::{
    call_gemini, generate_image_with_gemini, generate_music_with_lyria, generate_video_with_veo,
    GeminiImageConfig,
};
use crate::state::{AppState, MediaGroupItem, PendingImageRequest};
use crate::tools::cwd_uploader::upload_image_bytes_to_cwd;
use crate::utils::logging::read_recent_log_lines;
use crate::utils::telegram::start_chat_action_heartbeat;
use crate::utils::timing::{complete_command_timer, start_command_timer};
use tracing::{error, warn};

const IMAGE_RESOLUTION_OPTIONS: [&str; 3] = ["2K", "4K", "1K"];
const IMAGE_ASPECT_RATIO_OPTIONS: [&str; 14] = [
    "4:3", "3:4", "16:9", "9:16", "1:1", "21:9", "3:2", "2:3", "5:4", "4:5", "4:1", "1:4", "8:1",
    "1:8",
];
const IMAGE_RESOLUTION_CALLBACK_PREFIX: &str = "image_res:";
const IMAGE_ASPECT_RATIO_CALLBACK_PREFIX: &str = "image_aspect:";
const IMAGE_DEFAULT_RESOLUTION: &str = "2K";
const IMAGE_DEFAULT_ASPECT_RATIO: &str = "4:3";
const IMAGE_CAPTION_LIMIT: usize = 1000;
const IMAGE_CAPTION_PROMPT_PREVIEW: usize = 900;
const VID_TELEGRAM_RETRY_ATTEMPTS: usize = 3;
const DIAGNOSE_LOG_TAIL_LINES: usize = 12;
const DIAGNOSE_TEXT_LIMIT: usize = 3900;
const MYSONG_LLM_MAX_ATTEMPTS: usize = 3;
const MYSONG_LLM_RETRY_BASE_DELAY_MS: u64 = 2_000;
const MYSONG_DEFAULT_LANGUAGE: &str = "English";
const MYSONG_SUMMARY_SYSTEM_PROMPT: &str = r#"You are preparing a music-generation brief for a Telegram user's personal theme song.

Analyze the user's recent chat history and summarize only stable patterns, not one-off comments.

Output plain text with these headings exactly:
Persona:
Communication style:
Recurring interests:
Emotional tone:
Social role in the chat:
Music style cues:
Theme song angle:
Language cues:
Constraints:

Requirements:
- Infer personality, rhythm, energy, humor, and likely musical vibe from the user's writing style.
- Suggest plausible genre, instrumentation, tempo, and vocal feel that match the user's chatting style.
- Do not quote the user's messages.
- Do not include timestamps, message IDs, or usernames.
- Keep it concise and useful for a music prompt writer.
- Always reply in English."#;
const MYSONG_PROMPT_SYSTEM_PROMPT: &str = r#"You are an expert prompt engineer for Google's Lyria 3 Pro music-generation model.

You will receive a persona summary plus optional user instructions. Your task is to write one final prompt for a full-length theme song about that user.

Requirements for the final prompt:
- Any explicit user direction about language, era, genre, instrumentation, or style is a hard requirement and must be preserved.
- Use the persona summary for subject matter and emotional tone, but do not override explicit user directions with your own defaults.
- Make it a full song of about 2 minutes.
- Match the musical style to the user's chatting style.
- Include genre, instrumentation, mood, tempo/BPM, vocal style, production details, and overall atmosphere.
- Request clear structure using tags such as [Intro], [Verse], [Chorus], [Bridge], and [Outro].
- Ask for memorable lyrics about the user's vibe, habits, interests, worldview, and role in the group.
- Do not mention timestamps, usernames, message IDs, or direct quotes from the chat history.
- Do not request any specific artist, band, or copyrighted lyrics.
- Output only the final Lyria prompt text, with no markdown fences or explanation."#;

#[derive(Debug, Clone)]
struct ImageRequestContext {
    prompt: String,
    image_urls: Vec<String>,
    telegraph_contents: Vec<String>,
    original_message_text: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct MysongLanguageSelection {
    target_language: &'static str,
    fallback_notice: Option<String>,
}

fn strip_command_prefix(text: &str, command_prefix: &str) -> String {
    if let Some(stripped) = text.strip_prefix(command_prefix) {
        stripped.trim().to_string()
    } else {
        text.to_string()
    }
}

fn format_user_history_for_persona(history: &[crate::db::models::MessageRow]) -> String {
    let mut formatted = String::from("Here is the user's recent chat history in this group:\n\n");
    for msg in history {
        let timestamp = msg.date.format("%Y-%m-%d %H:%M:%S");
        let text = msg.text.as_deref().unwrap_or_default();
        formatted.push_str(&format!("{}: {}\n", timestamp, text));
    }
    formatted
}

fn note_mentions_any(note: &str, ascii_keywords: &[&str], native_keywords: &[&str]) -> bool {
    let lower = note.to_ascii_lowercase();
    ascii_keywords.iter().any(|keyword| lower.contains(keyword))
        || native_keywords.iter().any(|keyword| note.contains(keyword))
}

fn resolve_mysong_language(note: Option<&str>) -> MysongLanguageSelection {
    let Some(note) = note.filter(|value| !value.trim().is_empty()) else {
        return MysongLanguageSelection {
            target_language: MYSONG_DEFAULT_LANGUAGE,
            fallback_notice: None,
        };
    };

    for (display_name, ascii_keywords, native_keywords) in [
        ("English", vec!["english"], vec!["英语", "英文"]),
        ("German", vec!["german"], vec!["德语", "德文"]),
        ("Spanish", vec!["spanish"], vec!["西班牙语", "西语", "西文"]),
        ("French", vec!["french"], vec!["法语", "法文"]),
        ("Hindi", vec!["hindi"], vec!["印地语", "印度语"]),
        (
            "Japanese",
            vec!["japanese", "j-pop", "anime song", "anisong"],
            vec!["日语", "日文", "日本语", "日语歌", "日文歌", "日本动漫"],
        ),
        (
            "Korean",
            vec!["korean", "k-pop"],
            vec!["韩语", "韓語", "韩文", "韓文"],
        ),
        ("Portuguese", vec!["portuguese"], vec!["葡语", "葡萄牙语"]),
    ] {
        if note_mentions_any(note, &ascii_keywords, &native_keywords) {
            return MysongLanguageSelection {
                target_language: display_name,
                fallback_notice: None,
            };
        }
    }

    for (display_name, ascii_keywords, native_keywords) in [
        (
            "Chinese",
            vec!["chinese", "mandarin", "cantonese"],
            vec!["中文", "汉语", "漢語", "国语", "國語", "粤语", "粵語"],
        ),
        ("Italian", vec!["italian"], vec!["意大利语", "意语"]),
        ("Arabic", vec!["arabic"], vec!["阿拉伯语"]),
        ("Russian", vec!["russian"], vec!["俄语", "俄文"]),
        ("Turkish", vec!["turkish"], vec!["土耳其语"]),
        ("Vietnamese", vec!["vietnamese"], vec!["越南语"]),
    ] {
        if note_mentions_any(note, &ascii_keywords, &native_keywords) {
            return MysongLanguageSelection {
                target_language: MYSONG_DEFAULT_LANGUAGE,
                fallback_notice: Some(format!(
                    "Lyria 3 currently does not support {} lyrics here, so I generated the song in English instead.",
                    display_name
                )),
            };
        }
    }

    MysongLanguageSelection {
        target_language: MYSONG_DEFAULT_LANGUAGE,
        fallback_notice: None,
    }
}

fn build_mysong_prompt_request(
    persona_summary: &str,
    note: Option<&str>,
    target_language: &str,
) -> String {
    let mut request = format!(
        "Target lyric language: {}\n\nPersona summary:\n{}\n",
        target_language,
        persona_summary.trim()
    );

    if let Some(note) = note.filter(|value| !value.trim().is_empty()) {
        request
            .push_str("\nMandatory user direction (must be preserved exactly where possible):\n");
        request.push_str(note.trim());
        request.push('\n');
    }

    request.push_str(&format!(
        "\nWrite the final Lyria prompt entirely in {} and explicitly request vocals and lyrics in {}. Treat any explicit era, genre, anime/J-pop reference, instrumentation request, or language request from the user direction as mandatory.",
        target_language, target_language
    ));
    request
}

fn audio_file_name_for_mime(mime_type: &str) -> &'static str {
    match mime_type.trim().to_ascii_lowercase().as_str() {
        "audio/wav" | "audio/x-wav" => "mysong.wav",
        _ => "mysong.mp3",
    }
}

fn audio_should_use_send_audio(mime_type: &str) -> bool {
    matches!(
        mime_type.trim().to_ascii_lowercase().as_str(),
        "audio/mpeg" | "audio/mp3"
    )
}

fn build_mysong_lyrics_message(
    lyrics_text: &str,
    notes_text: Option<&str>,
    fallback_notice: Option<&str>,
) -> String {
    let mut message = String::new();
    if let Some(fallback_notice) = fallback_notice.filter(|value| !value.trim().is_empty()) {
        message.push_str(fallback_notice.trim());
        message.push_str("\n\n");
    }

    message.push_str("Lyrics\n\n");
    message.push_str(lyrics_text.trim());

    if let Some(notes_text) = notes_text.filter(|value| !value.trim().is_empty()) {
        message.push_str("\n\nSong Notes\n\n");
        message.push_str(notes_text.trim());
    }

    message
}

async fn build_mysong_audio_caption(
    lyrics_message: &str,
    model_name: &str,
    prompt_language: &str,
) -> String {
    let base_caption = format!(
        "Generated by {} in {}.",
        escape_html(model_name),
        escape_html(prompt_language)
    );

    if let Some(url) = create_telegraph_page("Your Theme Song Lyrics", lyrics_message).await {
        return format!(
            "{}\n<a href=\"{}\">Lyrics and notes</a>",
            base_caption,
            escape_html(&url)
        );
    }

    let (preview, was_truncated) = truncate_chars(lyrics_message, 700);
    let preview = if was_truncated {
        format!("{}...", preview)
    } else {
        preview
    };

    let caption = format!("{}\n<pre>{}</pre>", base_caption, escape_html(&preview));
    if caption.chars().count() <= IMAGE_CAPTION_LIMIT {
        caption
    } else {
        base_caption
    }
}

fn message_entities_for_text(message: &Message) -> Option<Vec<MessageEntityRef<'_>>> {
    if message.text().is_some() {
        message.parse_entities()
    } else {
        message.parse_caption_entities()
    }
}

fn build_factcheck_system_prompt(telegram_user_language_hint: Option<&str>) -> String {
    let now = chrono::Utc::now().format("%Y-%m-%d %H:%M:%S").to_string();
    FACTCHECK_SYSTEM_PROMPT
        .replace("{current_datetime}", &now)
        .replace(
            "{telegram_user_language_hint}",
            telegram_user_language_hint.unwrap_or("unknown"),
        )
}

fn build_factcheck_statement(
    query_text: &str,
    reply_text: &str,
    media_summary: &MediaSummary,
) -> String {
    let query_text = query_text.trim();
    let reply_text = reply_text.trim();

    if !query_text.is_empty() && !reply_text.is_empty() {
        return format!(
            "<reply_context>\n{}\n</reply_context>\n\n<factcheck_target>\n{}\n</factcheck_target>",
            reply_text, query_text
        );
    }

    if !query_text.is_empty() {
        return format!("<factcheck_target>\n{}\n</factcheck_target>", query_text);
    }

    if !reply_text.is_empty() {
        return format!("<factcheck_target>\n{}\n</factcheck_target>", reply_text);
    }

    if media_summary.videos > 0 {
        return "<auto_factcheck_target source=\"media_only\" kind=\"video\" />".to_string();
    }
    if media_summary.audios > 0 {
        return "<auto_factcheck_target source=\"media_only\" kind=\"audio\" />".to_string();
    }
    if media_summary.images > 0 {
        return "<auto_factcheck_target source=\"media_only\" kind=\"image\" />".to_string();
    }
    if media_summary.documents > 0 {
        return "<auto_factcheck_target source=\"media_only\" kind=\"document\" />".to_string();
    }

    String::new()
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

fn truncate_chars(text: &str, max_chars: usize) -> (String, bool) {
    let mut iter = text.chars();
    let truncated: String = iter.by_ref().take(max_chars).collect();
    let was_truncated = iter.next().is_some();
    (truncated, was_truncated)
}

fn bool_label(value: bool) -> &'static str {
    if value {
        "yes"
    } else {
        "no"
    }
}

fn redact_sensitive_text(text: &str) -> String {
    let mut redacted = text.to_string();
    let secrets = [
        CONFIG.bot_token.as_str(),
        CONFIG.gemini_api_key.as_str(),
        CONFIG.openrouter_api_key.as_str(),
        CONFIG.nvidia_api_key.as_str(),
        CONFIG.openai_api_key.as_str(),
        CONFIG.jina_ai_api_key.as_str(),
        CONFIG.brave_search_api_key.as_str(),
        CONFIG.exa_api_key.as_str(),
        CONFIG.cwd_pw_api_key.as_str(),
        CONFIG.telegraph_access_token.as_str(),
    ];

    for secret in secrets {
        let secret = secret.trim();
        if !secret.is_empty() {
            redacted = redacted.replace(secret, "[REDACTED]");
        }
    }

    redacted
}

fn append_log_tail(report: &mut String, base_name: &str, title: &str, max_lines: usize) {
    report.push_str(&format!("\n{title}\n"));
    match read_recent_log_lines(base_name, max_lines) {
        Ok(Some(tail)) => {
            report.push_str(&format!("source: {}\n", tail.path.display()));
            if tail.lines.is_empty() {
                report.push_str("(no lines available)\n");
            } else {
                for line in tail.lines {
                    let line = redact_sensitive_text(&line);
                    report.push_str(&line);
                    report.push('\n');
                }
            }
        }
        Ok(None) => {
            report.push_str("No matching log files found.\n");
        }
        Err(err) => {
            report.push_str(&format!("Failed to read log tail: {err}\n"));
        }
    }
}

async fn build_status_report(state: &AppState) -> String {
    let db_result = state.db.health_check().await;
    let db_status = if db_result.is_ok() { "ok" } else { "error" };
    let db_detail = db_result.err().map(|err| err.to_string());

    let queue_max = state.db.queue_max_capacity();
    let queue_pending = state.db.queue_len();
    let queue_available = state.db.queue_available_capacity();

    let brave_ready = CONFIG.enable_brave_search && !CONFIG.brave_search_api_key.trim().is_empty();
    let exa_ready = CONFIG.enable_exa_search && !CONFIG.exa_api_key.trim().is_empty();
    let jina_ready = CONFIG.enable_jina_mcp;
    let openrouter_ready =
        CONFIG.is_third_party_provider_ready(crate::config::ThirdPartyProvider::OpenRouter);
    let nvidia_ready =
        CONFIG.is_third_party_provider_ready(crate::config::ThirdPartyProvider::Nvidia);
    let openai_ready =
        CONFIG.is_third_party_provider_ready(crate::config::ThirdPartyProvider::OpenAI);
    let codex_auth = openai_codex::auth_summary();
    let codex_selected_model = selected_codex_model_record();
    let codex_ready = crate::llm::runtime_models::is_runtime_provider_ready(
        crate::config::ThirdPartyProvider::OpenAICodex,
    );
    let active_codex_login = state.active_codex_login.lock().clone();

    let whitelist_path = Path::new(&CONFIG.whitelist_file_path);
    let whitelist_ready = whitelist_path.exists();
    let logs_ready = Path::new("logs").exists();

    let mut report = String::new();
    report.push_str("Status snapshot\n");
    report.push_str(&format!("time_utc: {}\n", Utc::now().to_rfc3339()));
    report.push_str(&format!("db: {db_status}\n"));
    if let Some(detail) = db_detail {
        report.push_str(&format!("db_error: {}\n", detail));
    }
    report.push_str(&format!(
        "db_queue: pending={} available={} max={}\n",
        queue_pending, queue_available, queue_max
    ));
    report.push_str(&format!(
        "gemini_configured: {}\n",
        bool_label(!CONFIG.gemini_api_key.trim().is_empty())
    ));
    report.push_str(&format!(
        "openrouter_ready: {}\n",
        bool_label(openrouter_ready)
    ));
    report.push_str(&format!("nvidia_ready: {}\n", bool_label(nvidia_ready)));
    report.push_str(&format!("openai_ready: {}\n", bool_label(openai_ready)));
    report.push_str(&format!(
        "openai_codex_ready: {}\n",
        bool_label(codex_ready)
    ));
    report.push_str(&format!(
        "openai_codex_auth_file: {}\n",
        CONFIG.openai_codex_auth_path
    ));
    report.push_str(&format!(
        "openai_codex_auth_present: {}\n",
        bool_label(codex_auth.auth_file_exists)
    ));
    if let Some(auth_mode) = codex_auth.auth_mode {
        report.push_str(&format!("openai_codex_auth_mode: {}\n", auth_mode));
    }
    if let Some(plan_type) = codex_auth.plan_type {
        report.push_str(&format!("openai_codex_plan_type: {}\n", plan_type));
    }
    if let Some(account_id) = codex_auth.account_id {
        report.push_str(&format!("openai_codex_account_id: {}\n", account_id));
    }
    if let Some(email) = codex_auth.email {
        report.push_str(&format!("openai_codex_email: {}\n", email));
    }
    if let Some(last_refresh) = codex_auth.last_refresh {
        report.push_str(&format!(
            "openai_codex_last_refresh: {}\n",
            last_refresh.to_rfc3339()
        ));
    }
    report.push_str(&format!(
        "openai_codex_model_file: {}\n",
        CONFIG.openai_codex_model_path
    ));
    report.push_str(&format!(
        "openai_codex_client_version: {}\n",
        CONFIG.openai_codex_client_version
    ));
    report.push_str(&format!(
        "openai_codex_web_search_mode: {}\n",
        CONFIG.openai_codex_web_search_mode
    ));
    if !CONFIG
        .openai_codex_web_search_context_size
        .trim()
        .is_empty()
    {
        report.push_str(&format!(
            "openai_codex_web_search_context_size: {}\n",
            CONFIG.openai_codex_web_search_context_size
        ));
    }
    if !CONFIG.openai_codex_web_search_allowed_domains.is_empty() {
        report.push_str(&format!(
            "openai_codex_web_search_allowed_domains: {}\n",
            CONFIG.openai_codex_web_search_allowed_domains.join(", ")
        ));
    }
    if let Some(model) = codex_selected_model {
        report.push_str(&format!(
            "openai_codex_selected_model: {} ({})\n",
            model.display_name, model.slug
        ));
        report.push_str(&format!(
            "openai_codex_selected_model_supports_native_search: {}\n",
            bool_label(model.supports_search_tool)
        ));
        if let Some(level) = model.selected_reasoning_level {
            report.push_str(&format!("openai_codex_reasoning_override: {}\n", level));
        } else if let Some(level) = model.default_reasoning_level {
            report.push_str(&format!("openai_codex_reasoning_default: {}\n", level));
        }
    }
    report.push_str(&format!(
        "openai_codex_login_pending: {}\n",
        bool_label(active_codex_login.is_some())
    ));
    if let Some(login) = active_codex_login {
        report.push_str(&format!(
            "openai_codex_login_user_id: {}\n",
            login.admin_user_id
        ));
        report.push_str(&format!("openai_codex_login_chat_id: {}\n", login.chat_id));
        report.push_str(&format!(
            "openai_codex_login_started_at: {}\n",
            login.started_at
        ));
        report.push_str(&format!(
            "openai_codex_login_status_message_id: {}\n",
            login.status_message_id
        ));
    }
    report.push_str(&format!(
        "third_party_models_config_path: {}\n",
        CONFIG.third_party_models_config_path.display()
    ));
    report.push_str(&format!(
        "third_party_models_count: {}\n",
        runtime_model_count()
    ));
    report.push_str(&format!(
        "web_search_enabled: {}\n",
        bool_label(is_search_enabled())
    ));
    report.push_str(&format!(
        "web_search_providers_order: {}\n",
        CONFIG.web_search_providers.join(", ")
    ));
    report.push_str(&format!("brave_ready: {}\n", bool_label(brave_ready)));
    report.push_str(&format!("exa_ready: {}\n", bool_label(exa_ready)));
    report.push_str(&format!("jina_ready: {}\n", bool_label(jina_ready)));
    report.push_str(&format!("whitelist_file: {}\n", CONFIG.whitelist_file_path));
    report.push_str(&format!(
        "whitelist_present: {}\n",
        bool_label(whitelist_ready)
    ));
    report.push_str(&format!("logs_dir_present: {}\n", bool_label(logs_ready)));
    report
}

async fn build_diagnose_report(state: &AppState) -> String {
    let mut report = String::new();
    report.push_str("Diagnosis report\n");
    report.push_str("Use /status for a compact health view.\n\n");

    let status = build_status_report(state).await;
    report.push_str(&status);

    report.push_str("\n\nConfig checks\n");
    report.push_str(&format!(
        "BOT_TOKEN_present: {}\n",
        bool_label(!CONFIG.bot_token.trim().is_empty())
    ));
    report.push_str(&format!(
        "GEMINI_API_KEY_present: {}\n",
        bool_label(!CONFIG.gemini_api_key.trim().is_empty())
    ));
    report.push_str(&format!(
        "OPENROUTER_API_KEY_present: {}\n",
        bool_label(!CONFIG.openrouter_api_key.trim().is_empty())
    ));
    report.push_str(&format!(
        "NVIDIA_API_KEY_present: {}\n",
        bool_label(!CONFIG.nvidia_api_key.trim().is_empty())
    ));
    report.push_str(&format!(
        "OPENAI_API_KEY_present: {}\n",
        bool_label(!CONFIG.openai_api_key.trim().is_empty())
    ));
    report.push_str(&format!(
        "JINA_AI_API_KEY_present: {}\n",
        bool_label(!CONFIG.jina_ai_api_key.trim().is_empty())
    ));
    report.push_str(&format!(
        "BRAVE_SEARCH_API_KEY_present: {}\n",
        bool_label(!CONFIG.brave_search_api_key.trim().is_empty())
    ));
    report.push_str(&format!(
        "EXA_API_KEY_present: {}\n",
        bool_label(!CONFIG.exa_api_key.trim().is_empty())
    ));
    report.push_str(&format!(
        "OPENAI_CODEX_AUTH_FILE_present: {}\n",
        bool_label(Path::new(&CONFIG.openai_codex_auth_path).exists())
    ));
    report.push_str(&format!(
        "OPENAI_CODEX_MODEL_FILE_present: {}\n",
        bool_label(Path::new(&CONFIG.openai_codex_model_path).exists())
    ));

    append_log_tail(
        &mut report,
        "bot.log",
        "Recent bot log lines",
        DIAGNOSE_LOG_TAIL_LINES,
    );
    append_log_tail(
        &mut report,
        "timing.log",
        "Recent timing log lines",
        DIAGNOSE_LOG_TAIL_LINES,
    );

    let report = redact_sensitive_text(&report);
    let (truncated, was_truncated) = truncate_chars(&report, DIAGNOSE_TEXT_LIMIT);
    if was_truncated {
        format!("{truncated}\n\n[truncated to fit Telegram message size]")
    } else {
        truncated
    }
}

async fn build_image_caption(model_name: &str, prompt: &str) -> String {
    let safe_model = escape_html(model_name);
    let base_caption = format!("Generated by {}", safe_model);
    let clean_prompt = if prompt.trim().is_empty() {
        "No prompt provided."
    } else {
        prompt
    };
    let escaped_prompt = escape_html(clean_prompt);
    let mut caption = format!(
        "{} with prompt:\n<pre>{}</pre>",
        base_caption, escaped_prompt
    );
    if caption.chars().count() <= IMAGE_CAPTION_LIMIT {
        return caption;
    }

    if let Some(url) = create_telegraph_page("Image Generation Prompt", clean_prompt).await {
        caption = format!(
            "{} with prompt:\n<a href=\"{}\">View it here</a>",
            base_caption,
            escape_html(&url)
        );
        if caption.chars().count() <= IMAGE_CAPTION_LIMIT {
            return caption;
        }
    }

    let (preview, was_truncated) = truncate_chars(clean_prompt, IMAGE_CAPTION_PROMPT_PREVIEW);
    let prompt_preview = if was_truncated {
        format!("{}...", preview)
    } else {
        preview
    };
    caption = format!(
        "{} with prompt:\n<pre>{}</pre>",
        base_caption,
        escape_html(&prompt_preview)
    );
    if caption.chars().count() <= IMAGE_CAPTION_LIMIT {
        caption
    } else {
        base_caption
    }
}

fn message_has_image(message: &Message) -> bool {
    if message.photo().is_some() {
        return true;
    }

    if let Some(document) = message.document() {
        let mime_is_image = document
            .mime_type
            .as_ref()
            .map(|mime| mime.essence_str().starts_with("image/"))
            .unwrap_or(false);
        let name_is_image = document
            .file_name
            .as_ref()
            .map(|name| {
                let lower = name.to_ascii_lowercase();
                lower.ends_with(".png")
                    || lower.ends_with(".jpg")
                    || lower.ends_with(".jpeg")
                    || lower.ends_with(".webp")
                    || lower.ends_with(".gif")
            })
            .unwrap_or(false);
        if mime_is_image || name_is_image {
            return true;
        }
    }

    if let Some(sticker) = message.sticker() {
        if !sticker.flags.is_animated && !sticker.flags.is_video {
            return true;
        }
    }

    false
}

fn telegram_retryable_error(err: &RequestError) -> bool {
    matches!(
        err,
        RequestError::Network(_) | RequestError::RetryAfter(_) | RequestError::Io(_)
    )
}

async fn send_message_with_retry(
    bot: &Bot,
    chat_id: ChatId,
    text: &str,
    reply_to: Option<MessageId>,
) -> Result<Message> {
    let mut delay = Duration::from_secs_f32(1.5);
    for attempt in 0..VID_TELEGRAM_RETRY_ATTEMPTS {
        let mut request = bot.send_message(chat_id, text.to_string());
        if let Some(reply_to) = reply_to {
            request = request.reply_parameters(ReplyParameters::new(reply_to));
        }
        match request.await {
            Ok(message) => return Ok(message),
            Err(err) => {
                if !telegram_retryable_error(&err) || attempt + 1 == VID_TELEGRAM_RETRY_ATTEMPTS {
                    return Err(err.into());
                }
                warn!("send_message attempt {} failed: {err}", attempt + 1);
                if let RequestError::RetryAfter(wait) = err {
                    tokio::time::sleep(wait.duration()).await;
                } else {
                    tokio::time::sleep(delay).await;
                    delay *= 2;
                }
            }
        }
    }

    unreachable!("send_message retry loop exhausted")
}

async fn edit_message_text_with_retry(
    bot: &Bot,
    chat_id: ChatId,
    message_id: MessageId,
    text: &str,
) -> Result<()> {
    let mut delay = Duration::from_secs_f32(1.5);
    for attempt in 0..VID_TELEGRAM_RETRY_ATTEMPTS {
        match bot
            .edit_message_text(chat_id, message_id, text.to_string())
            .await
        {
            Ok(_) => return Ok(()),
            Err(err) => {
                if !telegram_retryable_error(&err) || attempt + 1 == VID_TELEGRAM_RETRY_ATTEMPTS {
                    return Err(err.into());
                }
                warn!("edit_message_text attempt {} failed: {err}", attempt + 1);
                if let RequestError::RetryAfter(wait) = err {
                    tokio::time::sleep(wait.duration()).await;
                } else {
                    tokio::time::sleep(delay).await;
                    delay *= 2;
                }
            }
        }
    }

    Ok(())
}

async fn send_video_with_retry(
    bot: &Bot,
    chat_id: ChatId,
    video_bytes: &[u8],
    reply_to: Option<MessageId>,
) -> Result<Message> {
    let mut delay = Duration::from_secs_f32(1.5);
    for attempt in 0..VID_TELEGRAM_RETRY_ATTEMPTS {
        let input = InputFile::memory(video_bytes.to_vec()).file_name("video.mp4");
        let mut request = bot.send_video(chat_id, input);
        if let Some(reply_to) = reply_to {
            request = request.reply_parameters(ReplyParameters::new(reply_to));
        }
        match request.await {
            Ok(message) => return Ok(message),
            Err(err) => {
                if !telegram_retryable_error(&err) || attempt + 1 == VID_TELEGRAM_RETRY_ATTEMPTS {
                    return Err(err.into());
                }
                warn!("send_video attempt {} failed: {err}", attempt + 1);
                if let RequestError::RetryAfter(wait) = err {
                    tokio::time::sleep(wait.duration()).await;
                } else {
                    tokio::time::sleep(delay).await;
                    delay *= 2;
                }
            }
        }
    }

    unreachable!("send_video retry loop exhausted")
}

async fn send_audio_file_with_retry(
    bot: &Bot,
    chat_id: ChatId,
    audio_bytes: &[u8],
    mime_type: &str,
    caption: Option<&str>,
    reply_to: Option<MessageId>,
) -> Result<Message> {
    let mut delay = Duration::from_secs_f32(1.5);
    let file_name = audio_file_name_for_mime(mime_type);

    for attempt in 0..VID_TELEGRAM_RETRY_ATTEMPTS {
        let input = InputFile::memory(audio_bytes.to_vec()).file_name(file_name.to_string());
        let send_audio = audio_should_use_send_audio(mime_type);

        let result = if send_audio {
            let mut request = bot.send_audio(chat_id, input);
            if let Some(reply_to) = reply_to {
                request = request.reply_parameters(ReplyParameters::new(reply_to));
            }
            if let Some(caption) = caption.filter(|value| !value.trim().is_empty()) {
                request = request
                    .caption(caption.to_string())
                    .parse_mode(ParseMode::Html);
            }
            request.await
        } else {
            let mut request = bot.send_document(chat_id, input);
            if let Some(reply_to) = reply_to {
                request = request.reply_parameters(ReplyParameters::new(reply_to));
            }
            if let Some(caption) = caption.filter(|value| !value.trim().is_empty()) {
                request = request
                    .caption(caption.to_string())
                    .parse_mode(ParseMode::Html);
            }
            request.await
        };

        match result {
            Ok(message) => return Ok(message),
            Err(err) => {
                if !telegram_retryable_error(&err) || attempt + 1 == VID_TELEGRAM_RETRY_ATTEMPTS {
                    return Err(err.into());
                }
                warn!("send_audio/document attempt {} failed: {err}", attempt + 1);
                if let RequestError::RetryAfter(wait) = err {
                    tokio::time::sleep(wait.duration()).await;
                } else {
                    tokio::time::sleep(delay).await;
                    delay *= 2;
                }
            }
        }
    }

    unreachable!("send_audio/document retry loop exhausted")
}

async fn retry_mysong_llm_step<T, F, Fut>(
    bot: &Bot,
    chat_id: ChatId,
    processing_message_id: MessageId,
    step_name: &str,
    retry_status_template: &str,
    mut action: F,
) -> Result<T>
where
    F: FnMut() -> Fut,
    Fut: Future<Output = Result<T>>,
{
    let mut delay = Duration::from_millis(MYSONG_LLM_RETRY_BASE_DELAY_MS);

    for attempt in 1..=MYSONG_LLM_MAX_ATTEMPTS {
        match action().await {
            Ok(value) => return Ok(value),
            Err(err) => {
                if attempt == MYSONG_LLM_MAX_ATTEMPTS {
                    return Err(err);
                }

                warn!(
                    "mysong {} failed on attempt {}/{}: {}",
                    step_name, attempt, MYSONG_LLM_MAX_ATTEMPTS, err
                );
                let retry_status = retry_status_template
                    .replace("{attempt}", &attempt.to_string())
                    .replace("{max}", &MYSONG_LLM_MAX_ATTEMPTS.to_string());
                let _ = edit_message_text_with_retry(
                    bot,
                    chat_id,
                    processing_message_id,
                    &retry_status,
                )
                .await;
                tokio::time::sleep(delay).await;
                delay *= 2;
            }
        }
    }

    unreachable!("mysong llm retry loop exhausted")
}

async fn prepare_image_request(
    bot: &Bot,
    state: &AppState,
    message: &Message,
    command_prefix: &str,
) -> Result<ImageRequestContext> {
    let original_message_text = message
        .text()
        .map(|value| value.to_string())
        .or_else(|| message.caption().map(|value| value.to_string()))
        .unwrap_or_default();

    let prompt_raw = strip_command_prefix(&original_message_text, command_prefix);
    let mut image_urls = Vec::new();
    let mut seen_file_ids: HashSet<FileId> = HashSet::new();
    let mut telegraph_texts = Vec::new();
    let prompt_entities = message_entities_for_text(message);

    if let Some(media_group_id) = message.media_group_id() {
        let group_items = state
            .media_groups
            .lock()
            .get(media_group_id)
            .cloned()
            .unwrap_or_default();
        for item in group_items {
            if seen_file_ids.insert(item.file_id.clone()) {
                if let Ok(url) = get_file_url(bot, &item.file_id).await {
                    image_urls.push(url);
                }
            }
        }
    }

    if let Some(photo_sizes) = message.photo() {
        if let Some(photo) = photo_sizes.last() {
            if seen_file_ids.insert(photo.file.id.clone()) {
                if let Ok(url) = get_file_url(bot, &photo.file.id).await {
                    image_urls.push(url);
                }
            }
        }
    }

    let (prompt, telegraph_contents) =
        extract_telegraph_urls_and_content(&prompt_raw, prompt_entities.as_deref(), 5).await;
    let (mut prompt, twitter_contents) =
        extract_twitter_urls_and_content(&prompt, prompt_entities.as_deref(), 5).await;
    telegraph_texts.extend(
        telegraph_contents
            .iter()
            .map(|content| content.text_content.clone()),
    );
    telegraph_texts.extend(
        twitter_contents
            .iter()
            .map(|content| content.text_content.clone()),
    );

    if let Some(reply) = message.reply_to_message() {
        let reply_has_images = message_has_image(reply);
        if let Some(media_group_id) = reply.media_group_id() {
            let group_items = state
                .media_groups
                .lock()
                .get(media_group_id)
                .cloned()
                .unwrap_or_default();
            for item in group_items {
                if seen_file_ids.insert(item.file_id.clone()) {
                    if let Ok(url) = get_file_url(bot, &item.file_id).await {
                        image_urls.push(url);
                    }
                }
            }
        }

        if image_urls.is_empty() {
            if let Some(photo_sizes) = reply.photo() {
                if let Some(photo) = photo_sizes.last() {
                    if seen_file_ids.insert(photo.file.id.clone()) {
                        if let Ok(url) = get_file_url(bot, &photo.file.id).await {
                            image_urls.push(url);
                        }
                    }
                }
            }
        }

        let reply_text = reply
            .text()
            .map(|value| value.to_string())
            .or_else(|| reply.caption().map(|value| value.to_string()))
            .unwrap_or_default();
        if !reply_text.trim().is_empty() && !reply_has_images {
            let reply_entities = message_entities_for_text(reply);
            let (reply_text, reply_telegraph) =
                extract_telegraph_urls_and_content(&reply_text, reply_entities.as_deref(), 5).await;
            let (reply_text, reply_twitter) =
                extract_twitter_urls_and_content(&reply_text, reply_entities.as_deref(), 5).await;
            telegraph_texts.extend(
                reply_telegraph
                    .iter()
                    .map(|content| content.text_content.clone()),
            );
            telegraph_texts.extend(
                reply_twitter
                    .iter()
                    .map(|content| content.text_content.clone()),
            );

            if prompt.trim().is_empty() {
                prompt = reply_text;
            } else {
                prompt = format!("{}\n\n{}", reply_text, prompt);
            }
        }
    }

    Ok(ImageRequestContext {
        prompt,
        image_urls,
        telegraph_contents: telegraph_texts,
        original_message_text,
    })
}

fn build_resolution_keyboard(request_key: &str) -> InlineKeyboardMarkup {
    let buttons = IMAGE_RESOLUTION_OPTIONS
        .iter()
        .map(|res| {
            InlineKeyboardButton::callback(
                res.to_string(),
                format!(
                    "{}{}|{}",
                    IMAGE_RESOLUTION_CALLBACK_PREFIX, request_key, res
                ),
            )
        })
        .collect::<Vec<_>>();

    let rows = buttons
        .chunks(3)
        .map(|chunk| chunk.to_vec())
        .collect::<Vec<_>>();
    InlineKeyboardMarkup::new(rows)
}

fn build_aspect_ratio_keyboard(request_key: &str) -> InlineKeyboardMarkup {
    let buttons = IMAGE_ASPECT_RATIO_OPTIONS
        .iter()
        .map(|aspect| {
            InlineKeyboardButton::callback(
                aspect.to_string(),
                format!(
                    "{}{}|{}",
                    IMAGE_ASPECT_RATIO_CALLBACK_PREFIX, request_key, aspect
                ),
            )
        })
        .collect::<Vec<_>>();

    let rows = buttons
        .chunks(3)
        .map(|chunk| chunk.to_vec())
        .collect::<Vec<_>>();
    InlineKeyboardMarkup::new(rows)
}

fn resolve_image_request_settings(
    request: &PendingImageRequest,
    resolution: Option<&str>,
    aspect_ratio: Option<&str>,
) -> (String, String) {
    let final_resolution = resolution
        .or(request.resolution.as_deref())
        .unwrap_or(IMAGE_DEFAULT_RESOLUTION)
        .to_string();
    let final_aspect = aspect_ratio
        .or(request.aspect_ratio.as_deref())
        .unwrap_or(IMAGE_DEFAULT_ASPECT_RATIO)
        .to_string();

    (final_resolution, final_aspect)
}

async fn finalize_image_request(
    bot: &Bot,
    state: &AppState,
    request_key: &str,
    resolution: Option<&str>,
    aspect_ratio: Option<&str>,
) -> Result<()> {
    let request = state.pending_image_requests.lock().remove(request_key);
    let Some(request) = request else {
        return Ok(());
    };

    let (final_resolution, final_aspect) =
        resolve_image_request_settings(&request, resolution, aspect_ratio);

    let mut prompt = request.prompt.clone();
    if !request.telegraph_contents.is_empty() {
        prompt.push_str("\n\nAdditional context:\n");
        for content in &request.telegraph_contents {
            prompt.push_str(content);
            prompt.push('\n');
        }
    }

    let image_config = Some(GeminiImageConfig {
        aspect_ratio: if final_aspect.trim().is_empty() {
            None
        } else {
            Some(final_aspect.clone())
        },
        image_size: if final_resolution.trim().is_empty() {
            None
        } else {
            Some(final_resolution.clone())
        },
    });

    let processing_message_id = MessageId(request.selection_message_id as i32);
    let _ = bot
        .edit_message_text(
            ChatId(request.chat_id),
            processing_message_id,
            format!(
                "Generating your image at {} resolution with {} aspect ratio...",
                final_resolution, final_aspect
            ),
        )
        .await?;
    let _chat_action = start_chat_action_heartbeat(
        bot.clone(),
        ChatId(request.chat_id),
        ChatAction::UploadPhoto,
    );

    let image_result = generate_image_with_gemini(
        &prompt,
        &request.image_urls,
        image_config,
        !CONFIG.cwd_pw_api_key.is_empty(),
    )
    .await;

    let model_name = CONFIG.gemini_image_model.as_str();
    let images = match image_result {
        Ok(images) => images,
        Err(err) => {
            error!(model = model_name, "Image generation failed: {}", err.0);
            let error_text = format!(
                "Sorry, I couldn't generate the image using {}.\n\nError: {}",
                model_name, err.0
            );
            let _ = bot
                .edit_message_text(ChatId(request.chat_id), processing_message_id, error_text)
                .await;
            return Ok(());
        }
    };
    let caption = build_image_caption(model_name, &prompt).await;

    let mut image_iter = images.into_iter();
    if let Some(first_image) = image_iter.next() {
        let media = InputMedia::Photo(
            InputMediaPhoto::new(InputFile::memory(first_image.clone()))
                .caption(caption.clone())
                .parse_mode(ParseMode::Html),
        );
        let edit_result = bot
            .edit_message_media(ChatId(request.chat_id), processing_message_id, media)
            .await;
        if edit_result.is_err() {
            bot.send_photo(ChatId(request.chat_id), InputFile::memory(first_image))
                .reply_parameters(ReplyParameters::new(MessageId(request.message_id as i32)))
                .caption(caption)
                .parse_mode(ParseMode::Html)
                .await?;
        }
    }

    for image in image_iter {
        bot.send_photo(ChatId(request.chat_id), InputFile::memory(image))
            .reply_parameters(ReplyParameters::new(MessageId(request.message_id as i32)))
            .await?;
    }

    Ok(())
}

pub async fn image_selection_callback(
    bot: Bot,
    state: AppState,
    query: CallbackQuery,
) -> Result<()> {
    let _ = bot.answer_callback_query(query.id.clone()).await;
    let Some(data) = &query.data else {
        return Ok(());
    };
    let query_user_id = i64::try_from(query.from.id.0).unwrap_or_default();

    if data.starts_with(IMAGE_RESOLUTION_CALLBACK_PREFIX) {
        let payload = data.trim_start_matches(IMAGE_RESOLUTION_CALLBACK_PREFIX);
        let mut parts = payload.split('|');
        let request_key = parts.next().unwrap_or("");
        let resolution = parts.next().unwrap_or("");
        if !IMAGE_RESOLUTION_OPTIONS.contains(&resolution) {
            return Ok(());
        }

        if let Some(request) = state.pending_image_requests.lock().get_mut(request_key) {
            if request.user_id != query_user_id {
                return Ok(());
            }
            request.resolution = Some(resolution.to_string());
        }

        if let Some(message) = &query.message {
            bot.edit_message_text(
                message.chat().id,
                message.id(),
                format!(
                    "Resolution set to {}. Choose an aspect ratio (default: {}).",
                    resolution, IMAGE_DEFAULT_ASPECT_RATIO
                ),
            )
            .reply_markup(build_aspect_ratio_keyboard(request_key))
            .await?;
        }
        return Ok(());
    }

    if data.starts_with(IMAGE_ASPECT_RATIO_CALLBACK_PREFIX) {
        let payload = data.trim_start_matches(IMAGE_ASPECT_RATIO_CALLBACK_PREFIX);
        let mut parts = payload.split('|');
        let request_key = parts.next().unwrap_or("");
        let aspect = parts.next().unwrap_or("");
        if !IMAGE_ASPECT_RATIO_OPTIONS.contains(&aspect) {
            return Ok(());
        }

        if let Some(request) = state.pending_image_requests.lock().get_mut(request_key) {
            if request.user_id != query_user_id {
                return Ok(());
            }
            request.aspect_ratio = Some(aspect.to_string());
        }

        finalize_image_request(&bot, &state, request_key, None, Some(aspect)).await?;
    }

    Ok(())
}

pub async fn img_handler(
    bot: Bot,
    state: AppState,
    message: Message,
    _prompt: Option<String>,
) -> Result<()> {
    if !check_access_control(&bot, &message, "img").await {
        return Ok(());
    }
    let user_id = message
        .from
        .as_ref()
        .and_then(|user| i64::try_from(user.id.0).ok())
        .unwrap_or_default();
    if is_rate_limited(user_id) {
        bot.send_message(
            message.chat.id,
            "Rate limit exceeded. Please try again later.",
        )
        .reply_parameters(ReplyParameters::new(message.id))
        .await?;
        return Ok(());
    }

    let context = prepare_image_request(&bot, &state, &message, "/img").await?;
    if context.prompt.trim().is_empty() && context.image_urls.is_empty() {
        bot.send_message(
            message.chat.id,
            "Please provide a prompt or reply to an image.",
        )
        .reply_parameters(ReplyParameters::new(message.id))
        .await?;
        return Ok(());
    }

    let processing_message = bot
        .send_message(message.chat.id, "Generating your image...")
        .reply_parameters(ReplyParameters::new(message.id))
        .await?;

    let mut prompt_text = context.prompt.clone();
    if !context.telegraph_contents.is_empty() {
        prompt_text.push_str("\n\nAdditional context:\n");
        for content in &context.telegraph_contents {
            prompt_text.push_str(content);
            prompt_text.push('\n');
        }
    }
    let _chat_action =
        start_chat_action_heartbeat(bot.clone(), message.chat.id, ChatAction::UploadPhoto);

    let image_result = generate_image_with_gemini(
        &prompt_text,
        &context.image_urls,
        None,
        !CONFIG.cwd_pw_api_key.is_empty(),
    )
    .await;

    let model_name = CONFIG.gemini_image_model.as_str();
    let images = match image_result {
        Ok(images) => images,
        Err(err) => {
            error!(model = model_name, "Image generation failed: {}", err.0);
            let error_text = format!(
                "Sorry, I couldn't generate the image using {}.\n\nError: {}",
                model_name, err.0
            );
            let _ = bot
                .edit_message_text(message.chat.id, processing_message.id, error_text)
                .await;
            return Ok(());
        }
    };

    let caption = build_image_caption(model_name, &prompt_text).await;
    let mut image_iter = images.into_iter();
    if let Some(first_image) = image_iter.next() {
        let media = InputMedia::Photo(
            InputMediaPhoto::new(InputFile::memory(first_image.clone()))
                .caption(caption.clone())
                .parse_mode(ParseMode::Html),
        );
        let edit_result = bot
            .edit_message_media(message.chat.id, processing_message.id, media)
            .await;
        if edit_result.is_err() {
            bot.send_photo(message.chat.id, InputFile::memory(first_image))
                .reply_parameters(ReplyParameters::new(message.id))
                .caption(caption)
                .parse_mode(ParseMode::Html)
                .await?;
            let _ = bot
                .edit_message_text(
                    message.chat.id,
                    processing_message.id,
                    "Generated image below.",
                )
                .await;
        }
    }

    for image in image_iter {
        bot.send_photo(message.chat.id, InputFile::memory(image))
            .reply_parameters(ReplyParameters::new(message.id))
            .await?;
    }

    Ok(())
}

pub async fn image_handler(
    bot: Bot,
    state: AppState,
    message: Message,
    _prompt: Option<String>,
) -> Result<()> {
    if !check_access_control(&bot, &message, "image").await {
        return Ok(());
    }

    let user_id = message
        .from
        .as_ref()
        .and_then(|user| i64::try_from(user.id.0).ok())
        .unwrap_or_default();
    if is_rate_limited(user_id) {
        bot.send_message(
            message.chat.id,
            "Rate limit exceeded. Please try again later.",
        )
        .reply_parameters(ReplyParameters::new(message.id))
        .await?;
        return Ok(());
    }

    let context = prepare_image_request(&bot, &state, &message, "/image").await?;
    if context.prompt.trim().is_empty() && context.image_urls.is_empty() {
        bot.send_message(
            message.chat.id,
            "Please provide a prompt or reply to an image.",
        )
        .reply_parameters(ReplyParameters::new(message.id))
        .await?;
        return Ok(());
    }

    let request_key = format!("{}_{}", message.chat.id.0, message.id.0);
    let selection_message = bot
        .send_message(message.chat.id, "Choose a resolution (default: 2K):")
        .reply_parameters(ReplyParameters::new(message.id))
        .reply_markup(build_resolution_keyboard(&request_key))
        .await?;
    let pending = PendingImageRequest {
        user_id,
        chat_id: message.chat.id.0,
        message_id: message.id.0 as i64,
        prompt: context.prompt,
        image_urls: context.image_urls,
        telegraph_contents: context.telegraph_contents,
        original_message_text: context.original_message_text,
        selection_message_id: selection_message.id.0 as i64,
        resolution: None,
        aspect_ratio: None,
    };

    state
        .pending_image_requests
        .lock()
        .insert(request_key.clone(), pending);
    let bot_clone = bot.clone();
    let state_clone = state.clone();
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_secs(CONFIG.model_selection_timeout)).await;
        let request = state_clone
            .pending_image_requests
            .lock()
            .get(&request_key)
            .cloned();
        if let Some(request) = request {
            if request.resolution.is_none() {
                let _ = finalize_image_request(
                    &bot_clone,
                    &state_clone,
                    &request_key,
                    Some(IMAGE_DEFAULT_RESOLUTION),
                    Some(IMAGE_DEFAULT_ASPECT_RATIO),
                )
                .await;
            }
        }
    });

    Ok(())
}

pub async fn vid_handler(bot: Bot, message: Message, prompt: Option<String>) -> Result<()> {
    if !check_access_control(&bot, &message, "vid").await {
        return Ok(());
    }

    let user_id = message
        .from
        .as_ref()
        .and_then(|user| i64::try_from(user.id.0).ok())
        .unwrap_or_default();
    if is_rate_limited(user_id) {
        bot.send_message(
            message.chat.id,
            "Rate limit exceeded. Please try again later.",
        )
        .reply_parameters(ReplyParameters::new(message.id))
        .await?;
        return Ok(());
    }

    let reply_has_image = message
        .reply_to_message()
        .map(message_has_image)
        .unwrap_or(false);
    if message_has_image(&message) || reply_has_image {
        send_message_with_retry(
            &bot,
            message.chat.id,
            "Image input isn't supported for /vid right now. Please send a text-only prompt.\nUsage: /vid [text prompt]",
            Some(message.id),
        )
        .await?;
        return Ok(());
    }

    let original_message_text = message
        .text()
        .map(|value| value.to_string())
        .or_else(|| message.caption().map(|value| value.to_string()))
        .unwrap_or_default();

    let prompt_text =
        prompt.unwrap_or_else(|| strip_command_prefix(&original_message_text, "/vid"));
    if prompt_text.trim().is_empty() {
        bot.send_message(
            message.chat.id,
            "Please provide a prompt for the video.\nUsage: /vid [text prompt]",
        )
        .reply_parameters(ReplyParameters::new(message.id))
        .await?;
        return Ok(());
    }

    let processing_message = send_message_with_retry(
        &bot,
        message.chat.id,
        "Processing video request... This may take a few minutes.",
        Some(message.id),
    )
    .await?;
    let _chat_action =
        start_chat_action_heartbeat(bot.clone(), message.chat.id, ChatAction::Typing);
    let (video_bytes, _mime_type) = generate_video_with_veo(&prompt_text).await?;

    if let Some(video_bytes) = video_bytes {
        send_video_with_retry(&bot, message.chat.id, &video_bytes, Some(message.id)).await?;
    } else {
        edit_message_text_with_retry(
            &bot,
            message.chat.id,
            processing_message.id,
            "Video generation is unavailable right now.",
        )
        .await?;
    }

    Ok(())
}

#[allow(deprecated)]
pub async fn tldr_handler(
    bot: Bot,
    state: AppState,
    message: Message,
    count: Option<String>,
) -> Result<()> {
    if !check_access_control(&bot, &message, "tldr").await {
        return Ok(());
    }

    let user_id = message
        .from
        .as_ref()
        .and_then(|user| i64::try_from(user.id.0).ok())
        .unwrap_or_default();
    if is_rate_limited(user_id) {
        bot.send_message(
            message.chat.id,
            "Rate limit exceeded. Please try again later.",
        )
        .reply_parameters(ReplyParameters::new(message.id))
        .await?;
        return Ok(());
    }

    let mut timer = start_command_timer("tldr", &message);
    let processing_message = bot
        .send_message(message.chat.id, "Summarizing recent messages...")
        .reply_parameters(ReplyParameters::new(message.id))
        .await?;
    let _chat_action =
        start_chat_action_heartbeat(bot.clone(), message.chat.id, ChatAction::Typing);

    let messages = if let Some(reply) = message.reply_to_message() {
        state
            .db
            .select_messages_from_id(message.chat.id.0, reply.id.0 as i64)
            .await?
    } else {
        let n = count
            .as_ref()
            .and_then(|value| value.trim().parse::<i64>().ok())
            .unwrap_or(100);
        state.db.select_messages(message.chat.id.0, n).await?
    };

    if messages.is_empty() {
        bot.edit_message_text(
            message.chat.id,
            processing_message.id,
            "No messages found to summarize.",
        )
        .await?;
        complete_command_timer(&mut timer, "error", Some("no_messages".to_string()));
        return Ok(());
    }

    let chat_content = super::format_tldr_chat_content(&messages);

    let system_prompt = TLDR_SYSTEM_PROMPT.replace("{bot_name}", &CONFIG.telegraph_author_name);
    let response = match call_gemini(
        &system_prompt,
        &chat_content,
        true,
        false,
        Some(&CONFIG.gemini_thinking_level),
        None,
        true,
        None,
        None,
        Some("TLDR_SYSTEM_PROMPT"),
    )
    .await
    {
        Ok(response) => response,
        Err(err) => {
            error!("TLDR summary generation failed: {}", err);
            bot.edit_message_text(
                processing_message.chat.id,
                processing_message.id,
                "Failed to generate a summary. Please try again later.",
            )
            .await?;
            complete_command_timer(
                &mut timer,
                "error",
                Some("summary_generation_failed".to_string()),
            );
            return Ok(());
        }
    };

    if response.text.trim().is_empty() {
        bot.edit_message_text(
            processing_message.chat.id,
            processing_message.id,
            "Failed to generate a summary. Please try again later.",
        )
        .await?;
        complete_command_timer(&mut timer, "error", Some("empty_summary".to_string()));
        return Ok(());
    }

    let summary_text = response.text;
    let summary_model = response.model_used;
    let summary_with_model = format!("{}\n\nModel: {}", summary_text, summary_model);

    let _ = bot
        .edit_message_text(
            processing_message.chat.id,
            processing_message.id,
            "Summary generated. Generating infographic...",
        )
        .await;

    let infographic_prompt = format!(
        "Create a clear infographic (no walls of text) summarizing the key points below. \
Use a 16:9 layout with readable labels and visual hierarchy suitable for Telegram. \
Use the same language as the summary text for any labels.\
\n\n{}",
        summary_text
    );

    let mut infographic_url = None;
    let infographic_config = Some(GeminiImageConfig {
        aspect_ratio: Some("16:9".to_string()),
        image_size: Some("4K".to_string()),
    });
    match generate_image_with_gemini(&infographic_prompt, &[], infographic_config, false).await {
        Ok(images) => {
            if let Some(image) = images.into_iter().next() {
                if CONFIG.cwd_pw_api_key.trim().is_empty() {
                    warn!("TLDR infographic generated but CWD_PW_API_KEY is not configured.");
                } else {
                    let mime_type =
                        detect_mime_type(&image).unwrap_or_else(|| "image/png".to_string());
                    infographic_url = upload_image_bytes_to_cwd(
                        &image,
                        &CONFIG.cwd_pw_api_key,
                        &mime_type,
                        Some(CONFIG.gemini_image_model.as_str()),
                        Some(&infographic_prompt),
                    )
                    .await;
                    if infographic_url.is_none() {
                        warn!("Failed to upload TLDR infographic to cwd.pw.");
                    }
                }
            } else {
                warn!("TLDR infographic generation returned no image.");
            }
        }
        Err(err) => {
            error!("Error generating TLDR infographic: {}", err);
        }
    }

    let mut telegraph_url = None;
    if let Some(url) = &infographic_url {
        let telegraph_content = format!(
            "![Infographic]({})\n\n{}\n\nModel: {}",
            url, summary_text, summary_model
        );
        telegraph_url =
            create_telegraph_page("Message Summary with Infographic", &telegraph_content).await;
    }

    let final_message = if let Some(url) = telegraph_url {
        format!(
            "Chat summary with infographic: [View it here]({})\n\nModel: {}",
            url, summary_model
        )
    } else if let Some(url) = infographic_url {
        format!("{}\n\nInfographic: {}", summary_with_model, url)
    } else {
        summary_with_model
    };

    let _ = bot
        .edit_message_text(
            processing_message.chat.id,
            processing_message.id,
            "Infographic step completed. Finalizing response...",
        )
        .await;

    send_response(
        &bot,
        processing_message.chat.id,
        processing_message.id,
        &final_message,
        "Message Summary",
        ParseMode::Markdown,
    )
    .await?;
    complete_command_timer(&mut timer, "success", None);

    Ok(())
}

#[allow(deprecated)]
pub async fn factcheck_handler(
    bot: Bot,
    state: AppState,
    message: Message,
    query: Option<String>,
) -> Result<()> {
    if !check_access_control(&bot, &message, "factcheck").await {
        return Ok(());
    }

    let user_id = message
        .from
        .as_ref()
        .and_then(|user| i64::try_from(user.id.0).ok())
        .unwrap_or_default();
    if is_rate_limited(user_id) {
        bot.send_message(
            message.chat.id,
            "Rate limit exceeded. Please try again later.",
        )
        .reply_parameters(ReplyParameters::new(message.id))
        .await?;
        return Ok(());
    }

    let reply_message = message.reply_to_message();
    let mut query_text = query.unwrap_or_default();
    let query_entities = message_entities_for_text(&message);
    let user_language_code = message
        .from
        .as_ref()
        .and_then(|user| user.language_code.as_deref());
    let mut telegraph_contents = Vec::new();
    let mut twitter_contents = Vec::new();

    let mut reply_text = String::new();
    if let Some(reply) = reply_message {
        reply_text = reply
            .text()
            .map(|value| value.to_string())
            .or_else(|| reply.caption().map(|value| value.to_string()))
            .unwrap_or_default();
        if !reply_text.trim().is_empty() {
            let reply_entities = message_entities_for_text(reply);
            let (reply_text_processed, reply_telegraph) =
                extract_telegraph_urls_and_content(&reply_text, reply_entities.as_deref(), 5).await;
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

    let mut media_options = MediaCollectionOptions::for_commands();
    media_options.include_reply = true;
    let max_files = media_options.max_files;
    let collected_media = collect_message_media(&bot, &state, &message, media_options).await;
    let mut media_files = collected_media.files;

    let mut remaining = max_files.saturating_sub(media_files.len());
    if remaining > 0 {
        let telegraph_files =
            crate::handlers::content::download_telegraph_media(&telegraph_contents, remaining)
                .await;
        remaining = remaining.saturating_sub(telegraph_files.len());
        media_files.extend(telegraph_files);
    }

    if remaining > 0 {
        let twitter_files =
            crate::handlers::content::download_twitter_media(&twitter_contents, remaining).await;
        media_files.extend(twitter_files);
    }

    let media_summary = summarize_media_files(&media_files);
    let statement = build_factcheck_statement(&query_text, &reply_text, &media_summary);

    if statement.trim().is_empty() {
        bot.send_message(message.chat.id, "Please reply to a message to fact-check.")
            .reply_parameters(ReplyParameters::new(message.id))
            .await?;
        return Ok(());
    }

    let mut processing_message_text = if media_summary.videos > 0 {
        "Analyzing video and fact-checking content...".to_string()
    } else if media_summary.audios > 0 {
        "Analyzing audio and fact-checking content...".to_string()
    } else if media_summary.images > 0 {
        format!(
            "Analyzing {} image(s) and fact-checking content...",
            media_summary.images
        )
    } else if media_summary.documents > 0 {
        format!(
            "Analyzing {} document(s) and fact-checking content...",
            media_summary.documents
        )
    } else {
        "Fact-checking message...".to_string()
    };

    if !telegraph_contents.is_empty() {
        let image_count: usize = telegraph_contents
            .iter()
            .map(|content| content.image_urls.len())
            .sum();
        let video_count: usize = telegraph_contents
            .iter()
            .map(|content| content.video_urls.len())
            .sum();
        let mut media_info = String::new();
        if image_count > 0 {
            media_info.push_str(&format!(" with {} image(s)", image_count));
        }
        if video_count > 0 {
            if media_info.is_empty() {
                media_info.push_str(&format!(" with {} video(s)", video_count));
            } else {
                media_info.push_str(&format!(" and {} video(s)", video_count));
            }
        }

        if processing_message_text == "Fact-checking message..." {
            processing_message_text = format!(
                "Extracting and fact-checking content from {} Telegraph page(s){}...",
                telegraph_contents.len(),
                media_info
            );
        } else {
            let base = processing_message_text.trim_end_matches("...");
            processing_message_text = format!(
                "{} and {} Telegraph page(s){}...",
                base,
                telegraph_contents.len(),
                media_info
            );
        }
    }

    if !twitter_contents.is_empty() {
        let image_count: usize = twitter_contents
            .iter()
            .map(|content| content.image_urls.len())
            .sum();
        let video_count: usize = twitter_contents
            .iter()
            .map(|content| content.video_urls.len())
            .sum();
        let mut media_info = String::new();
        if image_count > 0 {
            media_info.push_str(&format!(" with {} image(s)", image_count));
        }
        if video_count > 0 {
            if media_info.is_empty() {
                media_info.push_str(&format!(" with {} video(s)", video_count));
            } else {
                media_info.push_str(&format!(" and {} video(s)", video_count));
            }
        }

        if processing_message_text == "Fact-checking message..." {
            processing_message_text = format!(
                "Extracting and fact-checking content from {} Twitter post(s){}...",
                twitter_contents.len(),
                media_info
            );
        } else {
            let base = processing_message_text.trim_end_matches("...");
            processing_message_text = format!(
                "{} and {} Twitter post(s){}...",
                base,
                twitter_contents.len(),
                media_info
            );
        }
    }

    let processing_message = bot
        .send_message(message.chat.id, processing_message_text)
        .reply_parameters(ReplyParameters::new(message.id))
        .await?;
    let _chat_action =
        start_chat_action_heartbeat(bot.clone(), message.chat.id, ChatAction::Typing);
    let system_prompt = build_factcheck_system_prompt(user_language_code);
    let response = match call_gemini(
        &system_prompt,
        &statement,
        true,
        false,
        Some(&CONFIG.gemini_thinking_level),
        None,
        media_summary.total > 0,
        Some(media_files),
        None,
        Some("FACTCHECK_SYSTEM_PROMPT"),
    )
    .await
    {
        Ok(response) => response,
        Err(err) => {
            error!("Fact-check generation failed: {}", err);
            bot.edit_message_text(
                processing_message.chat.id,
                processing_message.id,
                "Failed to fact-check this message. Please try again later.",
            )
            .await?;
            return Ok(());
        }
    };

    let response_with_model = format!("{}\n\nModel: {}", response.text, response.model_used);

    send_response(
        &bot,
        processing_message.chat.id,
        processing_message.id,
        &response_with_model,
        "Fact Check",
        ParseMode::Markdown,
    )
    .await?;

    Ok(())
}

#[allow(deprecated)]
pub async fn profileme_handler(
    bot: Bot,
    state: AppState,
    message: Message,
    style: Option<String>,
) -> Result<()> {
    if !check_access_control(&bot, &message, "profileme").await {
        return Ok(());
    }

    let user_id = message
        .from
        .as_ref()
        .and_then(|user| i64::try_from(user.id.0).ok())
        .unwrap_or_default();
    if is_rate_limited(user_id) {
        bot.send_message(
            message.chat.id,
            "Rate limit exceeded. Please try again later.",
        )
        .reply_parameters(ReplyParameters::new(message.id))
        .await?;
        return Ok(());
    }

    let processing_message = bot
        .send_message(message.chat.id, "Generating your profile...")
        .reply_parameters(ReplyParameters::new(message.id))
        .await?;
    let _chat_action =
        start_chat_action_heartbeat(bot.clone(), message.chat.id, ChatAction::Typing);
    let history = state
        .db
        .select_messages_by_user(
            message.chat.id.0,
            user_id,
            CONFIG.user_history_message_count,
            true,
        )
        .await?;

    if history.is_empty() {
        bot.edit_message_text(
            message.chat.id,
            processing_message.id,
            "I don't have enough of your messages in this chat yet.",
        )
        .await?;
        return Ok(());
    }

    let mut formatted_history =
        String::from("Here is the user's recent chat history in this group:\n\n");
    for msg in history {
        let timestamp = msg.date.format("%Y-%m-%d %H:%M:%S");
        let text = msg.text.unwrap_or_default();
        formatted_history.push_str(&format!("{}: {}\n", timestamp, text));
    }

    let system_prompt = if let Some(style) = style.filter(|value| !value.trim().is_empty()) {
        format!(
            "{}\n\nStyle Instruction: {}",
            PROFILEME_SYSTEM_PROMPT,
            style.trim()
        )
    } else {
        format!(
            "{}\n\nStyle Instruction: Keep the profile professional, friendly and respectful.",
            PROFILEME_SYSTEM_PROMPT
        )
    };

    let response = call_gemini(
        &system_prompt,
        &formatted_history,
        false,
        false,
        Some(&CONFIG.gemini_thinking_level),
        None,
        false,
        None,
        None,
        Some("PROFILEME_SYSTEM_PROMPT"),
    )
    .await?;

    send_response(
        &bot,
        processing_message.chat.id,
        processing_message.id,
        &response.text,
        "Your User Profile",
        ParseMode::Markdown,
    )
    .await?;

    Ok(())
}

#[allow(deprecated)]
pub async fn mysong_handler(
    bot: Bot,
    state: AppState,
    message: Message,
    note: Option<String>,
) -> Result<()> {
    if !check_access_control(&bot, &message, "mysong").await {
        return Ok(());
    }

    let user_id = message
        .from
        .as_ref()
        .and_then(|user| i64::try_from(user.id.0).ok())
        .unwrap_or_default();
    if is_rate_limited(user_id) {
        bot.send_message(
            message.chat.id,
            "Rate limit exceeded. Please try again later.",
        )
        .reply_parameters(ReplyParameters::new(message.id))
        .await?;
        return Ok(());
    }

    let mut timer = start_command_timer("mysong", &message);
    let processing_message = send_message_with_retry(
        &bot,
        message.chat.id,
        "Composing your theme song... This can take a little while.",
        Some(message.id),
    )
    .await?;

    let result: Result<()> = async {
        let _chat_action =
            start_chat_action_heartbeat(bot.clone(), message.chat.id, ChatAction::Typing);

        let history = state
            .db
            .select_messages_by_user(
                message.chat.id.0,
                user_id,
                CONFIG.user_history_message_count,
                true,
            )
            .await?;

        if history.is_empty() {
            edit_message_text_with_retry(
                &bot,
                message.chat.id,
                processing_message.id,
                "I don't have enough of your messages in this chat yet.",
            )
            .await?;
            complete_command_timer(&mut timer, "error", Some("no_history".to_string()));
            return Ok(());
        }

        let formatted_history = format_user_history_for_persona(&history);
        let language_selection = resolve_mysong_language(note.as_deref());

        let persona_summary = retry_mysong_llm_step(
            &bot,
            message.chat.id,
            processing_message.id,
            "persona summary generation",
            "Summarizing your chat style failed, retrying ({attempt}/{max})...",
            || async {
                call_gemini(
                    MYSONG_SUMMARY_SYSTEM_PROMPT,
                    &formatted_history,
                    false,
                    false,
                    Some(&CONFIG.gemini_thinking_level),
                    None,
                    false,
                    None,
                    None,
                    Some("MYSONG_SUMMARY_SYSTEM_PROMPT"),
                )
                .await
            },
        )
        .await?;

        edit_message_text_with_retry(
            &bot,
            message.chat.id,
            processing_message.id,
            "Writing the final song prompt...",
        )
        .await?;

        let prompt_request = build_mysong_prompt_request(
            &persona_summary.text,
            note.as_deref(),
            language_selection.target_language,
        );
        let lyria_prompt = retry_mysong_llm_step(
            &bot,
            message.chat.id,
            processing_message.id,
            "final prompt generation",
            "Writing the final song prompt failed, retrying ({attempt}/{max})...",
            || async {
                call_gemini(
                    MYSONG_PROMPT_SYSTEM_PROMPT,
                    &prompt_request,
                    false,
                    false,
                    Some(&CONFIG.gemini_thinking_level),
                    None,
                    true,
                    None,
                    None,
                    Some("MYSONG_PROMPT_SYSTEM_PROMPT"),
                )
                .await
            },
        )
        .await?
        .text;

        edit_message_text_with_retry(
            &bot,
            message.chat.id,
            processing_message.id,
            "Generating your song with Lyria 3 Pro...",
        )
        .await?;

        let song = retry_mysong_llm_step(
            &bot,
            message.chat.id,
            processing_message.id,
            "Lyria song generation",
            "Generating your song failed, retrying ({attempt}/{max})...",
            || async { generate_music_with_lyria(&lyria_prompt).await },
        )
        .await?;

        edit_message_text_with_retry(
            &bot,
            message.chat.id,
            processing_message.id,
            "Sending your song and lyrics...",
        )
        .await?;

        let _upload_chat_action =
            start_chat_action_heartbeat(bot.clone(), message.chat.id, ChatAction::UploadDocument);
        let lyrics_message = build_mysong_lyrics_message(
            &song.lyrics_text,
            song.notes_text.as_deref(),
            language_selection.fallback_notice.as_deref(),
        );
        let audio_caption = build_mysong_audio_caption(
            &lyrics_message,
            &song.model_used,
            language_selection.target_language,
        )
        .await;
        send_audio_file_with_retry(
            &bot,
            message.chat.id,
            &song.audio_bytes,
            &song.audio_mime_type,
            Some(&audio_caption),
            Some(message.id),
        )
        .await?;

        let _ = bot
            .delete_message(processing_message.chat.id, processing_message.id)
            .await;

        complete_command_timer(
            &mut timer,
            "success",
            Some(format!(
                "model={} language={}",
                song.model_used, language_selection.target_language
            )),
        );
        Ok(())
    }
    .await;

    if let Err(err) = result {
        complete_command_timer(&mut timer, "error", Some(err.to_string()));
        error!("mysong generation failed: {err}");
        let _ = edit_message_text_with_retry(
            &bot,
            message.chat.id,
            processing_message.id,
            "Failed to generate your theme song. Please try again later.",
        )
        .await;
    }

    Ok(())
}

pub async fn paintme_handler(
    bot: Bot,
    state: AppState,
    message: Message,
    portrait: bool,
) -> Result<()> {
    if !check_access_control(&bot, &message, "paintme").await {
        return Ok(());
    }

    let user_id = message
        .from
        .as_ref()
        .and_then(|user| i64::try_from(user.id.0).ok())
        .unwrap_or_default();
    if is_rate_limited(user_id) {
        bot.send_message(
            message.chat.id,
            "Rate limit exceeded. Please try again later.",
        )
        .reply_parameters(ReplyParameters::new(message.id))
        .await?;
        return Ok(());
    }

    let processing_message = bot
        .send_message(message.chat.id, "Creating your image prompt...")
        .reply_parameters(ReplyParameters::new(message.id))
        .await?;
    let typing_chat_action =
        start_chat_action_heartbeat(bot.clone(), message.chat.id, ChatAction::Typing);
    let history = state
        .db
        .select_messages_by_user(
            message.chat.id.0,
            user_id,
            CONFIG.user_history_message_count,
            true,
        )
        .await?;

    if history.is_empty() {
        bot.edit_message_text(
            message.chat.id,
            processing_message.id,
            "I don't have enough of your messages in this chat yet.",
        )
        .await?;
        return Ok(());
    }

    let mut formatted_history =
        String::from("Here is the user's recent chat history in this group:\n\n");
    for msg in history {
        let timestamp = msg.date.format("%Y-%m-%d %H:%M:%S");
        let text = msg.text.unwrap_or_default();
        formatted_history.push_str(&format!("{}: {}\n", timestamp, text));
    }

    let prompt_system = if portrait {
        PORTRAIT_SYSTEM_PROMPT
    } else {
        PAINTME_SYSTEM_PROMPT
    };

    let prompt = call_gemini(
        prompt_system,
        &formatted_history,
        false,
        false,
        Some(&CONFIG.gemini_thinking_level),
        None,
        false,
        None,
        None,
        Some(if portrait {
            "PORTRAIT_SYSTEM_PROMPT"
        } else {
            "PAINTME_SYSTEM_PROMPT"
        }),
    )
    .await?
    .text;
    drop(typing_chat_action);

    let status_text = if portrait {
        "Generating your portrait..."
    } else {
        "Generating your image..."
    };
    let _ = bot
        .edit_message_text(message.chat.id, processing_message.id, status_text)
        .await;
    let _photo_chat_action =
        start_chat_action_heartbeat(bot.clone(), message.chat.id, ChatAction::UploadPhoto);

    let image_result =
        generate_image_with_gemini(&prompt, &[], None, !CONFIG.cwd_pw_api_key.is_empty()).await;

    let model_name = CONFIG.gemini_image_model.as_str();
    let images = match image_result {
        Ok(images) => images,
        Err(err) => {
            error!(model = model_name, "Image generation failed: {}", err.0);
            let error_text = format!(
                "Sorry, I couldn't generate the image using {}.\n\nError: {}",
                model_name, err.0
            );
            let _ = bot
                .edit_message_text(message.chat.id, processing_message.id, error_text)
                .await;
            return Ok(());
        }
    };
    let caption = build_image_caption(model_name, &prompt).await;

    let mut image_iter = images.into_iter();
    if let Some(first_image) = image_iter.next() {
        let media = InputMedia::Photo(
            InputMediaPhoto::new(InputFile::memory(first_image.clone()))
                .caption(caption.clone())
                .parse_mode(ParseMode::Html),
        );
        let edit_result = bot
            .edit_message_media(message.chat.id, processing_message.id, media)
            .await;
        if edit_result.is_err() {
            bot.send_photo(message.chat.id, InputFile::memory(first_image))
                .reply_parameters(ReplyParameters::new(message.id))
                .caption(caption)
                .parse_mode(ParseMode::Html)
                .await?;
            let _ = bot
                .edit_message_text(
                    message.chat.id,
                    processing_message.id,
                    "Generated image below.",
                )
                .await;
        }
    }

    for image in image_iter {
        bot.send_photo(message.chat.id, InputFile::memory(image))
            .reply_parameters(ReplyParameters::new(message.id))
            .await?;
    }

    Ok(())
}

#[allow(deprecated)]
pub async fn help_handler(bot: Bot, message: Message) -> Result<()> {
    if !check_access_control(&bot, &message, "help").await {
        return Ok(());
    }

    let help_text = r#"
*TelegramGroupHelperBot 指令说明*

/tldr - 汇总最近 N 条消息
用法：回复一条消息后发送 `/tldr`，会汇总从那条消息到现在的聊天内容。
也可以直接使用 `/tldr 50` 指定汇总最近 50 条消息。

/factcheck - 对文字、图片、视频、音频消息做事实核查
用法：`/factcheck [要核查的内容]`
或回复一条消息后发送 `/factcheck`

/q - 提问或分析媒体内容
用法：`/q [你的问题]`

/qc - 询问本群历史内容
用法：`/qc [你的问题]`

/qq - Quick Question（快问快答）
用法：`/qq [你的问题]`

/s - 搜索本群相关消息并返回直达链接
用法：`/s [搜索关键词]`

/img - 用 Gemini 生成或编辑图片
用法：`/img [描述]` 用于生成新图片
或回复一张图片后发送 `/img [描述]` 来编辑图片

/image - 与 /img 相同，但会让你选择分辨率和长宽比
用法：`/image [描述]`，然后选择分辨率（2K/4K/1K）和长宽比

/vid - 用 Veo 生成视频
用法：`/vid [文本提示词]`

/profileme - 基于你在本群的聊天记录生成个人简介
用法：`/profileme`
或：`/profileme [简介风格说明]`

/mysong - 基于你在本群的聊天记录生成你的主题歌
用法：`/mysong`
或：`/mysong [风格、语言或额外要求]`

/paintme - 基于你在本群的聊天记录生成艺术形象
用法：`/paintme`

/portraitme - 基于你在本群的聊天记录生成肖像
用法：`/portraitme`

/status - 查看机器人状态（仅管理员）
用法：`/status`

/diagnose - 查看扩展诊断信息与最近日志（仅管理员）
用法：`/diagnose`

/support - 查看投喂信息
用法：`/support`

/help - 查看这份帮助说明
/codexlogin - log in to ChatGPT Codex (admin only)
Usage: `/codexlogin`

/codexlogout - log out from ChatGPT Codex (admin only)
Usage: `/codexlogout`

/codexmodel - choose the active Codex model (admin only)
Usage: `/codexmodel`

/codexreasoning - choose the active Codex reasoning level (admin only)
Usage: `/codexreasoning`

/codexusage - show current Codex usage/rate limits (admin only)
Usage: `/codexusage`

"#;

    bot.send_message(message.chat.id, help_text)
        .reply_parameters(ReplyParameters::new(message.id))
        .parse_mode(ParseMode::Markdown)
        .await?;
    Ok(())
}

pub async fn status_handler(bot: Bot, state: AppState, message: Message) -> Result<()> {
    if !check_admin_access(&bot, &message, "status").await {
        return Ok(());
    }

    let report = build_status_report(&state).await;
    bot.send_message(message.chat.id, report)
        .reply_parameters(ReplyParameters::new(message.id))
        .await?;
    Ok(())
}

pub async fn diagnose_handler(bot: Bot, state: AppState, message: Message) -> Result<()> {
    if !check_admin_access(&bot, &message, "diagnose").await {
        return Ok(());
    }

    let report = build_diagnose_report(&state).await;
    bot.send_message(message.chat.id, report)
        .reply_parameters(ReplyParameters::new(message.id))
        .await?;
    Ok(())
}

#[allow(deprecated)]
pub async fn support_handler(bot: Bot, message: Message) -> Result<()> {
    if !check_access_control(&bot, &message, "support").await {
        return Ok(());
    }

    let support_url = match reqwest::Url::parse(CONFIG.support_link.trim()) {
        Ok(url) => url,
        Err(_) => {
            bot.send_message(message.chat.id, CONFIG.support_message.clone())
                .reply_parameters(ReplyParameters::new(message.id))
                .parse_mode(ParseMode::Markdown)
                .await?;
            return Ok(());
        }
    };

    let keyboard = InlineKeyboardMarkup::new(vec![vec![InlineKeyboardButton::url(
        "Support the bot",
        support_url,
    )]]);

    bot.send_message(message.chat.id, CONFIG.support_message.clone())
        .reply_parameters(ReplyParameters::new(message.id))
        .reply_markup(keyboard)
        .parse_mode(ParseMode::Markdown)
        .await?;
    Ok(())
}

pub async fn start_handler(bot: Bot, message: Message) -> Result<()> {
    bot.send_message(
        message.chat.id,
        "Hello! I am TelegramGroupHelperBot. Use /help to see commands.",
    )
    .reply_parameters(ReplyParameters::new(message.id))
    .await?;
    Ok(())
}

pub async fn handle_media_group(state: AppState, message: Message) {
    if let Some(media_group_id) = message.media_group_id() {
        if let Some(photo_sizes) = message.photo() {
            if let Some(photo) = photo_sizes.last() {
                let mut groups = state.media_groups.lock();
                let entry = groups.entry(media_group_id.clone()).or_default();
                entry.push(MediaGroupItem {
                    file_id: photo.file.id.clone(),
                });
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_mysong_language_defaults_to_english() {
        let selection = resolve_mysong_language(None);

        assert_eq!(selection.target_language, "English");
        assert_eq!(selection.fallback_notice, None);
    }

    #[test]
    fn resolve_mysong_language_uses_supported_override() {
        let selection = resolve_mysong_language(Some("make it dreamy and sing it in Japanese"));

        assert_eq!(selection.target_language, "Japanese");
        assert_eq!(selection.fallback_notice, None);
    }

    #[test]
    fn resolve_mysong_language_detects_japanese_in_chinese_text() {
        let selection = resolve_mysong_language(Some("90年代日本动漫风格，日语歌"));

        assert_eq!(selection.target_language, "Japanese");
        assert_eq!(selection.fallback_notice, None);
    }

    #[test]
    fn resolve_mysong_language_falls_back_for_unsupported_request() {
        let selection = resolve_mysong_language(Some("please sing it in Chinese"));

        assert_eq!(selection.target_language, "English");
        assert_eq!(
            selection.fallback_notice.as_deref(),
            Some(
                "Lyria 3 currently does not support Chinese lyrics here, so I generated the song in English instead."
            )
        );
    }

    #[test]
    fn resolve_image_request_settings_prefers_saved_resolution_and_aspect_ratio() {
        let request = PendingImageRequest {
            user_id: 1,
            chat_id: 2,
            message_id: 3,
            prompt: "test".to_string(),
            image_urls: Vec::new(),
            telegraph_contents: Vec::new(),
            original_message_text: "test".to_string(),
            selection_message_id: 4,
            resolution: Some("4K".to_string()),
            aspect_ratio: Some("16:9".to_string()),
        };

        let (final_resolution, final_aspect) =
            resolve_image_request_settings(&request, None, Some("1:1"));

        assert_eq!(final_resolution, "4K");
        assert_eq!(final_aspect, "1:1");
    }
}
