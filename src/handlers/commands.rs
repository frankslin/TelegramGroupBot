use std::collections::HashSet;
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
use crate::llm::web_search::is_search_enabled;
use crate::llm::{
    call_gemini, generate_image_with_gemini, generate_video_with_veo, GeminiImageConfig,
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

#[derive(Debug, Clone)]
struct ImageRequestContext {
    prompt: String,
    image_urls: Vec<String>,
    telegraph_contents: Vec<String>,
    original_message_text: String,
}

fn strip_command_prefix(text: &str, command_prefix: &str) -> String {
    if text.starts_with(command_prefix) {
        text[command_prefix.len()..].trim().to_string()
    } else {
        text.to_string()
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
    report.push_str(&format!(
        "third_party_models_config_path: {}\n",
        CONFIG.third_party_models_config_path.display()
    ));
    report.push_str(&format!(
        "third_party_models_count: {}\n",
        CONFIG.iter_third_party_models().len()
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

    let final_resolution = resolution.unwrap_or(IMAGE_DEFAULT_RESOLUTION);
    let final_aspect = aspect_ratio.unwrap_or(IMAGE_DEFAULT_ASPECT_RATIO);

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
            Some(final_aspect.to_string())
        },
        image_size: if final_resolution.trim().is_empty() {
            None
        } else {
            Some(final_resolution.to_string())
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

    let mut chat_content = String::new();
    for msg in messages {
        let timestamp = msg.date.format("%Y-%m-%d %H:%M:%S");
        let username = msg.username.unwrap_or_else(|| "Anonymous".to_string());
        let text = msg.text.unwrap_or_default();
        chat_content.push_str(&format!("{} {}: {}\n", timestamp, username, text));
    }

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

    let help_text = "
*TelegramGroupHelperBot Commands*

/tldr - Summarize previous messages in the chat
Usage: Reply to a message with `/tldr` to summarize all messages between that message and the present.

/factcheck - Fact-check a statement or text
Usage: `/factcheck [statement]` or reply to a message with `/factcheck`

/q - Ask a question
Usage: `/q [your question]`

/qc - Ask a question about this chat using same-chat retrieval
Usage: `/qc [your question]`

/qq - Quick Gemini answer using the default Gemini model
Usage: `/qq [your quick question]`

/s - Search this chat and return relevant message links
Usage: `/s [search query]`

/img - Generate or edit an image using Gemini
Usage: `/img [description]` for generating a new image
Or reply to an image with `/img [description]` to edit that image

/image - Generate or edit an image with resolution and aspect ratio choices
Usage: `/image [description]` and pick resolution (2K/4K/1K) and aspect ratio

/vid - Generate a video
Usage: `/vid [text prompt]`

/profileme - Generate your user profile based on your chat history in this group.
Usage: `/profileme`
Or: `/profileme [Language style of the profile]`

/paintme - Generate an image representing you based on your chat history in this group.
Usage: `/paintme`

/portraitme - Generate a portrait of you based on your chat history in this group.
Usage: `/portraitme`

/status - Show bot health snapshot (admin-only)
Usage: `/status`

/diagnose - Show extended diagnostics with recent log tails (admin-only)
Usage: `/diagnose`

/support - Show support information and Ko-fi link
Usage: `/support`

/help - Show this help message
";

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
