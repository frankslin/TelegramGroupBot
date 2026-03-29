use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use anyhow::{Result, anyhow};
use chrono::Utc;
use teloxide::prelude::*;
use teloxide::types::{InlineKeyboardButton, InlineKeyboardMarkup, MessageId, ReplyParameters};

use crate::handlers::access::check_admin_access;
use crate::llm::openai_codex::{self, CodexInputModality, CodexModelVisibility, CodexRemoteModel};
use crate::llm::runtime_models::{self, CodexSelectedModelRecord};
use crate::state::{
    ActiveCodexLogin, AppState, PendingCodexModelRequest, PendingCodexReasoningRequest,
};
use tracing::warn;

pub const CODEX_MODEL_SELECT_CALLBACK_PREFIX: &str = "codex_model_select:";
pub const CODEX_MODEL_PAGE_CALLBACK_PREFIX: &str = "codex_model_page:";
pub const CODEX_REASONING_SELECT_CALLBACK_PREFIX: &str = "codex_reasoning_select:";
const CODEX_MODEL_PAGE_SIZE: usize = 8;

fn now_unix_seconds() -> i64 {
    chrono::Utc::now().timestamp()
}

fn request_key(chat_id: ChatId, message_id: MessageId) -> String {
    format!("{}_{}", chat_id.0, message_id.0)
}

fn format_login_message(verification_url: &str, user_code: &str) -> String {
    format!(
        "Open this URL in your browser and complete ChatGPT sign-in:\n{}\n\nEnter this one-time code:\n{}\n\nThe bot will keep polling in the background. When login finishes, it will send a follow-up message here.",
        verification_url, user_code
    )
}

fn filter_picker_models(mut models: Vec<CodexRemoteModel>) -> Vec<CodexRemoteModel> {
    models.retain(|model| model.visibility == CodexModelVisibility::List);
    models.sort_by_key(|model| model.priority);
    models
}

fn keyboard_model_label(model: &CodexRemoteModel) -> String {
    let label = model.display_name.trim();
    if label.chars().count() <= 28 {
        label.to_string()
    } else {
        let truncated: String = label.chars().take(28).collect();
        format!("{truncated}...")
    }
}

fn modality_summary(model: &CodexRemoteModel) -> String {
    let mut parts = vec!["text".to_string()];
    if model.input_modalities.contains(&CodexInputModality::Image) {
        parts.push("image".to_string());
    }
    parts.join(", ")
}

fn build_model_selection_text(models: &[CodexRemoteModel], page: usize) -> String {
    let total_pages = models.len().div_ceil(CODEX_MODEL_PAGE_SIZE).max(1);
    format!(
        "Select the active Codex model for the bot.\n\nVisible models: {}\nPage {}/{}",
        models.len(),
        page + 1,
        total_pages
    )
}

fn build_model_selection_keyboard(models: &[CodexRemoteModel], page: usize) -> InlineKeyboardMarkup {
    let total_pages = models.len().div_ceil(CODEX_MODEL_PAGE_SIZE).max(1);
    let start = page.saturating_mul(CODEX_MODEL_PAGE_SIZE);
    let end = (start + CODEX_MODEL_PAGE_SIZE).min(models.len());
    let page_models = &models[start..end];

    let mut rows = Vec::new();
    for chunk in page_models.chunks(2) {
        let mut row = Vec::new();
        for model in chunk {
            row.push(InlineKeyboardButton::callback(
                keyboard_model_label(model),
                format!("{}{}", CODEX_MODEL_SELECT_CALLBACK_PREFIX, model.slug),
            ));
        }
        rows.push(row);
    }

    if total_pages > 1 {
        let mut nav = Vec::new();
        if page > 0 {
            nav.push(InlineKeyboardButton::callback(
                "Prev",
                format!("{}{}", CODEX_MODEL_PAGE_CALLBACK_PREFIX, page - 1),
            ));
        }
        if page + 1 < total_pages {
            nav.push(InlineKeyboardButton::callback(
                "Next",
                format!("{}{}", CODEX_MODEL_PAGE_CALLBACK_PREFIX, page + 1),
            ));
        }
        if !nav.is_empty() {
            rows.push(nav);
        }
    }

    InlineKeyboardMarkup::new(rows)
}

fn selected_model_record(
    model: &CodexRemoteModel,
    etag: Option<String>,
) -> CodexSelectedModelRecord {
    let previous_selection = runtime_models::selected_codex_model_record()
        .and_then(|record| record.selected_reasoning_level)
        .filter(|level| {
            model.supported_reasoning_levels
                .iter()
                .any(|option| option.effort == *level)
        });

    CodexSelectedModelRecord {
        slug: model.slug.clone(),
        display_name: model.display_name.clone(),
        description: model.description.clone(),
        input_modalities: model
            .input_modalities
            .iter()
            .map(|modality| match modality {
                CodexInputModality::Text => "text".to_string(),
                CodexInputModality::Image => "image".to_string(),
            })
        .collect(),
        priority: model.priority,
        etag,
        default_reasoning_level: model.default_reasoning_level.clone(),
        supported_reasoning_levels: model.supported_reasoning_levels.clone(),
        selected_reasoning_level: previous_selection,
        fetched_at: Utc::now(),
    }
}

fn build_reasoning_selection_text(
    display_name: &str,
    selected_level: Option<&str>,
    default_level: Option<&str>,
) -> String {
    let selected = selected_level.unwrap_or("backend default");
    let default = default_level.unwrap_or("unknown");
    format!(
        "Select the active Codex reasoning level.\n\nModel: {}\nSelected: {}\nModel default: {}",
        display_name, selected, default
    )
}

fn build_reasoning_selection_keyboard(
    supported_levels: &[openai_codex::CodexReasoningEffortOption],
    selected_level: Option<&str>,
    default_level: Option<&str>,
) -> InlineKeyboardMarkup {
    let mut rows = Vec::new();
    for chunk in supported_levels.chunks(2) {
        let mut row = Vec::new();
        for level in chunk {
            let mut label = level.effort.clone();
            if selected_level == Some(level.effort.as_str()) {
                label.push_str(" *");
            } else if selected_level.is_none() && default_level == Some(level.effort.as_str()) {
                label.push_str(" (default)");
            }
            row.push(InlineKeyboardButton::callback(
                label,
                format!("{}{}", CODEX_REASONING_SELECT_CALLBACK_PREFIX, level.effort),
            ));
        }
        rows.push(row);
    }
    rows.push(vec![InlineKeyboardButton::callback(
        "Use model default",
        format!("{}default", CODEX_REASONING_SELECT_CALLBACK_PREFIX),
    )]);
    InlineKeyboardMarkup::new(rows)
}

async fn handle_model_selection_timeout(bot: Bot, state: AppState, request_id: String) {
    tokio::time::sleep(Duration::from_secs(crate::config::CONFIG.model_selection_timeout)).await;
    let pending = state.pending_codex_model_requests.lock().remove(&request_id);
    let Some(pending) = pending else {
        return;
    };

    let _ = bot
        .edit_message_text(
            ChatId(pending.chat_id),
            MessageId(pending.selection_message_id as i32),
            "Codex model selection timed out. Run /codexmodel again when you want to change it.",
        )
        .reply_markup(InlineKeyboardMarkup::new(
            Vec::<Vec<InlineKeyboardButton>>::new(),
        ))
        .await;
}

pub async fn codex_login_handler(bot: Bot, state: AppState, message: Message) -> Result<()> {
    if !check_admin_access(&bot, &message, "codexlogin").await {
        return Ok(());
    }

    let admin_user_id = message
        .from
        .as_ref()
        .and_then(|user| i64::try_from(user.id.0).ok())
        .unwrap_or_default();

    let existing_login = { state.active_codex_login.lock().clone() };
    if let Some(existing) = existing_login {
        bot.send_message(
            message.chat.id,
            format_login_message(&existing.verification_url, &existing.user_code),
        )
        .reply_parameters(ReplyParameters::new(message.id))
        .await?;
        return Ok(());
    }

    let start = openai_codex::request_device_code().await?;
    let status_message = bot
        .send_message(
            message.chat.id,
            format_login_message(&start.verification_url, &start.user_code),
        )
        .reply_parameters(ReplyParameters::new(message.id))
        .await?;

    let cancel_flag = Arc::new(AtomicBool::new(false));
    {
        let mut login = state.active_codex_login.lock();
        *login = Some(ActiveCodexLogin {
            admin_user_id,
            chat_id: message.chat.id.0,
            status_message_id: status_message.id.0 as i64,
            verification_url: start.verification_url.clone(),
            user_code: start.user_code.clone(),
            started_at: now_unix_seconds(),
            cancel_flag: cancel_flag.clone(),
        });
    }

    let bot_clone = bot.clone();
    let state_clone = state.clone();
    tokio::spawn(async move {
        let result = openai_codex::complete_device_code_login(&start, cancel_flag.clone()).await;

        let active = state_clone.active_codex_login.lock().clone();
        let should_clear = active
            .as_ref()
            .is_some_and(|entry| Arc::ptr_eq(&entry.cancel_flag, &cancel_flag));
        if should_clear {
            *state_clone.active_codex_login.lock() = None;
        }

        if cancel_flag.load(Ordering::SeqCst) {
            return;
        }

        match result {
            Ok(auth) => {
                let plan_type = auth
                    .tokens
                    .as_ref()
                    .and_then(|tokens| tokens.plan_type.as_deref())
                    .unwrap_or("unknown");
                let _ = bot_clone
                    .send_message(
                        ChatId(message.chat.id.0),
                        format!(
                            "Codex ChatGPT login completed.\n\nPlan: {}\nNext step: run /codexmodel to choose the active Codex model.",
                            plan_type
                        ),
                    )
                    .await;
            }
            Err(err) => {
                warn!("Codex device-code login failed: {}", err);
                let _ = bot_clone
                    .send_message(
                        ChatId(message.chat.id.0),
                        format!("Codex login failed: {}", err),
                    )
                    .await;
            }
        }
    });

    Ok(())
}

pub async fn codex_logout_handler(bot: Bot, state: AppState, message: Message) -> Result<()> {
    if !check_admin_access(&bot, &message, "codexlogout").await {
        return Ok(());
    }

    if let Some(active) = state.active_codex_login.lock().take() {
        active.cancel_flag.store(true, Ordering::SeqCst);
    }

    let removed = openai_codex::logout()?;
    let text = if removed {
        "Codex auth credentials were removed."
    } else {
        "Codex auth credentials were already absent."
    };
    bot.send_message(message.chat.id, text)
        .reply_parameters(ReplyParameters::new(message.id))
        .await?;
    Ok(())
}

pub async fn codex_model_handler(bot: Bot, state: AppState, message: Message) -> Result<()> {
    if !check_admin_access(&bot, &message, "codexmodel").await {
        return Ok(());
    }

    if !openai_codex::is_auth_ready() {
        bot.send_message(
            message.chat.id,
            "Codex is not logged in yet. Run /codexlogin first.",
        )
        .reply_parameters(ReplyParameters::new(message.id))
        .await?;
        return Ok(());
    }

    let list = openai_codex::fetch_models().await?;
    let models = filter_picker_models(list.models);
    if models.is_empty() {
        bot.send_message(
            message.chat.id,
            "No picker-visible Codex models were returned for this account.",
        )
        .reply_parameters(ReplyParameters::new(message.id))
        .await?;
        return Ok(());
    }

    let page = 0;
    let selection_message = bot
        .send_message(
            message.chat.id,
            build_model_selection_text(&models, page),
        )
        .reply_parameters(ReplyParameters::new(message.id))
        .reply_markup(build_model_selection_keyboard(&models, page))
        .await?;

    let admin_user_id = message
        .from
        .as_ref()
        .and_then(|user| i64::try_from(user.id.0).ok())
        .unwrap_or_default();
    let request_id = request_key(message.chat.id, selection_message.id);
    state.pending_codex_model_requests.lock().insert(
        request_id.clone(),
        PendingCodexModelRequest {
            admin_user_id,
            chat_id: message.chat.id.0,
            selection_message_id: selection_message.id.0 as i64,
            timestamp: now_unix_seconds(),
            page,
            etag: list.etag,
            models,
        },
    );

    let bot_clone = bot.clone();
    let state_clone = state.clone();
    tokio::spawn(async move {
        handle_model_selection_timeout(bot_clone, state_clone, request_id).await;
    });

    Ok(())
}

pub async fn codex_reasoning_handler(bot: Bot, state: AppState, message: Message) -> Result<()> {
    if !check_admin_access(&bot, &message, "codexreasoning").await {
        return Ok(());
    }

    let Some(mut record) = runtime_models::selected_codex_model_record() else {
        bot.send_message(
            message.chat.id,
            "No Codex model is selected yet. Run /codexmodel first.",
        )
        .reply_parameters(ReplyParameters::new(message.id))
        .await?;
        return Ok(());
    };

    if record.supported_reasoning_levels.is_empty() {
        if let Ok(list) = openai_codex::fetch_models().await {
            if let Some(model) = filter_picker_models(list.models)
                .into_iter()
                .find(|model| model.slug == record.slug)
            {
                let refreshed = selected_model_record(&model, list.etag);
                let _ = runtime_models::save_selected_codex_model(&refreshed)?;
                record = refreshed;
            }
        }
    }

    if record.supported_reasoning_levels.is_empty() {
        bot.send_message(
            message.chat.id,
            "The selected Codex model does not advertise any configurable reasoning levels.",
        )
        .reply_parameters(ReplyParameters::new(message.id))
        .await?;
        return Ok(());
    }

    let text = build_reasoning_selection_text(
        &record.display_name,
        record.selected_reasoning_level.as_deref(),
        record.default_reasoning_level.as_deref(),
    );
    let keyboard = build_reasoning_selection_keyboard(
        &record.supported_reasoning_levels,
        record.selected_reasoning_level.as_deref(),
        record.default_reasoning_level.as_deref(),
    );
    let selection_message = bot
        .send_message(message.chat.id, text)
        .reply_parameters(ReplyParameters::new(message.id))
        .reply_markup(keyboard)
        .await?;

    let admin_user_id = message
        .from
        .as_ref()
        .and_then(|user| i64::try_from(user.id.0).ok())
        .unwrap_or_default();
    let request_id = request_key(message.chat.id, selection_message.id);
    state.pending_codex_reasoning_requests.lock().insert(
        request_id.clone(),
        PendingCodexReasoningRequest {
            admin_user_id,
            chat_id: message.chat.id.0,
            selection_message_id: selection_message.id.0 as i64,
            timestamp: now_unix_seconds(),
            default_level: record.default_reasoning_level.clone(),
            supported_levels: record.supported_reasoning_levels.clone(),
        },
    );

    let bot_clone = bot.clone();
    let state_clone = state.clone();
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_secs(crate::config::CONFIG.model_selection_timeout)).await;
        let pending = state_clone
            .pending_codex_reasoning_requests
            .lock()
            .remove(&request_id);
        let Some(pending) = pending else {
            return;
        };
        let _ = bot_clone
            .edit_message_text(
                ChatId(pending.chat_id),
                MessageId(pending.selection_message_id as i32),
                "Codex reasoning selection timed out. Run /codexreasoning again when you want to change it.",
            )
            .reply_markup(InlineKeyboardMarkup::new(
                Vec::<Vec<InlineKeyboardButton>>::new(),
            ))
            .await;
    });

    Ok(())
}

pub async fn codex_admin_callback(bot: Bot, state: AppState, query: CallbackQuery) -> Result<()> {
    let Some(data) = query.data.as_deref() else {
        return Ok(());
    };
    if !data.starts_with(CODEX_MODEL_SELECT_CALLBACK_PREFIX)
        && !data.starts_with(CODEX_MODEL_PAGE_CALLBACK_PREFIX)
        && !data.starts_with(CODEX_REASONING_SELECT_CALLBACK_PREFIX)
    {
        return Ok(());
    }

    let _ = bot.answer_callback_query(query.id.clone()).await;
    let Some(message) = query.message.clone() else {
        return Ok(());
    };
    let request_id = request_key(message.chat().id, message.id());
    let query_user_id = i64::try_from(query.from.id.0).unwrap_or_default();

    if data.starts_with(CODEX_REASONING_SELECT_CALLBACK_PREFIX) {
        enum ReasoningAction {
            Expired,
            Ignore,
            Apply(Option<String>, Option<String>, Option<String>),
        }

        let action = {
            let mut pending_map = state.pending_codex_reasoning_requests.lock();
            match pending_map.get(&request_id) {
                None => ReasoningAction::Expired,
                Some(pending) if pending.admin_user_id != query_user_id => ReasoningAction::Ignore,
                Some(pending)
                    if now_unix_seconds() - pending.timestamp
                        > crate::config::CONFIG.model_selection_timeout as i64 =>
                {
                    pending_map.remove(&request_id);
                    ReasoningAction::Expired
                }
                Some(pending) => {
                    let raw = data
                        .strip_prefix(CODEX_REASONING_SELECT_CALLBACK_PREFIX)
                        .unwrap_or_default();
                    let level = if raw == "default" {
                        None
                    } else if pending.supported_levels.iter().any(|entry| entry.effort == raw) {
                        Some(raw.to_string())
                    } else {
                        return Err(anyhow!("invalid Codex reasoning selection"));
                    };
                    let selected_level = level.clone();
                    let default_level = pending.default_level.clone();
                    let display_name = runtime_models::selected_codex_model_record()
                        .map(|record| record.display_name);
                    pending_map.remove(&request_id);
                    ReasoningAction::Apply(selected_level, default_level, display_name)
                }
            }
        };

        match action {
            ReasoningAction::Expired => {
                bot.edit_message_text(
                    message.chat().id,
                    message.id(),
                    "This Codex reasoning request has expired.",
                )
                .reply_markup(InlineKeyboardMarkup::new(
                    Vec::<Vec<InlineKeyboardButton>>::new(),
                ))
                .await?;
            }
            ReasoningAction::Ignore => {}
            ReasoningAction::Apply(level, default_level, display_name) => {
                let updated = runtime_models::save_selected_codex_reasoning_level(level.clone())?;
                let effective = level
                    .clone()
                    .or_else(|| default_level.clone())
                    .unwrap_or_else(|| "backend default".to_string());
                bot.edit_message_text(
                    message.chat().id,
                    message.id(),
                    format!(
                        "Codex reasoning updated.\n\nModel: {}\nSelected reasoning: {}\nSaved override: {}",
                        display_name.unwrap_or(updated.display_name),
                        effective,
                        level.unwrap_or_else(|| "none (use model default)".to_string())
                    ),
                )
                .reply_markup(InlineKeyboardMarkup::new(
                    Vec::<Vec<InlineKeyboardButton>>::new(),
                ))
                .await?;
            }
        }

        return Ok(());
    }

    enum CallbackAction {
        Expired,
        Ignore,
        ShowPage {
            text: String,
            keyboard: InlineKeyboardMarkup,
        },
        SelectModel {
            model: CodexRemoteModel,
            etag: Option<String>,
        },
    }

    let action = {
        let mut pending_map = state.pending_codex_model_requests.lock();
        match pending_map.get_mut(&request_id) {
            None => CallbackAction::Expired,
            Some(pending) => {
                if pending.admin_user_id != query_user_id {
                    CallbackAction::Ignore
                } else if now_unix_seconds() - pending.timestamp
                    > crate::config::CONFIG.model_selection_timeout as i64
                {
                    pending_map.remove(&request_id);
                    CallbackAction::Expired
                } else if let Some(page_raw) = data.strip_prefix(CODEX_MODEL_PAGE_CALLBACK_PREFIX) {
                    let page = page_raw.parse::<usize>().unwrap_or(0);
                    pending.page = page.min(
                        pending
                            .models
                            .len()
                            .div_ceil(CODEX_MODEL_PAGE_SIZE)
                            .saturating_sub(1),
                    );
                    CallbackAction::ShowPage {
                        text: build_model_selection_text(&pending.models, pending.page),
                        keyboard: build_model_selection_keyboard(&pending.models, pending.page),
                    }
                } else {
                    let Some(slug) = data.strip_prefix(CODEX_MODEL_SELECT_CALLBACK_PREFIX) else {
                        return Err(anyhow!("invalid Codex model callback payload"));
                    };
                    let model = pending
                        .models
                        .iter()
                        .find(|model| model.slug == slug)
                        .cloned()
                        .ok_or_else(|| anyhow!("selected Codex model is no longer available"))?;
                    let etag = pending.etag.clone();
                    pending_map.remove(&request_id);
                    CallbackAction::SelectModel { model, etag }
                }
            }
        }
    };

    match action {
        CallbackAction::Expired => {
            bot.edit_message_text(
                message.chat().id,
                message.id(),
                "This Codex model request has expired.",
            )
            .reply_markup(InlineKeyboardMarkup::new(
                Vec::<Vec<InlineKeyboardButton>>::new(),
            ))
            .await?;
        }
        CallbackAction::Ignore => {}
        CallbackAction::ShowPage { text, keyboard } => {
            bot.edit_message_text(message.chat().id, message.id(), text)
                .reply_markup(keyboard)
                .await?;
        }
        CallbackAction::SelectModel { model, etag } => {
            let record = selected_model_record(&model, etag);
            let config = runtime_models::save_selected_codex_model(&record)?;
            let summary = modality_summary(&model);
            bot.edit_message_text(
                message.chat().id,
                message.id(),
                format!(
                    "Active Codex model updated.\n\nName: {}\nSlug: {}\nCapabilities: {}\nRuntime alias: {}",
                    model.display_name,
                    model.slug,
                    summary,
                    config.id
                ),
            )
            .reply_markup(InlineKeyboardMarkup::new(
                Vec::<Vec<InlineKeyboardButton>>::new(),
            ))
            .await?;
        }
    }

    Ok(())
}
