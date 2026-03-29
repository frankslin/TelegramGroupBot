use std::collections::HashMap;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use parking_lot::Mutex;
use teloxide::types::{FileId, MediaGroupId};

use crate::db::database::Database;
use crate::llm::media::MediaFile;
use crate::llm::openai_codex::{CodexReasoningEffortOption, CodexRemoteModel};
use crate::utils::timing::CommandTimer;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QaCommandMode {
    Standard,
    ChatContext,
}

impl QaCommandMode {
    pub fn requires_custom_tools(self) -> bool {
        matches!(self, Self::ChatContext)
    }
}

#[allow(dead_code)]
#[derive(Debug)]
pub struct PendingQRequest {
    pub user_id: i64,
    pub username: String,
    pub query: String,
    pub original_query: String,
    pub db_query_text: String,
    pub telegram_language_code: Option<String>,
    pub media_files: Vec<MediaFile>,
    pub youtube_urls: Vec<String>,
    pub telegraph_contents: Vec<String>,
    pub twitter_contents: Vec<String>,
    pub chat_id: i64,
    pub message_id: i64,
    pub selection_message_id: i64,
    pub original_user_id: i64,
    pub reply_to_message_id: Option<i64>,
    pub timestamp: i64,
    pub command_timer: Option<CommandTimer>,
    pub mode: QaCommandMode,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct PendingImageRequest {
    pub user_id: i64,
    pub chat_id: i64,
    pub message_id: i64,
    pub prompt: String,
    pub image_urls: Vec<String>,
    pub telegraph_contents: Vec<String>,
    pub original_message_text: String,
    pub selection_message_id: i64,
    pub resolution: Option<String>,
    pub aspect_ratio: Option<String>,
}

#[derive(Debug, Clone)]
pub struct PendingCodexModelRequest {
    pub admin_user_id: i64,
    pub chat_id: i64,
    pub selection_message_id: i64,
    pub timestamp: i64,
    pub page: usize,
    pub etag: Option<String>,
    pub models: Vec<CodexRemoteModel>,
}

#[derive(Debug, Clone)]
pub struct PendingCodexReasoningRequest {
    pub admin_user_id: i64,
    pub chat_id: i64,
    pub selection_message_id: i64,
    pub timestamp: i64,
    pub default_level: Option<String>,
    pub supported_levels: Vec<CodexReasoningEffortOption>,
}

#[derive(Debug, Clone)]
pub struct ActiveCodexLogin {
    pub admin_user_id: i64,
    pub chat_id: i64,
    pub status_message_id: i64,
    pub verification_url: String,
    pub user_code: String,
    pub started_at: i64,
    pub cancel_flag: Arc<AtomicBool>,
}

#[derive(Debug, Clone)]
pub struct MediaGroupItem {
    pub file_id: FileId,
}

#[derive(Clone)]
pub struct AppState {
    pub db: Database,
    pub bot_user_id: i64,
    pub bot_username_lower: String,
    pub pending_q_requests: Arc<Mutex<HashMap<String, PendingQRequest>>>,
    pub pending_image_requests: Arc<Mutex<HashMap<String, PendingImageRequest>>>,
    pub pending_codex_model_requests: Arc<Mutex<HashMap<String, PendingCodexModelRequest>>>,
    pub pending_codex_reasoning_requests: Arc<Mutex<HashMap<String, PendingCodexReasoningRequest>>>,
    pub active_codex_login: Arc<Mutex<Option<ActiveCodexLogin>>>,
    pub media_groups: Arc<Mutex<HashMap<MediaGroupId, Vec<MediaGroupItem>>>>,
}

impl AppState {
    pub fn new(db: Database, bot_user_id: i64, bot_username_lower: String) -> Self {
        AppState {
            db,
            bot_user_id,
            bot_username_lower,
            pending_q_requests: Arc::new(Mutex::new(HashMap::new())),
            pending_image_requests: Arc::new(Mutex::new(HashMap::new())),
            pending_codex_model_requests: Arc::new(Mutex::new(HashMap::new())),
            pending_codex_reasoning_requests: Arc::new(Mutex::new(HashMap::new())),
            active_codex_login: Arc::new(Mutex::new(None)),
            media_groups: Arc::new(Mutex::new(HashMap::new())),
        }
    }
}
