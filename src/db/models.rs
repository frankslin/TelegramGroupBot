use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sqlx::FromRow;

#[derive(Debug, Clone, FromRow, Serialize, Deserialize)]
pub struct MessageRow {
    pub id: i64,
    pub message_id: i64,
    pub chat_id: i64,
    pub user_id: Option<i64>,
    pub username: Option<String>,
    pub text: Option<String>,
    pub language: Option<String>,
    pub date: DateTime<Utc>,
    pub reply_to_message_id: Option<i64>,
}

#[derive(Debug, Clone)]
pub struct MessageInsert {
    pub message_id: i64,
    pub chat_id: i64,
    pub user_id: Option<i64>,
    pub username: Option<String>,
    pub text: Option<String>,
    pub language: Option<String>,
    pub date: DateTime<Utc>,
    pub reply_to_message_id: Option<i64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatSearchHit {
    pub message_id: i64,
    pub chat_id: i64,
    pub user_id: Option<i64>,
    pub username: Option<String>,
    pub text: String,
    pub language: Option<String>,
    pub date: DateTime<Utc>,
    pub reply_to_message_id: Option<i64>,
    pub snippet: String,
    pub link: Option<String>,
    pub score: f64,
}
