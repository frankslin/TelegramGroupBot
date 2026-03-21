use crate::db::models::{ChatSearchHit, MessageInsert, MessageRow};
use crate::utils::telegram::build_message_link;
use anyhow::{anyhow, Result};
use once_cell::sync::Lazy;
use regex::Regex;
use sqlx::sqlite::SqlitePoolOptions;
use sqlx::{FromRow, SqlitePool};
use tokio::sync::mpsc;
use tracing::{info, warn};

const SEARCH_LIMIT_MAX: i64 = 20;
const SEARCH_OFFSET_MAX: i64 = 250;
const WINDOW_LIMIT_MAX: i64 = 5;
const SEARCH_TERM_LIMIT: usize = 8;
const SEARCH_TERM_MAX_CHARS: usize = 48;
const SNIPPET_LIMIT: usize = 140;

static SEARCH_TERM_REGEX: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"[\p{L}\p{N}_]+").expect("valid search term regex"));

#[derive(Clone)]
pub struct Database {
    pool: SqlitePool,
    sender: mpsc::Sender<MessageInsert>,
}

#[derive(Debug, Clone, FromRow)]
struct SearchRow {
    id: i64,
    message_id: i64,
    chat_id: i64,
    user_id: Option<i64>,
    username: Option<String>,
    text: Option<String>,
    language: Option<String>,
    date: chrono::DateTime<chrono::Utc>,
    reply_to_message_id: Option<i64>,
    score: f64,
}

fn truncate_chars(value: &str, max_chars: usize) -> String {
    let char_count = value.chars().count();
    if char_count <= max_chars {
        return value.to_string();
    }

    let mut truncated: String = value.chars().take(max_chars).collect();
    truncated.push_str("...");
    truncated
}

fn extract_search_terms(query: &str) -> Vec<String> {
    SEARCH_TERM_REGEX
        .find_iter(query)
        .map(|capture| capture.as_str().trim().to_lowercase())
        .filter(|term| !term.is_empty())
        .map(|term| truncate_chars(&term, SEARCH_TERM_MAX_CHARS))
        .take(SEARCH_TERM_LIMIT)
        .collect()
}

fn build_fts_query_from_terms(terms: &[String]) -> Option<String> {
    if terms.is_empty() {
        return None;
    }

    let query = terms
        .iter()
        .map(|term| {
            if term.chars().count() >= 2 {
                format!("{term}*")
            } else {
                term.clone()
            }
        })
        .collect::<Vec<_>>()
        .join(" OR ");

    (!query.trim().is_empty()).then_some(query)
}

#[cfg_attr(not(test), allow(dead_code))]
pub fn sanitize_chat_search_query(query: &str) -> Option<String> {
    let terms = extract_search_terms(query);
    build_fts_query_from_terms(&terms)
}

fn find_snippet_offset(text: &str, terms: &[String]) -> usize {
    let lower = text.to_lowercase();
    terms
        .iter()
        .filter_map(|term| lower.find(term))
        .min()
        .unwrap_or(0)
}

fn build_snippet(text: &str, terms: &[String]) -> String {
    let normalized = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if normalized.is_empty() {
        return String::new();
    }

    if normalized.chars().count() <= SNIPPET_LIMIT {
        return normalized;
    }

    let start = find_snippet_offset(&normalized, terms);
    let prefix_char_count = normalized[..start.min(normalized.len())].chars().count();
    let snippet_start = prefix_char_count.saturating_sub(SNIPPET_LIMIT / 3);
    let snippet_body: String = normalized
        .chars()
        .skip(snippet_start)
        .take(SNIPPET_LIMIT)
        .collect();
    let mut snippet = snippet_body.trim().to_string();

    if snippet_start > 0 {
        snippet.insert_str(0, "...");
    }
    if snippet_start + snippet_body.chars().count() < normalized.chars().count() {
        snippet.push_str("...");
    }

    snippet
}

impl Database {
    pub async fn init(database_url: &str) -> Result<Self> {
        let pool = SqlitePoolOptions::new()
            .max_connections(5)
            .connect(database_url)
            .await?;

        sqlx::query(
            "CREATE TABLE IF NOT EXISTS messages (\
                id INTEGER PRIMARY KEY AUTOINCREMENT,\
                message_id INTEGER NOT NULL,\
                chat_id INTEGER NOT NULL,\
                user_id INTEGER,\
                username TEXT,\
                text TEXT,\
                language TEXT,\
                date TEXT NOT NULL,\
                reply_to_message_id INTEGER,\
                UNIQUE(chat_id, message_id)\
            );",
        )
        .execute(&pool)
        .await?;

        sqlx::query("CREATE INDEX IF NOT EXISTS idx_messages_chat_id ON messages(chat_id);")
            .execute(&pool)
            .await?;
        sqlx::query("CREATE INDEX IF NOT EXISTS idx_messages_message_id ON messages(message_id);")
            .execute(&pool)
            .await?;
        sqlx::query("CREATE INDEX IF NOT EXISTS idx_messages_date ON messages(date);")
            .execute(&pool)
            .await?;

        sqlx::query("CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(text, tokenize = 'unicode61');")
            .execute(&pool)
            .await?;
        sqlx::query(
            "CREATE TRIGGER IF NOT EXISTS messages_ai AFTER INSERT ON messages \
             WHEN NEW.text IS NOT NULL BEGIN \
             INSERT INTO messages_fts(rowid, text) VALUES (NEW.id, NEW.text); \
             END;",
        )
        .execute(&pool)
        .await?;
        sqlx::query(
            "CREATE TRIGGER IF NOT EXISTS messages_ad AFTER DELETE ON messages BEGIN \
             DELETE FROM messages_fts WHERE rowid = OLD.id; \
             END;",
        )
        .execute(&pool)
        .await?;
        sqlx::query(
            "CREATE TRIGGER IF NOT EXISTS messages_au AFTER UPDATE ON messages BEGIN \
             DELETE FROM messages_fts WHERE rowid = OLD.id; \
             INSERT INTO messages_fts(rowid, text) \
             SELECT NEW.id, NEW.text WHERE NEW.text IS NOT NULL; \
             END;",
        )
        .execute(&pool)
        .await?;
        sqlx::query(
            "INSERT INTO messages_fts(rowid, text) \
             SELECT messages.id, messages.text \
             FROM messages \
             LEFT JOIN messages_fts ON messages_fts.rowid = messages.id \
             WHERE messages.text IS NOT NULL AND messages_fts.rowid IS NULL;",
        )
        .execute(&pool)
        .await?;
        sqlx::query(
            "DELETE FROM messages_fts \
             WHERE rowid IN ( \
                 SELECT messages_fts.rowid \
                 FROM messages_fts \
                 LEFT JOIN messages ON messages.id = messages_fts.rowid \
                 WHERE messages.id IS NULL OR messages.text IS NULL \
             );",
        )
        .execute(&pool)
        .await?;

        info!("Database tables created successfully");

        let (sender, receiver) = mpsc::channel(1000);
        let writer_pool = pool.clone();
        tokio::spawn(async move {
            db_writer(writer_pool, receiver).await;
        });

        info!("Database writer task started");

        Ok(Database { pool, sender })
    }

    pub async fn queue_message_insert(&self, insert: MessageInsert) -> Result<()> {
        self.sender
            .send(insert)
            .await
            .map_err(|err| anyhow!("Failed to queue message insert: {err}"))
    }

    pub async fn health_check(&self) -> Result<()> {
        sqlx::query("SELECT 1").execute(&self.pool).await?;
        Ok(())
    }

    pub fn queue_max_capacity(&self) -> usize {
        self.sender.max_capacity()
    }

    pub fn queue_available_capacity(&self) -> usize {
        self.sender.capacity()
    }

    pub fn queue_len(&self) -> usize {
        self.queue_max_capacity()
            .saturating_sub(self.queue_available_capacity())
    }

    pub async fn select_messages(&self, chat_id: i64, limit: i64) -> Result<Vec<MessageRow>> {
        self.get_last_n_text_messages(chat_id, limit, true).await
    }

    pub async fn select_messages_by_user(
        &self,
        chat_id: i64,
        user_id: i64,
        limit: i64,
        exclude_commands: bool,
    ) -> Result<Vec<MessageRow>> {
        let mut query = String::from(
            "SELECT id, message_id, chat_id, user_id, username, text, language, date, reply_to_message_id \
             FROM messages WHERE chat_id = ? AND user_id = ? AND text IS NOT NULL",
        );
        if exclude_commands {
            query.push_str(" AND text NOT LIKE '/%'");
        }
        query.push_str(" ORDER BY date DESC LIMIT ?");

        let rows = sqlx::query_as::<_, MessageRow>(&query)
            .bind(chat_id)
            .bind(user_id)
            .bind(limit)
            .fetch_all(&self.pool)
            .await?;

        Ok(rows.into_iter().rev().collect())
    }

    pub async fn select_messages_from_id(
        &self,
        chat_id: i64,
        message_id: i64,
    ) -> Result<Vec<MessageRow>> {
        self.get_messages_from_id(chat_id, message_id, true).await
    }

    pub async fn get_last_n_text_messages(
        &self,
        chat_id: i64,
        limit: i64,
        exclude_commands: bool,
    ) -> Result<Vec<MessageRow>> {
        let mut query = String::from(
            "SELECT id, message_id, chat_id, user_id, username, text, language, date, reply_to_message_id \
             FROM messages WHERE chat_id = ? AND text IS NOT NULL",
        );
        if exclude_commands {
            query.push_str(" AND text NOT LIKE '/%'");
        }
        query.push_str(" ORDER BY date DESC LIMIT ?");

        let rows = sqlx::query_as::<_, MessageRow>(&query)
            .bind(chat_id)
            .bind(limit)
            .fetch_all(&self.pool)
            .await?;

        Ok(rows.into_iter().rev().collect())
    }

    pub async fn get_messages_from_id(
        &self,
        chat_id: i64,
        from_message_id: i64,
        exclude_commands: bool,
    ) -> Result<Vec<MessageRow>> {
        let mut query = String::from(
            "SELECT id, message_id, chat_id, user_id, username, text, language, date, reply_to_message_id \
             FROM messages WHERE chat_id = ? AND message_id >= ? AND text IS NOT NULL",
        );
        if exclude_commands {
            query.push_str(" AND text NOT LIKE '/%'");
        }
        query.push_str(" ORDER BY date DESC");

        let rows = sqlx::query_as::<_, MessageRow>(&query)
            .bind(chat_id)
            .bind(from_message_id)
            .fetch_all(&self.pool)
            .await?;

        Ok(rows.into_iter().rev().collect())
    }

    pub async fn search_chat_messages(
        &self,
        chat_id: i64,
        query: &str,
        limit: i64,
        offset: i64,
    ) -> Result<Vec<ChatSearchHit>> {
        let terms = extract_search_terms(query);
        let sanitized_query = build_fts_query_from_terms(&terms)
            .ok_or_else(|| anyhow!("query must contain searchable text"))?;
        let limit = limit.clamp(1, SEARCH_LIMIT_MAX);
        let offset = offset.clamp(0, SEARCH_OFFSET_MAX);

        let rows = sqlx::query_as::<_, SearchRow>(
            "SELECT \
                 m.id, \
                 m.message_id, \
                 m.chat_id, \
                 m.user_id, \
                 m.username, \
                 m.text, \
                 m.language, \
                 m.date, \
                 m.reply_to_message_id, \
                 bm25(messages_fts) AS score \
             FROM messages_fts \
             JOIN messages m ON m.id = messages_fts.rowid \
             WHERE m.chat_id = ? AND messages_fts MATCH ? \
             ORDER BY score ASC, m.message_id DESC \
             LIMIT ? OFFSET ?",
        )
        .bind(chat_id)
        .bind(sanitized_query)
        .bind(limit)
        .bind(offset)
        .fetch_all(&self.pool)
        .await?;

        let hits = rows
            .into_iter()
            .filter_map(|row| {
                let _ = row.id;
                let text = row.text.unwrap_or_default();
                if text.trim().is_empty() {
                    return None;
                }
                Some(ChatSearchHit {
                    message_id: row.message_id,
                    chat_id: row.chat_id,
                    user_id: row.user_id,
                    username: row.username,
                    text: text.clone(),
                    language: row.language,
                    date: row.date,
                    reply_to_message_id: row.reply_to_message_id,
                    snippet: build_snippet(&text, &terms),
                    link: build_message_link(row.chat_id, row.message_id),
                    score: row.score,
                })
            })
            .collect();

        Ok(hits)
    }

    pub async fn get_message_window(
        &self,
        chat_id: i64,
        message_id: i64,
        context_before: i64,
        context_after: i64,
    ) -> Result<Option<Vec<MessageRow>>> {
        let context_before = context_before.clamp(0, WINDOW_LIMIT_MAX);
        let context_after = context_after.clamp(0, WINDOW_LIMIT_MAX);

        let center = sqlx::query_as::<_, MessageRow>(
            "SELECT id, message_id, chat_id, user_id, username, text, language, date, reply_to_message_id \
             FROM messages \
             WHERE chat_id = ? AND message_id = ? AND text IS NOT NULL",
        )
        .bind(chat_id)
        .bind(message_id)
        .fetch_optional(&self.pool)
        .await?;

        let Some(center) = center else {
            return Ok(None);
        };

        let mut before = sqlx::query_as::<_, MessageRow>(
            "SELECT id, message_id, chat_id, user_id, username, text, language, date, reply_to_message_id \
             FROM messages \
             WHERE chat_id = ? AND message_id < ? AND text IS NOT NULL \
             ORDER BY message_id DESC LIMIT ?",
        )
        .bind(chat_id)
        .bind(message_id)
        .bind(context_before)
        .fetch_all(&self.pool)
        .await?;
        before.reverse();

        let after = sqlx::query_as::<_, MessageRow>(
            "SELECT id, message_id, chat_id, user_id, username, text, language, date, reply_to_message_id \
             FROM messages \
             WHERE chat_id = ? AND message_id > ? AND text IS NOT NULL \
             ORDER BY message_id ASC LIMIT ?",
        )
        .bind(chat_id)
        .bind(message_id)
        .bind(context_after)
        .fetch_all(&self.pool)
        .await?;

        let mut messages = before;
        messages.push(center);
        messages.extend(after);
        Ok(Some(messages))
    }

    #[allow(dead_code)]
    pub fn pool(&self) -> &SqlitePool {
        &self.pool
    }
}

async fn db_writer(pool: SqlitePool, mut receiver: mpsc::Receiver<MessageInsert>) {
    while let Some(message) = receiver.recv().await {
        let result = sqlx::query(
            "INSERT INTO messages (message_id, chat_id, user_id, username, text, language, date, reply_to_message_id) \
             VALUES (?, ?, ?, ?, ?, ?, ?, ?) \
             ON CONFLICT(chat_id, message_id) DO UPDATE SET \
             user_id = excluded.user_id, \
             username = excluded.username, \
             text = excluded.text, \
             language = excluded.language, \
             date = excluded.date, \
             reply_to_message_id = excluded.reply_to_message_id",
        )
        .bind(message.message_id)
        .bind(message.chat_id)
        .bind(message.user_id)
        .bind(message.username)
        .bind(message.text)
        .bind(message.language)
        .bind(message.date)
        .bind(message.reply_to_message_id)
        .execute(&pool)
        .await;

        if let Err(err) = result {
            warn!("Error in db_writer: {err}");
        }
    }

    let _ = pool.close().await;
    info!("Database writer task stopped");
}

pub fn build_message_insert(
    user_id: Option<i64>,
    username: Option<String>,
    text: Option<String>,
    language: Option<String>,
    date: chrono::DateTime<chrono::Utc>,
    reply_to_message_id: Option<i64>,
    chat_id: Option<i64>,
    message_id: Option<i64>,
) -> MessageInsert {
    let resolved_user_id = user_id.unwrap_or_default();
    let resolved_chat_id = chat_id.unwrap_or(resolved_user_id);
    MessageInsert {
        message_id: message_id.unwrap_or_default(),
        chat_id: resolved_chat_id,
        user_id,
        username,
        text,
        language,
        date,
        reply_to_message_id,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use std::path::PathBuf;

    fn test_db_path(test_name: &str) -> PathBuf {
        let mut path = PathBuf::from("target");
        path.push("test-dbs");
        std::fs::create_dir_all(&path).expect("test db directory should exist");
        path.push(format!(
            "telegram-chat-bot-{}-{}-{}.db",
            test_name,
            std::process::id(),
            Utc::now().timestamp_nanos_opt().unwrap_or_default()
        ));
        let _ = std::fs::File::create(&path).expect("test db file should be creatable");
        path
    }

    fn sqlite_url_for_path(path: &std::path::Path) -> String {
        format!("sqlite://{}", path.to_string_lossy().replace('\\', "/"))
    }

    async fn init_test_db(test_name: &str) -> Database {
        let path = test_db_path(test_name);
        Database::init(&sqlite_url_for_path(&path))
            .await
            .expect("test database should initialize")
    }

    async fn insert_message(
        db: &Database,
        message_id: i64,
        chat_id: i64,
        username: &str,
        text: &str,
    ) {
        sqlx::query(
            "INSERT INTO messages (message_id, chat_id, user_id, username, text, language, date, reply_to_message_id) \
             VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        )
        .bind(message_id)
        .bind(chat_id)
        .bind(123_i64)
        .bind(username)
        .bind(text)
        .bind("en")
        .bind(Utc::now())
        .bind(None::<i64>)
        .execute(db.pool())
        .await
        .expect("message insert should succeed");
    }

    #[test]
    fn sanitize_chat_search_query_removes_unsafe_operators() {
        assert_eq!(
            sanitize_chat_search_query("hello; DROP TABLE messages --"),
            Some("hello* OR drop* OR table* OR messages*".to_string())
        );
    }

    #[tokio::test]
    async fn search_chat_messages_stays_within_the_requested_chat() {
        let db = init_test_db("chat-scope").await;
        insert_message(&db, 1, -1001374348669, "alice", "Bitcoin treasury update").await;
        insert_message(&db, 2, -1001374348669, "bob", "ETH treasury update").await;
        insert_message(&db, 3, -1002631835259, "mallory", "Bitcoin treasury leak").await;

        let hits = db
            .search_chat_messages(-1001374348669, "bitcoin treasury", 10, 0)
            .await
            .expect("search should succeed");

        assert_eq!(hits.len(), 2);
        assert!(hits.iter().all(|hit| hit.chat_id == -1001374348669));
        assert!(hits.iter().all(|hit| hit
            .link
            .as_deref()
            .unwrap_or_default()
            .starts_with("https://t.me/c/1374348669/")));
    }

    #[tokio::test]
    async fn get_message_window_rejects_cross_chat_requests() {
        let db = init_test_db("window-scope").await;
        insert_message(&db, 1, -1001374348669, "alice", "Alpha keyword").await;
        insert_message(&db, 2, -1002631835259, "mallory", "Alpha keyword").await;

        let window = db
            .get_message_window(-1001374348669, 2, 1, 1)
            .await
            .expect("window lookup should succeed");

        assert!(window.is_none());
    }

    #[tokio::test]
    async fn init_backfills_fts_for_existing_rows() {
        let path = test_db_path("fts-backfill");
        let url = sqlite_url_for_path(&path);
        let pool = SqlitePoolOptions::new()
            .max_connections(1)
            .connect(&url)
            .await
            .expect("raw pool should initialize");

        sqlx::query(
            "CREATE TABLE messages (\
                id INTEGER PRIMARY KEY AUTOINCREMENT,\
                message_id INTEGER NOT NULL,\
                chat_id INTEGER NOT NULL,\
                user_id INTEGER,\
                username TEXT,\
                text TEXT,\
                language TEXT,\
                date TEXT NOT NULL,\
                reply_to_message_id INTEGER,\
                UNIQUE(chat_id, message_id)\
            );",
        )
        .execute(&pool)
        .await
        .expect("messages table should exist");
        sqlx::query(
            "INSERT INTO messages (message_id, chat_id, user_id, username, text, language, date, reply_to_message_id) \
             VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        )
        .bind(77_i64)
        .bind(-1001374348669_i64)
        .bind(123_i64)
        .bind("alice")
        .bind("Retroactive FTS backfill works")
        .bind("en")
        .bind(Utc::now())
        .bind(None::<i64>)
        .execute(&pool)
        .await
        .expect("seed message should insert");
        pool.close().await;

        let db = Database::init(&url)
            .await
            .expect("database should initialize");
        let hits = db
            .search_chat_messages(-1001374348669, "retroactive backfill", 10, 0)
            .await
            .expect("search should succeed");

        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].message_id, 77);
    }
}
