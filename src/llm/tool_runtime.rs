use std::collections::BTreeMap;

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::db::database::Database;
use crate::db::models::{ChatSearchHit, MessageRow};
use crate::llm::web_search::{self, web_search_tool};
use crate::utils::telegram::build_message_link;

const DEFAULT_QC_SEARCH_LIMIT: usize = 8;
const DEFAULT_S_SEARCH_LIMIT: usize = 20;
const MAX_SEARCH_LIMIT: usize = 20;
const MAX_SEARCH_OFFSET: usize = 250;
const MAX_CONTEXT_WINDOW: usize = 5;
const MAX_WEB_RESULTS: usize = 10;
const MAX_TOOL_TEXT_CHARS: usize = 1200;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolProfile {
    ChatQuestion,
    ChatSearch,
}

#[derive(Debug, Clone, Copy)]
pub struct ToolBudgetConfig {
    pub max_total_successful_calls: usize,
    pub max_web_search_calls: usize,
    pub max_chat_context_query_calls: usize,
}

#[derive(Debug, Clone, Copy, Serialize)]
struct ToolBudgetSnapshot {
    total_remaining: usize,
    web_search_remaining: usize,
    chat_context_query_remaining: usize,
}

#[derive(Debug, Clone, Copy)]
enum ToolBudgetErrorKind {
    Total,
    WebSearch,
    ChatContextQuery,
    Disabled,
}

#[derive(Debug, Clone, Copy)]
struct ToolBudgetError {
    kind: ToolBudgetErrorKind,
}

#[derive(Clone)]
pub struct ToolRuntime {
    db: Database,
    chat_id: i64,
    profile: ToolProfile,
    budget: ToolBudgetConfig,
    successful_calls: usize,
    web_search_calls: usize,
    chat_context_query_calls: usize,
    force_final_answer: bool,
    accumulated_hits: BTreeMap<i64, ChatSearchHit>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "operation", rename_all = "snake_case")]
enum ChatContextQueryArgs {
    Search {
        query: String,
        limit: Option<usize>,
        offset: Option<usize>,
        context_before: Option<usize>,
        context_after: Option<usize>,
    },
    Window {
        message_id: i64,
        context_before: Option<usize>,
        context_after: Option<usize>,
    },
}

#[derive(Debug, Serialize)]
struct ToolMessage {
    message_id: i64,
    username: Option<String>,
    date_utc: String,
    text: String,
    link: Option<String>,
}

#[derive(Debug, Serialize)]
struct ToolSearchHit {
    message_id: i64,
    username: Option<String>,
    date_utc: String,
    text: String,
    snippet: String,
    link: Option<String>,
    score: f64,
    context_messages: Vec<ToolMessage>,
}

impl ToolRuntime {
    pub fn for_qc(db: Database, chat_id: i64) -> Self {
        Self {
            db,
            chat_id,
            profile: ToolProfile::ChatQuestion,
            budget: ToolBudgetConfig {
                max_total_successful_calls: 8,
                max_web_search_calls: 3,
                max_chat_context_query_calls: 5,
            },
            successful_calls: 0,
            web_search_calls: 0,
            chat_context_query_calls: 0,
            force_final_answer: false,
            accumulated_hits: BTreeMap::new(),
        }
    }

    pub fn for_search(db: Database, chat_id: i64) -> Self {
        Self {
            db,
            chat_id,
            profile: ToolProfile::ChatSearch,
            budget: ToolBudgetConfig {
                max_total_successful_calls: 5,
                max_web_search_calls: 0,
                max_chat_context_query_calls: 5,
            },
            successful_calls: 0,
            web_search_calls: 0,
            chat_context_query_calls: 0,
            force_final_answer: false,
            accumulated_hits: BTreeMap::new(),
        }
    }

    pub fn force_final_answer(&self) -> bool {
        self.force_final_answer
    }

    pub fn max_total_successful_calls(&self) -> usize {
        self.budget.max_total_successful_calls
    }

    pub fn tool_limit_guidance(&self) -> String {
        match self.profile {
            ToolProfile::ChatQuestion => {
                "Tool budgets for this request: use web_search at most 3 times and chat_context_query at most 5 times. Once a budget is exhausted, answer with the evidence you already have.".to_string()
            }
            ToolProfile::ChatSearch => {
                "Tool budgets for this request: use chat_context_query at most 5 times total. Search is keyword-based FTS, not semantic, so inspect snippets carefully and refine your query if needed.".to_string()
            }
        }
    }

    pub fn build_openai_function_tools(&self) -> Vec<Value> {
        let mut tools = Vec::new();
        if self.profile == ToolProfile::ChatQuestion && web_search::is_search_enabled() {
            tools.push(json!({
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "Search the web using the configured providers and return a concise Markdown summary.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query to look up on the public web."
                            },
                            "max_results": {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": MAX_WEB_RESULTS,
                                "description": "Maximum number of results to return."
                            }
                        },
                        "required": ["query"]
                    }
                }
            }));
        }

        tools.push(json!({
            "type": "function",
            "function": {
                "name": "chat_context_query",
                "description": "Retrieve messages from the current Telegram chat only. This tool never accesses other chats.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "operation": {
                            "type": "string",
                            "enum": ["search", "window"]
                        },
                        "query": {
                            "type": "string",
                            "description": "Plain text search intent for keyword/FTS search. Never send SQL or raw FTS syntax."
                        },
                        "limit": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": MAX_SEARCH_LIMIT,
                            "description": "Maximum number of hits to return."
                        },
                        "offset": {
                            "type": "integer",
                            "minimum": 0,
                            "maximum": MAX_SEARCH_OFFSET,
                            "description": "Offset for additional pages of search hits."
                        },
                        "context_before": {
                            "type": "integer",
                            "minimum": 0,
                            "maximum": MAX_CONTEXT_WINDOW,
                            "description": "Number of earlier messages to include around each hit."
                        },
                        "context_after": {
                            "type": "integer",
                            "minimum": 0,
                            "maximum": MAX_CONTEXT_WINDOW,
                            "description": "Number of later messages to include around each hit."
                        },
                        "message_id": {
                            "type": "integer",
                            "description": "Target message ID for the window operation."
                        }
                    },
                    "required": ["operation"]
                }
            }
        }));

        tools
    }

    pub fn build_gemini_tools(&self) -> Vec<Value> {
        let mut declarations = Vec::new();
        if self.profile == ToolProfile::ChatQuestion && web_search::is_search_enabled() {
            declarations.push(json!({
                "name": "web_search",
                "description": "Search the web using the configured providers and return a concise Markdown summary.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query to look up on the public web."
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum number of results to return."
                        }
                    },
                    "required": ["query"]
                }
            }));
        }

        declarations.push(json!({
            "name": "chat_context_query",
            "description": "Retrieve messages from the current Telegram chat only. This tool never accesses other chats.",
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["search", "window"]
                    },
                    "query": {
                        "type": "string",
                        "description": "Plain text search intent for keyword/FTS search. Never send SQL or raw FTS syntax."
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of hits to return."
                    },
                    "offset": {
                        "type": "integer",
                        "description": "Offset for additional pages of search hits."
                    },
                    "context_before": {
                        "type": "integer",
                        "description": "Number of earlier messages to include around each hit."
                    },
                    "context_after": {
                        "type": "integer",
                        "description": "Number of later messages to include around each hit."
                    },
                    "message_id": {
                        "type": "integer",
                        "description": "Target message ID for the window operation."
                    }
                },
                "required": ["operation"]
            }
        }));

        vec![json!({ "functionDeclarations": declarations })]
    }

    pub fn select_hits_by_message_ids(&self, ids: &[i64], max_hits: usize) -> Vec<ChatSearchHit> {
        let mut selected = Vec::new();
        let mut seen = BTreeMap::new();
        for message_id in ids {
            if selected.len() >= max_hits {
                break;
            }
            if seen.insert(*message_id, true).is_some() {
                continue;
            }
            if let Some(hit) = self.accumulated_hits.get(message_id) {
                selected.push(hit.clone());
            }
        }
        selected
    }

    pub async fn execute_tool(&mut self, name: &str, arguments: &Value) -> String {
        match name {
            "web_search" => match self.begin_tool_call(ToolName::WebSearch) {
                Ok(()) => self.execute_web_search(arguments).await,
                Err(err) => self.tool_budget_error_payload("web_search", err),
            },
            "chat_context_query" => match self.begin_tool_call(ToolName::ChatContextQuery) {
                Ok(()) => self.execute_chat_context_query(arguments).await,
                Err(err) => self.tool_budget_error_payload("chat_context_query", err),
            },
            _ => {
                self.force_final_answer = true;
                self.error_payload(
                    name,
                    "unsupported_tool",
                    "Unsupported tool call requested by the model.",
                )
            }
        }
    }

    fn begin_tool_call(&mut self, tool: ToolName) -> std::result::Result<(), ToolBudgetError> {
        if self.force_final_answer {
            return Err(ToolBudgetError {
                kind: ToolBudgetErrorKind::Disabled,
            });
        }
        if self.successful_calls >= self.budget.max_total_successful_calls {
            self.force_final_answer = true;
            return Err(ToolBudgetError {
                kind: ToolBudgetErrorKind::Total,
            });
        }

        match tool {
            ToolName::WebSearch => {
                if self.profile != ToolProfile::ChatQuestion || !web_search::is_search_enabled() {
                    self.force_final_answer = true;
                    return Err(ToolBudgetError {
                        kind: ToolBudgetErrorKind::Disabled,
                    });
                }
                if self.web_search_calls >= self.budget.max_web_search_calls {
                    self.force_final_answer = true;
                    return Err(ToolBudgetError {
                        kind: ToolBudgetErrorKind::WebSearch,
                    });
                }
                self.web_search_calls += 1;
            }
            ToolName::ChatContextQuery => {
                if self.chat_context_query_calls >= self.budget.max_chat_context_query_calls {
                    self.force_final_answer = true;
                    return Err(ToolBudgetError {
                        kind: ToolBudgetErrorKind::ChatContextQuery,
                    });
                }
                self.chat_context_query_calls += 1;
            }
        }

        self.successful_calls += 1;
        Ok(())
    }

    fn remaining_budget_snapshot(&self) -> ToolBudgetSnapshot {
        ToolBudgetSnapshot {
            total_remaining: self
                .budget
                .max_total_successful_calls
                .saturating_sub(self.successful_calls),
            web_search_remaining: self
                .budget
                .max_web_search_calls
                .saturating_sub(self.web_search_calls),
            chat_context_query_remaining: self
                .budget
                .max_chat_context_query_calls
                .saturating_sub(self.chat_context_query_calls),
        }
    }

    async fn execute_web_search(&self, arguments: &Value) -> String {
        let query = arguments
            .get("query")
            .and_then(Value::as_str)
            .unwrap_or("")
            .trim();
        let max_results = arguments
            .get("max_results")
            .and_then(Value::as_u64)
            .map(|value| value as usize)
            .unwrap_or(5)
            .clamp(1, MAX_WEB_RESULTS);

        if query.is_empty() {
            return self.error_payload(
                "web_search",
                "invalid_arguments",
                "The web_search tool requires a non-empty query string.",
            );
        }

        match web_search_tool(query, Some(max_results)).await {
            Ok(result) => self.success_payload(
                "web_search",
                json!({
                    "query": query,
                    "max_results": max_results,
                    "result_markdown": result,
                }),
            ),
            Err(err) => self.error_payload("web_search", "tool_execution_failed", &err.to_string()),
        }
    }

    async fn execute_chat_context_query(&mut self, arguments: &Value) -> String {
        let args: ChatContextQueryArgs = match serde_json::from_value(arguments.clone()) {
            Ok(args) => args,
            Err(err) => {
                return self.error_payload(
                    "chat_context_query",
                    "invalid_arguments",
                    &format!("Invalid chat_context_query arguments: {err}"),
                );
            }
        };

        match self.run_chat_context_query(args).await {
            Ok(payload) => self.success_payload("chat_context_query", payload),
            Err(err) => self.error_payload(
                "chat_context_query",
                "tool_execution_failed",
                &err.to_string(),
            ),
        }
    }

    async fn run_chat_context_query(&mut self, args: ChatContextQueryArgs) -> Result<Value> {
        match args {
            ChatContextQueryArgs::Search {
                query,
                limit,
                offset,
                context_before,
                context_after,
            } => {
                let query = query.trim();
                if query.is_empty() {
                    return Err(anyhow!(
                        "chat_context_query search requires a non-empty query"
                    ));
                }

                let limit = limit
                    .unwrap_or(match self.profile {
                        ToolProfile::ChatQuestion => DEFAULT_QC_SEARCH_LIMIT,
                        ToolProfile::ChatSearch => DEFAULT_S_SEARCH_LIMIT,
                    })
                    .clamp(1, MAX_SEARCH_LIMIT);
                let offset = offset.unwrap_or(0).clamp(0, MAX_SEARCH_OFFSET);
                let context_before = context_before.unwrap_or(0).clamp(0, MAX_CONTEXT_WINDOW);
                let context_after = context_after.unwrap_or(0).clamp(0, MAX_CONTEXT_WINDOW);

                let hits = self
                    .db
                    .search_chat_messages(self.chat_id, query, limit as i64, offset as i64)
                    .await?;
                for hit in &hits {
                    self.accumulated_hits.insert(hit.message_id, hit.clone());
                }

                let mut results = Vec::new();
                for hit in hits {
                    let context_messages = if context_before > 0 || context_after > 0 {
                        self.db
                            .get_message_window(
                                self.chat_id,
                                hit.message_id,
                                context_before as i64,
                                context_after as i64,
                            )
                            .await?
                            .unwrap_or_default()
                            .into_iter()
                            .map(message_row_to_tool_message)
                            .collect()
                    } else {
                        Vec::new()
                    };
                    results.push(hit_to_tool_search_hit(hit, context_messages));
                }

                Ok(json!({
                    "operation": "search",
                    "query": query,
                    "limit": limit,
                    "offset": offset,
                    "result_count": results.len(),
                    "results": results,
                }))
            }
            ChatContextQueryArgs::Window {
                message_id,
                context_before,
                context_after,
            } => {
                let context_before = context_before.unwrap_or(2).clamp(0, MAX_CONTEXT_WINDOW);
                let context_after = context_after.unwrap_or(2).clamp(0, MAX_CONTEXT_WINDOW);
                let Some(messages) = self
                    .db
                    .get_message_window(
                        self.chat_id,
                        message_id,
                        context_before as i64,
                        context_after as i64,
                    )
                    .await?
                else {
                    return Err(anyhow!(
                        "The requested message_id does not belong to the current chat or is unavailable."
                    ));
                };

                let messages = messages
                    .into_iter()
                    .map(message_row_to_tool_message)
                    .collect::<Vec<_>>();

                Ok(json!({
                    "operation": "window",
                    "message_id": message_id,
                    "result_count": messages.len(),
                    "messages": messages,
                }))
            }
        }
    }

    fn success_payload(&self, tool: &str, data: Value) -> String {
        json!({
            "ok": true,
            "tool": tool,
            "remaining": self.remaining_budget_snapshot(),
            "data": data,
        })
        .to_string()
    }

    fn error_payload(&self, tool: &str, error_code: &str, message: &str) -> String {
        json!({
            "ok": false,
            "tool": tool,
            "error_code": error_code,
            "error": message,
            "remaining": self.remaining_budget_snapshot(),
        })
        .to_string()
    }

    fn tool_budget_error_payload(&self, tool: &str, error: ToolBudgetError) -> String {
        let (error_code, message) = match error.kind {
            ToolBudgetErrorKind::Total => (
                "total_budget_exhausted",
                "The total tool-call budget for this request is exhausted. Answer using the evidence already gathered.",
            ),
            ToolBudgetErrorKind::WebSearch => (
                "web_search_budget_exhausted",
                "The web_search budget for this request is exhausted. Answer using the evidence already gathered.",
            ),
            ToolBudgetErrorKind::ChatContextQuery => (
                "chat_context_query_budget_exhausted",
                "The chat_context_query budget for this request is exhausted. Answer using the evidence already gathered.",
            ),
            ToolBudgetErrorKind::Disabled => (
                "tool_disabled",
                "This tool is unavailable for the current request. Answer using the evidence already gathered.",
            ),
        };
        self.error_payload(tool, error_code, message)
    }
}

#[derive(Debug, Clone, Copy)]
enum ToolName {
    WebSearch,
    ChatContextQuery,
}

fn truncate_tool_text(text: &str) -> String {
    let char_count = text.chars().count();
    if char_count <= MAX_TOOL_TEXT_CHARS {
        return text.to_string();
    }

    let mut truncated: String = text.chars().take(MAX_TOOL_TEXT_CHARS).collect();
    truncated.push_str("...");
    truncated
}

fn message_row_to_tool_message(row: MessageRow) -> ToolMessage {
    ToolMessage {
        message_id: row.message_id,
        username: row.username,
        date_utc: row.date.to_rfc3339(),
        text: truncate_tool_text(&row.text.unwrap_or_default()),
        link: build_message_link(row.chat_id, row.message_id),
    }
}

fn hit_to_tool_search_hit(hit: ChatSearchHit, context_messages: Vec<ToolMessage>) -> ToolSearchHit {
    ToolSearchHit {
        message_id: hit.message_id,
        username: hit.username,
        date_utc: hit.date.to_rfc3339(),
        text: truncate_tool_text(&hit.text),
        snippet: hit.snippet,
        link: hit.link,
        score: hit.score,
        context_messages,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::database::Database;
    use chrono::Utc;
    use tokio::runtime::Runtime;

    fn test_db_path(test_name: &str) -> std::path::PathBuf {
        let mut path = std::path::PathBuf::from("target");
        path.push("test-dbs");
        std::fs::create_dir_all(&path).expect("test db directory should exist");
        path.push(format!(
            "telegram-chat-bot-tool-runtime-{}-{}-{}.db",
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

    #[test]
    fn qc_budget_stops_after_expected_counts() {
        let runtime = Runtime::new().expect("tokio runtime should initialize");
        let db = runtime.block_on(init_test_db("qc-budget"));
        let mut runtime = ToolRuntime::for_qc(db, -1001374348669);

        runtime.successful_calls = runtime.budget.max_total_successful_calls;
        assert!(runtime.begin_tool_call(ToolName::ChatContextQuery).is_err());
        assert!(runtime.force_final_answer());
    }

    #[test]
    fn search_budget_stops_after_five_chat_queries() {
        let runtime = Runtime::new().expect("tokio runtime should initialize");
        let db = runtime.block_on(init_test_db("s-budget"));
        let mut runtime = ToolRuntime::for_search(db, -1001374348669);

        for _ in 0..5 {
            assert!(runtime.begin_tool_call(ToolName::ChatContextQuery).is_ok());
        }
        assert!(runtime.begin_tool_call(ToolName::ChatContextQuery).is_err());
        assert!(runtime.force_final_answer());
    }
}
