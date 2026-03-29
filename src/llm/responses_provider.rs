use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use anyhow::{Context, Result, anyhow};
use base64::{Engine as _, engine::general_purpose};
use reqwest::StatusCode;
use serde_json::{Value, json};
use tracing::{debug, warn};

use crate::config::{ThirdPartyModelConfig, ThirdPartyProvider, CONFIG};
use crate::llm::openai_codex;
use crate::llm::runtime_models::selected_codex_model_record;
use crate::llm::tool_runtime::ToolRuntime;
use crate::llm::web_search::{self, web_search_tool};
use crate::utils::http::get_http_client;
use crate::utils::timing::log_llm_timing;

const MAX_TOOL_CALL_ITERATIONS: usize = 3;
const RESPONSES_MAX_ATTEMPTS: usize = 3;
const RESPONSES_RETRY_BASE_DELAY_MS: u64 = 900;
const RESPONSES_REQUEST_TIMEOUT_SECS: u64 = 60;
const TOOL_LIMIT_SYSTEM_PROMPT: &str = "Tool call limit reached. Provide the best possible answer using the available information without requesting more tool calls.";
const RESPONSES_TOOL_LIMIT_GUIDANCE: &str = "Tool usage limit: you may use tools for at most {max_tool_calls} rounds total in this conversation. Plan your searches efficiently, avoid redundant tool calls, and after the final allowed tool round you must answer using the information already gathered without requesting more tool calls.";
static SESSION_COUNTER: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, Clone)]
struct ResponsesRequestDetails {
    provider: ThirdPartyProvider,
    display_name: &'static str,
    url: String,
    headers: Vec<(String, String)>,
    payload: Value,
    streaming_sse: bool,
}

#[derive(Debug, Clone)]
struct ResponsesToolCall {
    call_id: String,
    name: String,
    arguments: String,
}

fn generate_session_id() -> String {
    let counter = SESSION_COUNTER.fetch_add(1, Ordering::Relaxed);
    let now = chrono::Utc::now().timestamp_millis();
    format!("tg-codex-{now}-{counter}")
}

fn truncate_for_log(value: &str, limit: usize) -> String {
    if value.chars().count() <= limit {
        return value.to_string();
    }
    let truncated: String = value.chars().take(limit).collect();
    format!("{truncated}... (truncated)")
}

fn summarize_error_body(body: &str) -> (Option<String>, String) {
    let trimmed = body.trim();
    if trimmed.is_empty() {
        return (None, "empty response body".to_string());
    }

    if let Ok(value) = serde_json::from_str::<Value>(trimmed) {
        let message = value
            .pointer("/error/message")
            .and_then(|v| v.as_str())
            .map(|v| v.to_string())
            .or_else(|| {
                value
                    .get("message")
                    .and_then(|v| v.as_str())
                    .map(|v| v.to_string())
            });
        return (message, truncate_for_log(&value.to_string(), 2000));
    }

    (None, truncate_for_log(trimmed, 2000))
}

fn responses_should_retry_error(err: &reqwest::Error) -> bool {
    err.is_timeout() || err.is_connect()
}

fn responses_should_retry_status(status: StatusCode) -> bool {
    status == StatusCode::TOO_MANY_REQUESTS
        || status == StatusCode::REQUEST_TIMEOUT
        || status.is_server_error()
}

fn responses_retry_delay(attempt: usize) -> Duration {
    let attempt = attempt.max(1) as u64;
    Duration::from_millis(RESPONSES_RETRY_BASE_DELAY_MS.saturating_mul(attempt))
}

fn build_responses_system_prompt(system_prompt: &str, include_tool_limit_guidance: bool) -> String {
    if !include_tool_limit_guidance {
        return system_prompt.to_string();
    }

    let tool_limit_guidance =
        RESPONSES_TOOL_LIMIT_GUIDANCE.replace("{max_tool_calls}", &MAX_TOOL_CALL_ITERATIONS.to_string());
    format!("{system_prompt}\n\n{tool_limit_guidance}")
}

fn build_responses_user_input(user_content: &str, image_data_list: &[Vec<u8>]) -> Vec<Value> {
    let mut content = vec![json!({
        "type": "input_text",
        "text": user_content.to_string(),
    })];

    for image_data in image_data_list {
        let mime_type = crate::llm::media::detect_mime_type(image_data)
            .unwrap_or_else(|| "image/png".to_string());
        let encoded = general_purpose::STANDARD.encode(image_data);
        let data_url = format!("data:{};base64,{}", mime_type, encoded);
        content.push(json!({
            "type": "input_image",
            "detail": "auto",
            "image_url": data_url,
        }));
    }

    vec![json!({
        "type": "message",
        "role": "user",
        "content": content,
    })]
}

fn build_responses_function_tools() -> Vec<Value> {
    if !web_search::is_search_enabled() {
        return Vec::new();
    }

    vec![json!({
        "type": "function",
        "name": "web_search",
        "description": "Search the web using the configured providers (Brave, Exa, Jina) and return a concise Markdown summary of the results.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query to look up."
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default 5).",
                    "minimum": 1,
                    "maximum": 10
                }
            },
            "required": ["query"]
        },
        "strict": false,
    })]
}

fn convert_openai_function_tools_to_responses(tools: Vec<Value>) -> Vec<Value> {
    tools.into_iter()
        .filter_map(|tool| {
            let function = tool.get("function")?;
            Some(json!({
                "type": "function",
                "name": function.get("name")?.as_str()?,
                "description": function
                    .get("description")
                    .and_then(|value| value.as_str())
                    .unwrap_or(""),
                "parameters": function.get("parameters").cloned().unwrap_or_else(|| json!({})),
                "strict": false,
            }))
        })
        .collect()
}

fn responses_base_url(base_url: &str) -> String {
    let normalized = base_url.trim().trim_end_matches('/');
    if normalized.ends_with("/responses") {
        normalized.to_string()
    } else {
        format!("{normalized}/responses")
    }
}

async fn build_request_details(
    model_config: &ThirdPartyModelConfig,
    instructions: &str,
    input_items: Vec<Value>,
    tools: Option<Vec<Value>>,
    session_id: &str,
) -> Result<ResponsesRequestDetails> {
    let (display_name, url, mut headers, streaming_sse) = match model_config.provider {
        ThirdPartyProvider::OpenAI => (
            "OpenAI",
            responses_base_url(&CONFIG.openai_base_url),
            vec![
                (
                    "Authorization".to_string(),
                    format!("Bearer {}", CONFIG.openai_api_key),
                ),
                (
                    "User-Agent".to_string(),
                    format!("{}/{}", env!("CARGO_PKG_NAME"), env!("CARGO_PKG_VERSION")),
                ),
            ],
            false,
        ),
        ThirdPartyProvider::OpenAICodex => {
            let auth = openai_codex::get_valid_auth_context().await?;
            (
                "OpenAI Codex",
                openai_codex::codex_response_url(),
                openai_codex::codex_headers(&auth, Some(session_id)),
                true,
            )
        }
        _ => return Err(anyhow!("Unsupported responses provider {:?}", model_config.provider)),
    };

    if streaming_sse {
        headers.push(("Accept".to_string(), "text/event-stream".to_string()));
    }

    let mut payload = json!({
        "model": model_config.model,
        "instructions": instructions,
        "input": input_items,
        "tool_choice": "auto",
        "parallel_tool_calls": true,
        "store": false,
        "stream": streaming_sse,
        "include": ["reasoning.encrypted_content"],
        "prompt_cache_key": session_id,
        "text": {
            "verbosity": "medium"
        },
    });

    if model_config.provider == ThirdPartyProvider::OpenAICodex {
        if let Some(record) = selected_codex_model_record() {
            if record.slug == model_config.model {
                if let Some(level) = record.selected_reasoning_level.as_ref() {
                    payload["reasoning"] = json!({
                        "effort": level,
                    });
                }
            }
        }
    }

    if let Some(tools) = tools.filter(|tools| !tools.is_empty()) {
        payload["tools"] = Value::Array(tools);
    }

    Ok(ResponsesRequestDetails {
        provider: model_config.provider,
        display_name,
        url,
        headers,
        payload,
        streaming_sse,
    })
}

fn parse_sse_responses_body(body: &str) -> Result<Value> {
    let mut output_items: Vec<Value> = Vec::new();
    let mut current_data_lines: Vec<String> = Vec::new();

    let flush_event = |lines: &mut Vec<String>, output_items: &mut Vec<Value>| -> Result<()> {
        if lines.is_empty() {
            return Ok(());
        }
        let payload = lines.join("\n");
        lines.clear();
        if payload.trim().is_empty() || payload.trim() == "[DONE]" {
            return Ok(());
        }

        let value: Value = serde_json::from_str(&payload)
            .with_context(|| format!("Failed to parse SSE event payload: {}", truncate_for_log(&payload, 500)))?;
        match value.get("type").and_then(|value| value.as_str()) {
            Some("response.output_item.done") => {
                if let Some(item) = value.get("item").cloned() {
                    output_items.push(item);
                }
            }
            Some("response.completed") => {
                if output_items.is_empty() {
                    if let Some(items) = value
                        .get("response")
                        .and_then(|response| response.get("output"))
                        .and_then(|items| items.as_array())
                    {
                        output_items.extend(items.iter().cloned());
                    }
                }
            }
            Some("response.failed") => {
                let detail = value
                    .pointer("/response/error/message")
                    .and_then(|value| value.as_str())
                    .or_else(|| value.pointer("/error/message").and_then(|value| value.as_str()))
                    .unwrap_or("unknown SSE failure");
                return Err(anyhow!("Codex SSE request failed: {}", detail));
            }
            _ => {}
        }

        Ok(())
    };

    for line in body.lines() {
        if line.trim().is_empty() {
            flush_event(&mut current_data_lines, &mut output_items)?;
            continue;
        }
        if let Some(data) = line.strip_prefix("data:") {
            current_data_lines.push(data.trim_start().to_string());
        }
    }
    flush_event(&mut current_data_lines, &mut output_items)?;

    Ok(json!({ "output": output_items }))
}

async fn call_provider_api(details: &ResponsesRequestDetails) -> Result<Value> {
    let client = get_http_client();
    for attempt in 1..=RESPONSES_MAX_ATTEMPTS {
        let mut request = client
            .post(&details.url)
            .timeout(Duration::from_secs(RESPONSES_REQUEST_TIMEOUT_SECS));
        for (name, value) in &details.headers {
            request = request.header(name, value);
        }

        let response = match request.json(&details.payload).send().await {
            Ok(response) => response,
            Err(err) => {
                let should_retry =
                    responses_should_retry_error(&err) && attempt < RESPONSES_MAX_ATTEMPTS;
                warn!(
                    "{} responses request failed to send: {} (attempt={}/{}, retrying={})",
                    details.display_name,
                    err,
                    attempt,
                    RESPONSES_MAX_ATTEMPTS,
                    should_retry
                );
                if should_retry {
                    tokio::time::sleep(responses_retry_delay(attempt)).await;
                    continue;
                }
                return Err(anyhow!("{} request failed: {}", details.display_name, err));
            }
        };

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            if status == StatusCode::UNAUTHORIZED
                && details.provider == ThirdPartyProvider::OpenAICodex
                && attempt < RESPONSES_MAX_ATTEMPTS
            {
                openai_codex::force_refresh_auth_tokens().await?;
                continue;
            }
            let (message, body_summary) = summarize_error_body(&body);
            let should_retry =
                responses_should_retry_status(status) && attempt < RESPONSES_MAX_ATTEMPTS;
            warn!(
                "{} responses API error: status={}, body={}, attempt={}/{}, retrying={}",
                details.display_name,
                status,
                body_summary,
                attempt,
                RESPONSES_MAX_ATTEMPTS,
                should_retry
            );
            if should_retry {
                tokio::time::sleep(responses_retry_delay(attempt)).await;
                continue;
            }
            let detail = message.unwrap_or(body_summary);
            return Err(anyhow!(
                "{} request failed with status {}: {}",
                details.display_name,
                status,
                detail
            ));
        }

        let value = if details.streaming_sse {
            let body = response.text().await?;
            parse_sse_responses_body(&body)?
        } else {
            response.json::<Value>().await?
        };
        debug!(
            "{} responses received for model={}",
            details.display_name,
            details
                .payload
                .get("model")
                .and_then(|value| value.as_str())
                .unwrap_or("unknown")
        );
        return Ok(value);
    }

    unreachable!("responses provider retry loop exhausted")
}

fn extract_response_output_items(response: &Value) -> Vec<Value> {
    response
        .get("output")
        .and_then(|value| value.as_array())
        .cloned()
        .unwrap_or_default()
}

fn extract_response_text(output_items: &[Value]) -> String {
    let mut text_parts = Vec::new();
    let mut reasoning_parts = Vec::new();

    for item in output_items {
        match item.get("type").and_then(|value| value.as_str()) {
            Some("message") => {
                if let Some(content_items) = item.get("content").and_then(|value| value.as_array()) {
                    for content_item in content_items {
                        let item_type = content_item.get("type").and_then(|value| value.as_str());
                        if matches!(item_type, Some("output_text") | Some("text")) {
                            if let Some(text) =
                                content_item.get("text").and_then(|value| value.as_str())
                            {
                                let trimmed = text.trim();
                                if !trimmed.is_empty() {
                                    text_parts.push(trimmed.to_string());
                                }
                            }
                        }
                    }
                }
            }
            Some("reasoning") => {
                if let Some(summary_items) = item.get("summary").and_then(|value| value.as_array()) {
                    for summary_item in summary_items {
                        if let Some(text) = summary_item.get("text").and_then(|value| value.as_str()) {
                            let trimmed = text.trim();
                            if !trimmed.is_empty() {
                                reasoning_parts.push(trimmed.to_string());
                            }
                        }
                    }
                }
            }
            _ => {}
        }
    }

    if !text_parts.is_empty() {
        return text_parts.join("\n");
    }
    reasoning_parts.join("\n")
}

fn extract_response_tool_calls(output_items: &[Value]) -> Vec<ResponsesToolCall> {
    output_items
        .iter()
        .filter(|item| item.get("type").and_then(|value| value.as_str()) == Some("function_call"))
        .filter_map(|item| {
            Some(ResponsesToolCall {
                call_id: item.get("call_id")?.as_str()?.to_string(),
                name: item.get("name")?.as_str()?.to_string(),
                arguments: item
                    .get("arguments")
                    .and_then(|value| value.as_str())
                    .unwrap_or("{}")
                    .to_string(),
            })
        })
        .collect()
}

async fn execute_function_tool(name: &str, arguments: &Value) -> Result<String> {
    match name {
        "web_search" => {
            let query = arguments
                .get("query")
                .and_then(|value| value.as_str())
                .unwrap_or("");
            let max_results = arguments
                .get("max_results")
                .and_then(|value| value.as_u64())
                .map(|value| value as usize);
            match web_search_tool(query, max_results).await {
                Ok(result) => Ok(result),
                Err(err) => Err(err),
            }
        }
        _ => Ok(String::from("Unsupported tool call")),
    }
}

async fn responses_completion_with_tools(
    instructions: &str,
    mut input_items: Vec<Value>,
    model_config: &ThirdPartyModelConfig,
) -> Result<String> {
    let tools = build_responses_function_tools();
    let session_id = generate_session_id();

    for iteration in 0..MAX_TOOL_CALL_ITERATIONS {
        let details = build_request_details(
            model_config,
            instructions,
            input_items.clone(),
            Some(tools.clone()),
            &session_id,
        )
        .await?;
        let response = call_provider_api(&details).await?;
        let output_items = extract_response_output_items(&response);
        let tool_calls = extract_response_tool_calls(&output_items);
        let content = extract_response_text(&output_items);

        if tool_calls.is_empty() {
            return Ok(content);
        }

        input_items.extend(output_items.clone());

        for tool_call in tool_calls {
            let args_value: Value = serde_json::from_str(&tool_call.arguments).unwrap_or(Value::Null);
            let result = execute_function_tool(&tool_call.name, &args_value)
                .await
                .unwrap_or_else(|err| err.to_string());
            input_items.push(json!({
                "type": "function_call_output",
                "call_id": tool_call.call_id,
                "output": result,
            }));
        }

        if iteration + 1 == MAX_TOOL_CALL_ITERATIONS {
            let final_instructions = format!("{instructions}\n\n{TOOL_LIMIT_SYSTEM_PROMPT}");
            let details = build_request_details(
                model_config,
                &final_instructions,
                input_items,
                None,
                &session_id,
            )
            .await?;
            let response = call_provider_api(&details).await?;
            return Ok(extract_response_text(&extract_response_output_items(&response)));
        }
    }

    unreachable!("responses tool loop exhausted without returning")
}

async fn responses_completion_with_tool_runtime(
    instructions: &str,
    mut input_items: Vec<Value>,
    model_config: &ThirdPartyModelConfig,
    runtime: &mut ToolRuntime,
) -> Result<String> {
    let tools = convert_openai_function_tools_to_responses(runtime.build_openai_function_tools());
    let mut tools_enabled = !tools.is_empty();
    let session_id = generate_session_id();
    let mut final_answer_requested = false;

    for _ in 0..runtime.max_total_successful_calls().saturating_add(2) {
        let details = build_request_details(
            model_config,
            instructions,
            input_items.clone(),
            tools_enabled.then_some(tools.clone()),
            &session_id,
        )
        .await?;
        let response = call_provider_api(&details).await?;
        let output_items = extract_response_output_items(&response);
        let tool_calls = if tools_enabled {
            extract_response_tool_calls(&output_items)
        } else {
            Vec::new()
        };

        if tool_calls.is_empty() {
            return Ok(extract_response_text(&output_items));
        }

        input_items.extend(output_items.clone());

        for tool_call in tool_calls {
            let args_value: Value = serde_json::from_str(&tool_call.arguments).unwrap_or_else(|_| json!({}));
            let result = runtime.execute_tool(&tool_call.name, &args_value).await;
            input_items.push(json!({
                "type": "function_call_output",
                "call_id": tool_call.call_id,
                "output": result,
            }));
        }

        if runtime.force_final_answer() && !final_answer_requested {
            final_answer_requested = true;
            tools_enabled = false;
        }
    }

    let final_instructions = format!("{instructions}\n\n{TOOL_LIMIT_SYSTEM_PROMPT}");
    let details =
        build_request_details(model_config, &final_instructions, input_items, None, &session_id)
            .await?;
    let response = call_provider_api(&details).await?;
    Ok(extract_response_text(&extract_response_output_items(&response)))
}

pub async fn call_responses_provider(
    system_prompt: &str,
    user_content: &str,
    model_config: &ThirdPartyModelConfig,
    response_title: &str,
    image_data_list: &[Vec<u8>],
    supports_tools: bool,
) -> Result<String> {
    let tools_enabled = supports_tools && web_search::is_search_enabled();
    let instructions = build_responses_system_prompt(system_prompt, tools_enabled);
    let input_items = build_responses_user_input(user_content, image_data_list);
    let operation = format!("{}:{}", model_config.provider.as_str(), response_title);

    if tools_enabled {
        return log_llm_timing(
            model_config.provider.as_str(),
            &model_config.id,
            &operation,
            None,
            || async {
                responses_completion_with_tools(&instructions, input_items, model_config)
                    .await
                    .map_err(|err| anyhow!(err))
            },
        )
        .await;
    }

    let session_id = generate_session_id();
    let details =
        build_request_details(model_config, &instructions, input_items, None, &session_id).await?;
    log_llm_timing(
        model_config.provider.as_str(),
        &model_config.id,
        &operation,
        None,
        || async {
            let response = call_provider_api(&details).await?;
            Ok(extract_response_text(&extract_response_output_items(&response)))
        },
    )
    .await
}

pub async fn call_responses_provider_with_tool_runtime(
    system_prompt: &str,
    user_content: &str,
    model_config: &ThirdPartyModelConfig,
    response_title: &str,
    image_data_list: &[Vec<u8>],
    runtime: &mut ToolRuntime,
) -> Result<String> {
    let instructions = format!("{}\n\n{}", system_prompt, runtime.tool_limit_guidance());
    let input_items = build_responses_user_input(user_content, image_data_list);
    let operation = format!("{}:{}", model_config.provider.as_str(), response_title);

    log_llm_timing(
        model_config.provider.as_str(),
        &model_config.id,
        &operation,
        None,
        || async {
            responses_completion_with_tool_runtime(&instructions, input_items, model_config, runtime)
                .await
                .map_err(|err| anyhow!(err))
        },
    )
    .await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_response_text_reads_output_text_blocks() {
        let output = vec![json!({
            "type": "message",
            "role": "assistant",
            "content": [
                { "type": "output_text", "text": "hello" },
                { "type": "output_text", "text": "world" }
            ]
        })];

        assert_eq!(extract_response_text(&output), "hello\nworld");
    }

    #[test]
    fn extract_response_tool_calls_reads_function_calls() {
        let output = vec![json!({
            "type": "function_call",
            "call_id": "call_123",
            "name": "web_search",
            "arguments": "{\"query\":\"rust\"}"
        })];

        let calls = extract_response_tool_calls(&output);

        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].call_id, "call_123");
        assert_eq!(calls[0].name, "web_search");
    }

    #[test]
    fn responses_base_url_appends_suffix_once() {
        assert_eq!(
            responses_base_url("https://api.openai.com/v1"),
            "https://api.openai.com/v1/responses"
        );
        assert_eq!(
            responses_base_url("https://chatgpt.com/backend-api/codex/responses"),
            "https://chatgpt.com/backend-api/codex/responses"
        );
    }

    #[test]
    fn parse_sse_responses_body_collects_output_items() {
        let body = r#"event: response.created
data: {"type":"response.created","response":{"id":"resp1"}}

event: response.output_item.done
data: {"type":"response.output_item.done","item":{"type":"message","role":"assistant","content":[{"type":"output_text","text":"hello"}]}}

event: response.output_item.done
data: {"type":"response.output_item.done","item":{"type":"function_call","call_id":"call_1","name":"web_search","arguments":"{}"}}

event: response.completed
data: {"type":"response.completed","response":{"id":"resp1","output":[]}}
"#;

        let parsed = parse_sse_responses_body(body).expect("SSE body should parse");
        let output = parsed
            .get("output")
            .and_then(|value| value.as_array())
            .expect("output array");

        assert_eq!(output.len(), 2);
        assert_eq!(output[0]["type"], "message");
        assert_eq!(output[1]["type"], "function_call");
    }
}
