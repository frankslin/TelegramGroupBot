use std::time::Duration;

use anyhow::{anyhow, Result};
use base64::{engine::general_purpose, Engine as _};
use regex::Regex;
use serde_json::{json, Value};
use tracing::{debug, warn};

use crate::config::{ThirdPartyModelConfig, ThirdPartyProvider, CONFIG};
use crate::llm::web_search::{self, web_search_tool};
use crate::utils::http::get_http_client;
use crate::utils::timing::log_llm_timing;

const MAX_TOOL_CALL_ITERATIONS: usize = 3;
const TOOL_LIMIT_SYSTEM_PROMPT: &str = "Tool call limit reached. Provide the best possible answer using the available information without requesting more tool calls.";
const OPENROUTER_REFERER: &str = "https://github.com/sailself/TelegramGroupHelperBot";
const OPENROUTER_TITLE: &str = "TelegramGroupHelperBot";

#[derive(Debug, Clone)]
struct ProviderRuntimeConfig {
    provider: ThirdPartyProvider,
    display_name: &'static str,
    base_url: String,
    api_key: String,
    temperature: f32,
    top_p: f32,
    top_k: Option<i32>,
}

#[derive(Debug, Clone)]
struct ProviderRequestDetails {
    display_name: &'static str,
    url: String,
    headers: Vec<(String, String)>,
    payload: Value,
}

fn truncate_for_log(value: &str, limit: usize) -> String {
    if value.chars().count() <= limit {
        return value.to_string();
    }
    let truncated: String = value.chars().take(limit).collect();
    format!("{truncated}... (truncated)")
}

fn summarize_payload(payload: &Value) -> String {
    let model = payload
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown");
    let message_count = payload
        .get("messages")
        .and_then(|v| v.as_array())
        .map(|messages| messages.len())
        .unwrap_or(0);
    let tool_names = payload
        .get("tools")
        .and_then(|v| v.as_array())
        .map(|tools| {
            tools
                .iter()
                .filter_map(|tool| {
                    tool.get("function")
                        .and_then(|f| f.get("name"))
                        .and_then(|v| v.as_str())
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let tool_choice = payload
        .get("tool_choice")
        .and_then(|v| v.as_str())
        .unwrap_or("auto");

    format!(
        "model={}, messages={}, tools={}, tool_choice={}, tool_names=[{}]",
        model,
        message_count,
        tool_names.len(),
        tool_choice,
        tool_names.join(",")
    )
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

fn parse_gpt_content(content: &str) -> String {
    if let Some(last_pos) = content.rfind("<|message|>") {
        let analysis = &content[..last_pos];
        let final_text = &content[last_pos + "<|message|>".len()..];
        let cleanup = Regex::new(r"<\|.*?\|>").unwrap();
        let final_clean = cleanup.replace_all(final_text, "").trim().to_string();
        if !final_clean.is_empty() {
            return final_clean;
        }
        let analysis_clean = cleanup.replace_all(analysis, "").trim().to_string();
        return analysis_clean;
    }
    content.to_string()
}

fn parse_qwen_content(content: &str) -> String {
    let re = Regex::new(r"<think>(.*?)</think>(.*)").unwrap();
    if let Some(caps) = re.captures(content) {
        let final_text = caps.get(2).map(|m| m.as_str()).unwrap_or("");
        let final_text = final_text.trim();
        if !final_text.is_empty() {
            return final_text.to_string();
        }
        let analysis = caps.get(1).map(|m| m.as_str()).unwrap_or("");
        return analysis.trim().to_string();
    }
    content.trim().to_string()
}

fn parse_third_party_response(model_config: &ThirdPartyModelConfig, content: &str) -> String {
    if model_config.provider != ThirdPartyProvider::OpenRouter {
        return content.to_string();
    }

    let haystack = format!("{} {}", model_config.name, model_config.model).to_lowercase();
    if haystack.contains("gpt") {
        return parse_gpt_content(content);
    }
    if haystack.contains("qwen") {
        return parse_qwen_content(content);
    }
    content.to_string()
}

fn extract_reasoning_text(message: &Value) -> Option<String> {
    if let Some(reasoning) = message.get("reasoning").and_then(|v| v.as_str()) {
        let trimmed = reasoning.trim();
        if !trimmed.is_empty() {
            return Some(trimmed.to_string());
        }
    }

    let details = message
        .get("reasoning_details")
        .and_then(|v| v.as_array())?;
    let mut parts = Vec::new();
    for detail in details {
        let text = detail.get("text").and_then(|v| v.as_str()).unwrap_or("");
        let text = text.trim();
        if !text.is_empty() {
            parts.push(text.to_string());
        }
    }

    if parts.is_empty() {
        None
    } else {
        Some(parts.join("\n"))
    }
}

fn extract_message_content(message: &Value) -> String {
    let content = message
        .get("content")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim()
        .to_string();

    if !content.is_empty() {
        return content;
    }

    extract_reasoning_text(message).unwrap_or_default()
}

fn build_function_tools() -> Vec<Value> {
    if !web_search::is_search_enabled() {
        return Vec::new();
    }

    vec![json!({
        "type": "function",
        "function": {
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
            }
        }
    })]
}

fn build_message_content(user_content: &str, image_data_list: &[Vec<u8>]) -> Value {
    if image_data_list.is_empty() {
        return Value::String(user_content.to_string());
    }

    let mut parts = Vec::new();
    parts.push(json!({
        "type": "text",
        "text": user_content
    }));

    for image_data in image_data_list {
        let mime_type = crate::llm::media::detect_mime_type(image_data)
            .unwrap_or_else(|| "image/png".to_string());
        let encoded = general_purpose::STANDARD.encode(image_data);
        let data_url = format!("data:{};base64,{}", mime_type, encoded);
        parts.push(json!({
            "type": "image_url",
            "image_url": { "url": data_url }
        }));
    }

    Value::Array(parts)
}

fn provider_runtime_config(provider: ThirdPartyProvider) -> Result<ProviderRuntimeConfig> {
    let config = match provider {
        ThirdPartyProvider::OpenRouter => ProviderRuntimeConfig {
            provider,
            display_name: "OpenRouter",
            base_url: CONFIG.openrouter_base_url.clone(),
            api_key: CONFIG.openrouter_api_key.clone(),
            temperature: CONFIG.openrouter_temperature,
            top_p: CONFIG.openrouter_top_p,
            top_k: Some(CONFIG.openrouter_top_k),
        },
        ThirdPartyProvider::Nvidia => ProviderRuntimeConfig {
            provider,
            display_name: "NVIDIA",
            base_url: CONFIG.nvidia_base_url.clone(),
            api_key: CONFIG.nvidia_api_key.clone(),
            temperature: CONFIG.nvidia_temperature,
            top_p: CONFIG.nvidia_top_p,
            top_k: None,
        },
    };

    if !CONFIG.is_third_party_provider_ready(provider) {
        return Err(anyhow!(
            "{} is not enabled or its API key is missing",
            config.display_name
        ));
    }

    Ok(config)
}

fn build_request_details_for_runtime(
    model_config: &ThirdPartyModelConfig,
    runtime: &ProviderRuntimeConfig,
    messages: Vec<Value>,
    tools: Option<Vec<Value>>,
    tool_choice: Option<&str>,
) -> ProviderRequestDetails {
    let mut headers = vec![(
        "Authorization".to_string(),
        format!("Bearer {}", runtime.api_key),
    )];
    if runtime.provider == ThirdPartyProvider::OpenRouter {
        headers.push(("HTTP-Referer".to_string(), OPENROUTER_REFERER.to_string()));
        headers.push(("X-Title".to_string(), OPENROUTER_TITLE.to_string()));
    }

    let mut payload = json!({
        "model": model_config.model,
        "messages": messages,
        "temperature": runtime.temperature,
        "top_p": runtime.top_p,
    });

    if let Some(top_k) = runtime.top_k {
        payload["top_k"] = json!(top_k);
    }

    if let Some(tools) = tools {
        payload["tools"] = Value::Array(tools);
        payload["tool_choice"] = Value::String(tool_choice.unwrap_or("auto").to_string());
    }

    ProviderRequestDetails {
        display_name: runtime.display_name,
        url: format!("{}/chat/completions", runtime.base_url.trim_end_matches('/')),
        headers,
        payload,
    }
}

fn build_request_details(
    model_config: &ThirdPartyModelConfig,
    messages: Vec<Value>,
    tools: Option<Vec<Value>>,
    tool_choice: Option<&str>,
) -> Result<ProviderRequestDetails> {
    let runtime = provider_runtime_config(model_config.provider)?;
    Ok(build_request_details_for_runtime(
        model_config,
        &runtime,
        messages,
        tools,
        tool_choice,
    ))
}

async fn call_provider_api(details: &ProviderRequestDetails) -> Result<Value> {
    debug!(
        "{} request: {}",
        details.display_name,
        summarize_payload(&details.payload)
    );

    let client = get_http_client();
    let mut request = client.post(&details.url).timeout(Duration::from_secs(60));
    for (name, value) in &details.headers {
        request = request.header(name, value);
    }
    let response = request.json(&details.payload).send().await?;

    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        let (message, body_summary) = summarize_error_body(&body);
        warn!(
            "{} API error: status={}, body={}",
            details.display_name, status, body_summary
        );
        let detail = message.unwrap_or(body_summary);
        return Err(anyhow!(
            "{} request failed with status {}: {}",
            details.display_name,
            status,
            detail
        ));
    }

    let value = response.json::<Value>().await?;
    debug!(
        "{} response received for model={}",
        details.display_name,
        details
            .payload
            .get("model")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown")
    );
    Ok(value)
}

async fn execute_function_tool(name: &str, arguments: &Value) -> Result<String> {
    match name {
        "web_search" => {
            let query = arguments
                .get("query")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let max_results = arguments
                .get("max_results")
                .and_then(|v| v.as_u64())
                .map(|v| v as usize);
            debug!(
                "Executing tool call web_search: query='{}', max_results={:?}",
                truncate_for_log(query, 200),
                max_results
            );
            match web_search_tool(query, max_results).await {
                Ok(result) => {
                    debug!("web_search returned {} chars", result.chars().count());
                    Ok(result)
                }
                Err(err) => {
                    warn!("web_search tool failed: {}", err);
                    Err(err)
                }
            }
        }
        _ => Ok(String::from("Unsupported tool call")),
    }
}

async fn chat_completion_with_tools(mut messages: Vec<Value>, model_config: &ThirdPartyModelConfig) -> Result<String> {
    let tools = build_function_tools();

    for iteration in 0..MAX_TOOL_CALL_ITERATIONS {
        debug!(
            "{} tool iteration {}/{}",
            model_config.provider.as_str(),
            iteration + 1,
            MAX_TOOL_CALL_ITERATIONS
        );
        let details = build_request_details(
            model_config,
            messages.clone(),
            Some(tools.clone()),
            Some("auto"),
        )?;
        let response = call_provider_api(&details).await?;
        let message = response
            .get("choices")
            .and_then(|v| v.get(0))
            .and_then(|v| v.get("message"))
            .cloned()
            .unwrap_or(Value::Null);

        let content = extract_message_content(&message);
        let tool_calls = message
            .get("tool_calls")
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_default();

        if tool_calls.is_empty() {
            if content.trim().is_empty() {
                warn!(
                    "{} response had empty content and no tool calls: {}",
                    details.display_name,
                    truncate_for_log(&response.to_string(), 2000)
                );
            }
            return Ok(parse_third_party_response(model_config, &content));
        }

        messages.push(message.clone());

        for tool_call in tool_calls {
            let tool_name = tool_call
                .get("function")
                .and_then(|f| f.get("name"))
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let args_text = tool_call
                .get("function")
                .and_then(|f| f.get("arguments"))
                .and_then(|v| v.as_str())
                .unwrap_or("{}");
            let args_value: Value = serde_json::from_str(args_text).unwrap_or(Value::Null);
            let result = execute_function_tool(tool_name, &args_value)
                .await
                .unwrap_or_else(|err| err.to_string());
            if result.trim().is_empty() {
                warn!("Tool call '{}' returned empty content", tool_name);
            }

            messages.push(json!({
                "role": "tool",
                "tool_call_id": tool_call.get("id").and_then(|v| v.as_str()).unwrap_or(""),
                "content": result
            }));
        }

        if iteration + 1 == MAX_TOOL_CALL_ITERATIONS {
            messages.push(json!({
                "role": "system",
                "content": TOOL_LIMIT_SYSTEM_PROMPT
            }));
        }
    }

    Ok("".to_string())
}

pub async fn call_third_party(
    system_prompt: &str,
    user_content: &str,
    model_id: &str,
    response_title: &str,
    image_data_list: &[Vec<u8>],
    supports_tools: bool,
) -> Result<String> {
    if model_id.trim().is_empty() {
        return Err(anyhow!("Model identifier is required"));
    }

    let model_config = CONFIG
        .get_third_party_model_config(model_id)
        .cloned()
        .ok_or_else(|| anyhow!("Unknown third-party model '{}'", model_id))?;
    let message_content = build_message_content(user_content, image_data_list);

    let messages = vec![
        json!({ "role": "system", "content": system_prompt }),
        json!({ "role": "user", "content": message_content }),
    ];

    let operation = format!("{}:{}", model_config.provider.as_str(), response_title);

    if supports_tools && web_search::is_search_enabled() {
        return log_llm_timing(
            model_config.provider.as_str(),
            &model_config.id,
            &operation,
            None,
            || async { chat_completion_with_tools(messages, &model_config).await.map_err(|err| anyhow!(err)) },
        )
        .await;
    }

    let details = build_request_details(&model_config, messages, None, None)?;
    let model_id = model_config.id.clone();
    let result = log_llm_timing(
        model_config.provider.as_str(),
        &model_id,
        &operation,
        None,
        || async {
            let response = call_provider_api(&details).await?;
            let content = response
                .get("choices")
                .and_then(|v| v.get(0))
                .and_then(|v| v.get("message"))
                .map(extract_message_content)
                .unwrap_or_default();
            Ok(parse_third_party_response(&model_config, &content))
        })
        .await;
    result
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
    fn openrouter_request_details_keep_headers_and_top_k() {
        let runtime = ProviderRuntimeConfig {
            provider: ThirdPartyProvider::OpenRouter,
            display_name: "OpenRouter",
            base_url: "https://openrouter.ai/api/v1".to_string(),
            api_key: "test-openrouter".to_string(),
            temperature: 0.7,
            top_p: 0.95,
            top_k: Some(40),
        };
        let details = build_request_details_for_runtime(
            &model(ThirdPartyProvider::OpenRouter, "Qwen 3", "qwen/qwen3-next-80b-a3b-instruct:free"),
            &runtime,
            vec![json!({ "role": "user", "content": "hello" })],
            None,
            None,
        );

        assert_eq!(
            details.url,
            "https://openrouter.ai/api/v1/chat/completions"
        );
        assert!(
            details
                .headers
                .iter()
                .any(|(name, value)| name == "HTTP-Referer" && value == OPENROUTER_REFERER)
        );
        assert!(
            details
                .headers
                .iter()
                .any(|(name, value)| name == "X-Title" && value == OPENROUTER_TITLE)
        );
        assert_eq!(details.payload.get("top_k").and_then(|v| v.as_i64()), Some(40));
    }

    #[test]
    fn nvidia_request_details_omit_top_k_and_openrouter_headers() {
        let runtime = ProviderRuntimeConfig {
            provider: ThirdPartyProvider::Nvidia,
            display_name: "NVIDIA",
            base_url: "https://integrate.api.nvidia.com/v1".to_string(),
            api_key: "test-nvidia".to_string(),
            temperature: 0.4,
            top_p: 0.8,
            top_k: None,
        };
        let details = build_request_details_for_runtime(
            &model(ThirdPartyProvider::Nvidia, "Gemma 3n", "google/gemma-3n-e4b-it"),
            &runtime,
            vec![json!({ "role": "user", "content": "hello" })],
            None,
            None,
        );

        assert_eq!(
            details.url,
            "https://integrate.api.nvidia.com/v1/chat/completions"
        );
        assert!(
            !details
                .headers
                .iter()
                .any(|(name, _)| name == "HTTP-Referer" || name == "X-Title")
        );
        assert!(details.payload.get("top_k").is_none());
    }
}
