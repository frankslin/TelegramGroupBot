use std::time::Duration;

use anyhow::{anyhow, Result};
use base64::{engine::general_purpose, Engine as _};
use reqwest::header::CONTENT_TYPE;
use reqwest::StatusCode;
use serde::de::DeserializeOwned;
use serde::Deserialize;
use serde_json::{json, Map, Value};
use tracing::{debug, info, warn};

use crate::config::CONFIG;
use crate::llm::media::{detect_mime_type, download_media, kind_for_mime, MediaFile, MediaKind};
use crate::llm::tool_runtime::ToolRuntime;
use crate::utils::http::get_http_client;
use crate::utils::timing::log_llm_timing;

#[derive(Debug, thiserror::Error)]
#[error("Image generation failed: {0}")]
pub struct ImageGenerationError(pub String);

#[derive(Debug, Clone)]
pub struct GeminiImageConfig {
    pub aspect_ratio: Option<String>,
    pub image_size: Option<String>,
}

#[derive(Debug, Clone)]
pub struct GeminiCallResult {
    pub text: String,
    pub model_used: String,
}

#[derive(Debug, Clone)]
pub struct GeminiMusicGenerationResult {
    pub lyrics_text: String,
    pub notes_text: Option<String>,
    pub audio_bytes: Vec<u8>,
    pub audio_mime_type: String,
    pub model_used: String,
}

#[derive(Debug, Deserialize)]
struct GeminiResponse {
    candidates: Option<Vec<GeminiCandidate>>,
}

#[derive(Debug, Deserialize)]
struct GeminiCandidate {
    content: Option<GeminiContent>,
}

#[derive(Debug, Deserialize)]
struct GeminiContent {
    parts: Option<Vec<GeminiPart>>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum GeminiPart {
    Text {
        text: String,
    },
    InlineData {
        #[serde(rename = "inlineData")]
        inline_data: GeminiInlineData,
    },
    ExecutableCode {
        #[serde(rename = "executableCode")]
        executable_code: GeminiExecutableCode,
    },
    CodeExecutionResult {
        #[serde(rename = "codeExecutionResult")]
        code_execution_result: GeminiCodeExecutionResult,
    },
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GeminiInlineData {
    mime_type: String,
    data: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GeminiExecutableCode {
    code: Option<String>,
    language: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GeminiCodeExecutionResult {
    output: Option<String>,
    outcome: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GeminiFileInfo {
    name: String,
    uri: String,
    mime_type: Option<String>,
    state: Option<String>,
}

#[derive(Debug, Deserialize)]
struct GeminiFileResponse {
    file: GeminiFileInfo,
}

#[derive(Debug, Clone)]
struct UploadedFileRef {
    uri: String,
}

const GEMINI_MAX_RETRY_ATTEMPTS: usize = 2;
const GEMINI_LITE_FALLBACK_MAX_ATTEMPTS: usize = 3;
const GEMINI_RETRY_BASE_DELAY_MS: u64 = 900;
const GEMINI_DEFAULT_TIMEOUT_SECS: u64 = 90;
const LYRIA_GENERATION_TIMEOUT_SECS: u64 = 240;
const VEO_DEFAULT_RESOLUTION: &str = "1080p";
const VEO_DEFAULT_DURATION_SECONDS: u32 = 8;
const VEO_DEFAULT_ASPECT_RATIO: &str = "16:9";
const VEO_POLL_INTERVAL_SECS: u64 = 20;
const VEO_MAX_POLL_ATTEMPTS: usize = 30;

fn redact_gemini_api_key(text: &str) -> String {
    let key = CONFIG.gemini_api_key.trim();
    if key.is_empty() {
        return text.to_string();
    }
    text.replace(key, "[redacted]")
}

fn gemini_should_retry_error(err: &reqwest::Error) -> bool {
    err.is_timeout() || err.is_connect()
}

fn gemini_should_retry_status(status: StatusCode) -> bool {
    status == StatusCode::TOO_MANY_REQUESTS
        || status == StatusCode::REQUEST_TIMEOUT
        || status.is_server_error()
}

fn gemini_retry_delay(attempt: usize) -> Duration {
    let attempt = attempt.max(1) as u64;
    Duration::from_millis(GEMINI_RETRY_BASE_DELAY_MS.saturating_mul(attempt))
}

fn build_safety_settings() -> Vec<serde_json::Value> {
    let profile = CONFIG.gemini_safety_settings.as_str();
    let threshold = match profile {
        "standard" => "BLOCK_MEDIUM_AND_ABOVE",
        "permissive" => "OFF",
        _ => {
            warn!(
                "Unknown GEMINI_SAFETY_SETTINGS value '{}', using permissive defaults.",
                profile
            );
            "OFF"
        }
    };

    vec![
        json!({ "category": "HARM_CATEGORY_HARASSMENT", "threshold": threshold }),
        json!({ "category": "HARM_CATEGORY_HATE_SPEECH", "threshold": threshold }),
        json!({ "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": threshold }),
        json!({ "category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": threshold }),
        json!({ "category": "HARM_CATEGORY_CIVIC_INTEGRITY", "threshold": threshold }),
    ]
}

fn build_image_config(config: Option<&GeminiImageConfig>) -> Option<Value> {
    let config = config?;
    let mut map = Map::new();

    if let Some(aspect_ratio) = config.aspect_ratio.as_deref() {
        let trimmed = aspect_ratio.trim();
        if !trimmed.is_empty() {
            map.insert("aspectRatio".to_string(), json!(trimmed));
        }
    }

    if let Some(image_size) = config.image_size.as_deref() {
        let trimmed = image_size.trim();
        if !trimmed.is_empty() {
            map.insert("imageSize".to_string(), json!(trimmed));
        }
    }

    if map.is_empty() {
        None
    } else {
        Some(Value::Object(map))
    }
}

fn truncate_for_log(value: &str, limit: usize) -> String {
    if value.chars().count() <= limit {
        return value.to_string();
    }
    let truncated: String = value.chars().take(limit).collect();
    format!("{truncated}... (truncated)")
}

async fn decode_json_response<T: DeserializeOwned>(
    response: reqwest::Response,
    context: &str,
) -> Result<T> {
    let status = response.status();
    let content_type = response
        .headers()
        .get(CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .unwrap_or("unknown")
        .to_string();
    let bytes = response.bytes().await?;
    if bytes.is_empty() {
        return Err(anyhow!(
            "{} returned empty response body (status {}, content-type {})",
            context,
            status,
            content_type
        ));
    }

    match serde_json::from_slice::<T>(&bytes) {
        Ok(value) => Ok(value),
        Err(err) => {
            let body = String::from_utf8_lossy(&bytes);
            let body_summary = truncate_for_log(&body, 4000);
            Err(anyhow!(
                "{} failed to decode JSON (status {}, content-type {}): {} | body={}",
                context,
                status,
                content_type,
                err,
                body_summary
            ))
        }
    }
}

fn decode_file_info_from_value(value: serde_json::Value, context: &str) -> Result<GeminiFileInfo> {
    if let Some(file_value) = value.get("file").cloned() {
        serde_json::from_value::<GeminiFileInfo>(file_value).map_err(|err| {
            anyhow!(
                "{} failed to decode file metadata wrapper: {}",
                context,
                err
            )
        })
    } else {
        serde_json::from_value::<GeminiFileInfo>(value)
            .map_err(|err| anyhow!("{} failed to decode file metadata: {}", context, err))
    }
}

fn summarize_gemini_parts(parts: &[Value]) -> Vec<Value> {
    parts
        .iter()
        .map(|part| {
            if let Some(text) = part.get("text").and_then(|value| value.as_str()) {
                json!({ "text": truncate_for_log(text, 200) })
            } else if let Some(inline_data) = part.get("inlineData") {
                let mime_type = inline_data
                    .get("mimeType")
                    .and_then(|value| value.as_str())
                    .unwrap_or("unknown");
                let data_len = inline_data
                    .get("data")
                    .and_then(|value| value.as_str())
                    .map(|value| value.len())
                    .unwrap_or(0);
                json!({ "inlineData": { "mimeType": mime_type, "dataLen": data_len } })
            } else if let Some(file_data) = part.get("fileData") {
                let file_uri = file_data
                    .get("fileUri")
                    .and_then(|value| value.as_str())
                    .map(|value| truncate_for_log(value, 200));
                let mime_type = file_data
                    .get("mimeType")
                    .and_then(|value| value.as_str())
                    .map(|value| value.to_string());
                json!({ "fileData": { "fileUri": file_uri, "mimeType": mime_type } })
            } else {
                json!({ "unknownPart": true })
            }
        })
        .collect()
}

fn summarize_gemini_payload(payload: &Value, system_prompt_label: Option<&str>) -> Value {
    let mut summary = Map::new();

    if payload.pointer("/systemInstruction").is_some() {
        let label = system_prompt_label.unwrap_or("inline_system_prompt");
        summary.insert(
            "systemInstruction".to_string(),
            Value::String(label.to_string()),
        );
    }

    if let Some(contents) = payload.get("contents").and_then(|value| value.as_array()) {
        let mut summarized_contents = Vec::new();
        for content in contents {
            let role = content
                .get("role")
                .and_then(|value| value.as_str())
                .unwrap_or("user");
            let parts = content
                .get("parts")
                .and_then(|value| value.as_array())
                .map(|parts| summarize_gemini_parts(parts))
                .unwrap_or_default();
            summarized_contents.push(json!({ "role": role, "parts": parts }));
        }
        summary.insert("contents".to_string(), Value::Array(summarized_contents));
    }

    if let Some(config) = payload.get("generationConfig") {
        summary.insert("generationConfig".to_string(), config.clone());
    }

    if let Some(tools) = payload.get("tools") {
        summary.insert("tools".to_string(), tools.clone());
    }

    if let Some(safety) = payload
        .get("safetySettings")
        .and_then(|value| value.as_array())
    {
        summary.insert("safetySettingsCount".to_string(), json!(safety.len()));
    }

    Value::Object(summary)
}

fn summarize_gemini_response(response: &GeminiResponse) -> Value {
    let mut text_parts = 0usize;
    let mut image_parts = 0usize;
    let mut text_preview = None;

    let candidates = response.candidates.as_deref().unwrap_or(&[]);
    for candidate in candidates {
        if let Some(content) = &candidate.content {
            if let Some(parts) = &content.parts {
                for part in parts {
                    match part {
                        GeminiPart::Text { text } => {
                            text_parts += 1;
                            if text_preview.is_none() && !text.trim().is_empty() {
                                text_preview = Some(truncate_for_log(text, 200));
                            }
                        }
                        GeminiPart::InlineData { inline_data } => {
                            if inline_data.mime_type.starts_with("image/") {
                                image_parts += 1;
                            }
                        }
                        GeminiPart::ExecutableCode { executable_code } => {
                            if text_preview.is_none() {
                                if let Some(code) = executable_code.code.as_deref() {
                                    if !code.trim().is_empty() {
                                        text_preview = Some(truncate_for_log(code, 200));
                                    }
                                }
                            }
                        }
                        GeminiPart::CodeExecutionResult {
                            code_execution_result,
                        } => {
                            if text_preview.is_none() {
                                if let Some(output) = code_execution_result.output.as_deref() {
                                    if !output.trim().is_empty() {
                                        text_preview = Some(truncate_for_log(output, 200));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    json!({
        "candidates": response.candidates.as_ref().map(|candidates| candidates.len()).unwrap_or(0),
        "textParts": text_parts,
        "imageParts": image_parts,
        "textPreview": text_preview
    })
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

fn kind_label(kind: MediaKind) -> &'static str {
    match kind {
        MediaKind::Image => "image",
        MediaKind::Video => "video",
        MediaKind::Audio => "audio",
        MediaKind::Document => "document",
    }
}

fn normalize_gemini_mime_type(mime_type: &str) -> String {
    let lowered = mime_type.trim().to_ascii_lowercase();
    match lowered.as_str() {
        "image/jpg" => "image/jpeg".to_string(),
        "audio/mpeg" => "audio/mp3".to_string(),
        "audio/x-wav" => "audio/wav".to_string(),
        "video/quicktime" => "video/mov".to_string(),
        "video/x-msvideo" => "video/avi".to_string(),
        "video/x-ms-wmv" => "video/wmv".to_string(),
        _ => lowered,
    }
}

fn gemini_supports_mime(kind: MediaKind, mime_type: &str) -> bool {
    match kind {
        MediaKind::Image => matches!(
            mime_type,
            "image/png" | "image/jpeg" | "image/webp" | "image/heic" | "image/heif"
        ),
        MediaKind::Video => matches!(
            mime_type,
            "video/mp4"
                | "video/mpeg"
                | "video/mov"
                | "video/avi"
                | "video/x-flv"
                | "video/mpg"
                | "video/webm"
                | "video/wmv"
                | "video/3gpp"
        ),
        MediaKind::Audio => matches!(
            mime_type,
            "audio/wav" | "audio/mp3" | "audio/aiff" | "audio/aac" | "audio/ogg" | "audio/flac"
        ),
        MediaKind::Document => mime_type == "application/pdf",
    }
}

fn gemini_mime_for_file(file: &MediaFile) -> Option<String> {
    let mut candidates = Vec::new();
    if !file.mime_type.trim().is_empty() {
        candidates.push(file.mime_type.clone());
    }
    if let Some(detected) = detect_mime_type(&file.bytes) {
        candidates.push(detected);
    }

    for candidate in candidates {
        let normalized = normalize_gemini_mime_type(&candidate);
        if gemini_supports_mime(file.kind, &normalized) {
            return Some(normalized);
        }
    }

    None
}

async fn upload_file_bytes(
    display_name: &str,
    mime_type: &str,
    bytes: &[u8],
) -> Result<GeminiFileInfo> {
    let client = get_http_client();
    let start_response = client
        .post("https://generativelanguage.googleapis.com/upload/v1beta/files")
        .header("x-goog-api-key", &CONFIG.gemini_api_key)
        .header("X-Goog-Upload-Protocol", "resumable")
        .header("X-Goog-Upload-Command", "start")
        .header(
            "X-Goog-Upload-Header-Content-Length",
            bytes.len().to_string(),
        )
        .header("X-Goog-Upload-Header-Content-Type", mime_type)
        .json(&json!({ "file": { "display_name": display_name } }))
        .send()
        .await?;

    if !start_response.status().is_success() {
        let status = start_response.status();
        let body = start_response.text().await.unwrap_or_default();
        let (message, body_summary) = summarize_error_body(&body);
        let detail = message.unwrap_or(body_summary);
        return Err(anyhow!(
            "Gemini file upload start failed with status {}: {}",
            status,
            detail
        ));
    }

    let upload_url = start_response
        .headers()
        .get("x-goog-upload-url")
        .and_then(|value| value.to_str().ok())
        .ok_or_else(|| anyhow!("Gemini file upload did not return an upload URL"))?;

    let finalize_response = client
        .post(upload_url)
        .header("X-Goog-Upload-Command", "upload, finalize")
        .header("X-Goog-Upload-Offset", "0")
        .header("Content-Length", bytes.len().to_string())
        .body(bytes.to_vec())
        .send()
        .await?;

    if !finalize_response.status().is_success() {
        let status = finalize_response.status();
        let body = finalize_response.text().await.unwrap_or_default();
        let (message, body_summary) = summarize_error_body(&body);
        let detail = message.unwrap_or(body_summary);
        return Err(anyhow!(
            "Gemini file upload failed with status {}: {}",
            status,
            detail
        ));
    }

    let payload =
        decode_json_response::<GeminiFileResponse>(finalize_response, "Gemini file upload").await?;
    Ok(payload.file)
}

async fn get_file_metadata(name: &str) -> Result<GeminiFileInfo> {
    let name = name.trim();
    let name = name.strip_prefix("files/").unwrap_or(name);
    let client = get_http_client();
    let response = client
        .get(format!(
            "https://generativelanguage.googleapis.com/v1beta/files/{}",
            name
        ))
        .header("x-goog-api-key", &CONFIG.gemini_api_key)
        .send()
        .await?;

    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        let (message, body_summary) = summarize_error_body(&body);
        let detail = message.unwrap_or(body_summary);
        return Err(anyhow!(
            "Gemini file metadata fetch failed with status {}: {}",
            status,
            detail
        ));
    }

    let payload =
        decode_json_response::<serde_json::Value>(response, "Gemini file metadata").await?;
    decode_file_info_from_value(payload, "Gemini file metadata")
}

async fn wait_for_file_active(file: GeminiFileInfo) -> Result<GeminiFileInfo> {
    let name = file.name.clone();
    let mut latest = file;

    for _ in 0..15 {
        match latest.state.as_deref().unwrap_or("PROCESSING") {
            "ACTIVE" => return Ok(latest),
            "FAILED" => return Err(anyhow!("Gemini file processing failed for {}", latest.uri)),
            _ => {}
        }

        tokio::time::sleep(Duration::from_secs(2)).await;
        latest = get_file_metadata(&name).await?;
    }

    Err(anyhow!(
        "Timed out waiting for Gemini file processing for {}",
        name
    ))
}

async fn upload_media_files(files: &[MediaFile]) -> Result<Vec<UploadedFileRef>> {
    let mut uploaded = Vec::new();

    for (index, file) in files.iter().enumerate() {
        let display_name = file
            .display_name
            .clone()
            .unwrap_or_else(|| format!("{}-{}", kind_label(file.kind), index + 1));
        if file.bytes.is_empty() {
            warn!("Skipping empty media file {}", display_name);
            continue;
        }
        let Some(mime_type) = gemini_mime_for_file(file) else {
            warn!(
                "Skipping unsupported Gemini media {} (kind={}, mime={})",
                display_name,
                kind_label(file.kind),
                file.mime_type
            );
            continue;
        };
        let info = upload_file_bytes(&display_name, &mime_type, &file.bytes).await?;
        let info = wait_for_file_active(info).await?;
        if let Some(uploaded_mime_type) = info
            .mime_type
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
        {
            debug!(
                display_name = %display_name,
                mime_type = %uploaded_mime_type,
                "Gemini file upload became active"
            );
        }
        let uri = if !info.uri.trim().is_empty() {
            info.uri
        } else if !info.name.trim().is_empty() {
            format!(
                "https://generativelanguage.googleapis.com/files/{}",
                info.name.trim_start_matches("files/")
            )
        } else {
            warn!(
                "Gemini file upload response missing uri/name for {}",
                display_name
            );
            continue;
        };
        uploaded.push(UploadedFileRef { uri });
    }

    Ok(uploaded)
}

fn build_gemini_parts(
    user_content: &str,
    image_data_list: &[Vec<u8>],
    video_data: Option<&[u8]>,
    audio_data: Option<&[u8]>,
    youtube_urls: &[String],
    text_after_media: bool,
) -> Vec<serde_json::Value> {
    let mut parts = Vec::new();
    let text_part = json!({ "text": user_content });

    if !text_after_media {
        parts.push(text_part.clone());
    }

    for url in youtube_urls {
        parts.push(json!({
            "fileData": {
                "fileUri": url
            }
        }));
    }

    for image_data in image_data_list {
        let mime_type = detect_mime_type(image_data).unwrap_or_else(|| "image/png".to_string());
        let encoded = general_purpose::STANDARD.encode(image_data);
        parts.push(json!({
            "inlineData": {
                "mimeType": mime_type,
                "data": encoded
            }
        }));
    }

    if let Some(video_bytes) = video_data {
        let mime_type = detect_mime_type(video_bytes).unwrap_or_else(|| "video/mp4".to_string());
        let encoded = general_purpose::STANDARD.encode(video_bytes);
        parts.push(json!({
            "inlineData": {
                "mimeType": mime_type,
                "data": encoded
            }
        }));
    }

    if let Some(audio_bytes) = audio_data {
        let mime_type = detect_mime_type(audio_bytes).unwrap_or_else(|| "audio/mpeg".to_string());
        let encoded = general_purpose::STANDARD.encode(audio_bytes);
        parts.push(json!({
            "inlineData": {
                "mimeType": mime_type,
                "data": encoded
            }
        }));
    }

    if text_after_media {
        parts.push(text_part);
    }

    parts
}

fn build_gemini_file_parts(
    user_content: &str,
    uploaded_files: &[UploadedFileRef],
    youtube_urls: &[String],
    text_after_media: bool,
) -> Vec<serde_json::Value> {
    let mut parts = Vec::new();
    let text_part = json!({ "text": user_content });

    if !text_after_media {
        parts.push(text_part.clone());
    }

    for file in uploaded_files {
        parts.push(json!({
            "fileData": {
                "fileUri": file.uri
            }
        }));
    }

    for url in youtube_urls {
        parts.push(json!({
            "fileData": {
                "fileUri": url
            }
        }));
    }

    if text_after_media {
        parts.push(text_part);
    }

    parts
}

fn extract_text_from_response(response: GeminiResponse) -> String {
    let mut text_parts = Vec::new();
    let mut fallback_parts = Vec::new();
    for candidate in response.candidates.unwrap_or_default() {
        if let Some(content) = candidate.content {
            if let Some(parts) = content.parts {
                for part in parts {
                    match part {
                        GeminiPart::Text { text } => {
                            if !text.trim().is_empty() {
                                text_parts.push(text);
                            }
                        }
                        GeminiPart::ExecutableCode { executable_code } => {
                            if let Some(language) = executable_code
                                .language
                                .as_deref()
                                .map(str::trim)
                                .filter(|value| !value.is_empty())
                            {
                                debug!(%language, "Gemini response included executable code");
                            }
                            if let Some(code) = executable_code.code.as_deref() {
                                if !code.trim().is_empty() {
                                    fallback_parts.push(code.to_string());
                                }
                            }
                        }
                        GeminiPart::CodeExecutionResult {
                            code_execution_result,
                        } => {
                            if let Some(outcome) = code_execution_result
                                .outcome
                                .as_deref()
                                .map(str::trim)
                                .filter(|value| !value.is_empty())
                            {
                                debug!(%outcome, "Gemini response included code execution result");
                            }
                            if let Some(output) = code_execution_result.output.as_deref() {
                                if !output.trim().is_empty() {
                                    fallback_parts.push(output.to_string());
                                }
                            }
                        }
                        _ => {}
                    }
                }
            }
        }
    }

    if text_parts.is_empty() {
        fallback_parts.join("\n")
    } else {
        text_parts.join("\n")
    }
}

fn extract_images_from_response(response: GeminiResponse) -> Vec<Vec<u8>> {
    let mut images = Vec::new();
    for candidate in response.candidates.unwrap_or_default() {
        if let Some(content) = candidate.content {
            if let Some(parts) = content.parts {
                for part in parts {
                    if let GeminiPart::InlineData { inline_data } = part {
                        if inline_data.mime_type.starts_with("image/") {
                            if let Ok(bytes) = general_purpose::STANDARD.decode(inline_data.data) {
                                images.push(bytes);
                            }
                        }
                    }
                }
            }
        }
    }
    images
}

async fn call_gemini_api(
    model: &str,
    payload: serde_json::Value,
    system_prompt_label: Option<&str>,
) -> Result<GeminiResponse> {
    let value = call_gemini_api_value(model, payload, system_prompt_label).await?;
    let parsed = serde_json::from_value::<GeminiResponse>(value)
        .map_err(|err| anyhow!("Gemini generateContent response decode failed: {}", err))?;
    Ok(parsed)
}

async fn call_gemini_api_with_timeout(
    model: &str,
    payload: serde_json::Value,
    system_prompt_label: Option<&str>,
    timeout: Duration,
) -> Result<GeminiResponse> {
    let value =
        call_gemini_api_value_with_timeout(model, payload, system_prompt_label, timeout).await?;
    let parsed = serde_json::from_value::<GeminiResponse>(value)
        .map_err(|err| anyhow!("Gemini generateContent response decode failed: {}", err))?;
    Ok(parsed)
}

async fn call_gemini_api_value(
    model: &str,
    payload: serde_json::Value,
    system_prompt_label: Option<&str>,
) -> Result<Value> {
    call_gemini_api_value_with_timeout(
        model,
        payload,
        system_prompt_label,
        Duration::from_secs(GEMINI_DEFAULT_TIMEOUT_SECS),
    )
    .await
}

async fn call_gemini_api_value_with_timeout(
    model: &str,
    payload: serde_json::Value,
    system_prompt_label: Option<&str>,
    timeout: Duration,
) -> Result<Value> {
    let client = get_http_client();
    let url = format!(
        "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent?key={}",
        model, CONFIG.gemini_api_key
    );

    if tracing::enabled!(tracing::Level::DEBUG) {
        let payload_summary = summarize_gemini_payload(&payload, system_prompt_label);
        debug!(target: "llm.gemini", model = model, payload = %payload_summary);
    }

    let mut attempt = 0usize;
    loop {
        attempt += 1;
        let response = match client
            .post(&url)
            .timeout(timeout)
            .json(&payload)
            .send()
            .await
        {
            Ok(response) => response,
            Err(err) => {
                let err_text = redact_gemini_api_key(&err.to_string());
                let url = err.url().map(|url| redact_gemini_api_key(url.as_str()));
                let should_retry =
                    gemini_should_retry_error(&err) && attempt < GEMINI_MAX_RETRY_ATTEMPTS;
                warn!(
                    "Gemini request failed to send: {} (timeout={}, connect={}, status={:?}, url={:?}, retrying={})",
                    err_text,
                    err.is_timeout(),
                    err.is_connect(),
                    err.status(),
                    url,
                    should_retry
                );
                if should_retry {
                    tokio::time::sleep(gemini_retry_delay(attempt)).await;
                    continue;
                }
                return Err(anyhow!("Gemini request failed: {}", err_text));
            }
        };

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            let (message, body_summary) = summarize_error_body(&body);
            let should_retry =
                gemini_should_retry_status(status) && attempt < GEMINI_MAX_RETRY_ATTEMPTS;
            warn!(
                "Gemini API error: status={}, body={}, retrying={}",
                status, body_summary, should_retry
            );
            if tracing::enabled!(tracing::Level::DEBUG) {
                debug!(
                    target: "llm.gemini",
                    status = %status,
                    body = %truncate_for_log(&body, 4000)
                );
            }
            if should_retry {
                tokio::time::sleep(gemini_retry_delay(attempt)).await;
                continue;
            }
            let detail = message.unwrap_or(body_summary);
            return Err(anyhow!(
                "Gemini request failed with status {}: {}",
                status,
                detail
            ));
        }

        let value = decode_json_response::<Value>(response, "Gemini generateContent").await?;
        if tracing::enabled!(tracing::Level::DEBUG) {
            let parsed = serde_json::from_value::<GeminiResponse>(value.clone()).ok();
            let response_summary = parsed
                .as_ref()
                .map(summarize_gemini_response)
                .unwrap_or_else(|| {
                    json!({
                        "rawResponsePreview": truncate_for_log(&value.to_string(), 400)
                    })
                });
            debug!(target: "llm.gemini", model = model, response = %response_summary);
        }
        return Ok(value);
    }
}

fn text_part_looks_like_music_metadata(text: &str) -> bool {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return false;
    }

    if trimmed.starts_with("Caption:") {
        return true;
    }

    if (trimmed.starts_with('{') && trimmed.ends_with('}'))
        || (trimmed.starts_with('[') && trimmed.ends_with(']') && trimmed.contains('{'))
    {
        return true;
    }

    let lower = trimmed.to_ascii_lowercase();
    lower.contains("\nbpm:")
        || lower.starts_with("bpm:")
        || lower.contains("\nmusic:")
        || lower.starts_with("music:")
        || lower.contains("\ncaption:")
}

fn extract_music_generation_result(
    response: GeminiResponse,
    model: &str,
) -> Result<GeminiMusicGenerationResult> {
    let mut lyric_parts = Vec::new();
    let mut note_parts = Vec::new();
    let mut audio_bytes = None;
    let mut audio_mime_type = None;

    for candidate in response.candidates.unwrap_or_default() {
        if let Some(content) = candidate.content {
            if let Some(parts) = content.parts {
                for part in parts {
                    match part {
                        GeminiPart::Text { text } => {
                            let trimmed = text.trim();
                            if trimmed.is_empty() {
                                continue;
                            }

                            if text_part_looks_like_music_metadata(trimmed) {
                                note_parts.push(trimmed.to_string());
                            } else {
                                lyric_parts.push(trimmed.to_string());
                            }
                        }
                        GeminiPart::InlineData { inline_data } => {
                            if inline_data.mime_type.starts_with("audio/") {
                                let bytes = general_purpose::STANDARD
                                    .decode(&inline_data.data)
                                    .map_err(|err| {
                                        anyhow!("Failed to decode Lyria audio payload: {}", err)
                                    })?;
                                audio_mime_type = Some(inline_data.mime_type);
                                audio_bytes = Some(bytes);
                            }
                        }
                        _ => {}
                    }
                }
            }
        }
    }

    if lyric_parts.is_empty() && !note_parts.is_empty() {
        lyric_parts = note_parts.clone();
        note_parts.clear();
    }

    let audio_bytes =
        audio_bytes.ok_or_else(|| anyhow!("No audio returned by Lyria (model: {})", model))?;

    Ok(GeminiMusicGenerationResult {
        lyrics_text: lyric_parts.join("\n\n"),
        notes_text: if note_parts.is_empty() {
            None
        } else {
            Some(note_parts.join("\n\n"))
        },
        audio_bytes,
        audio_mime_type: audio_mime_type.unwrap_or_else(|| "audio/mpeg".to_string()),
        model_used: model.to_string(),
    })
}

fn extract_text_from_response_value(response: &Value) -> String {
    let mut text_parts = Vec::new();
    let mut fallback_parts = Vec::new();
    let candidates = response
        .get("candidates")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();

    for candidate in candidates {
        let parts = candidate
            .get("content")
            .and_then(|content| content.get("parts"))
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        for part in parts {
            if let Some(text) = part.get("text").and_then(Value::as_str) {
                if !text.trim().is_empty() {
                    text_parts.push(text.to_string());
                }
            } else if let Some(code) = part
                .get("executableCode")
                .and_then(|value| value.get("code"))
                .and_then(Value::as_str)
            {
                if !code.trim().is_empty() {
                    fallback_parts.push(code.to_string());
                }
            } else if let Some(output) = part
                .get("codeExecutionResult")
                .and_then(|value| value.get("output"))
                .and_then(Value::as_str)
            {
                if !output.trim().is_empty() {
                    fallback_parts.push(output.to_string());
                }
            }
        }
    }

    if text_parts.is_empty() {
        fallback_parts.join("\n")
    } else {
        text_parts.join("\n")
    }
}

fn extract_candidate_content(response: &Value) -> Option<Value> {
    response
        .get("candidates")
        .and_then(Value::as_array)
        .and_then(|candidates| candidates.first())
        .and_then(|candidate| candidate.get("content"))
        .cloned()
}

fn extract_function_calls(content: &Value) -> Vec<Value> {
    content
        .get("parts")
        .and_then(Value::as_array)
        .map(|parts| {
            parts
                .iter()
                .filter_map(|part| part.get("functionCall").cloned())
                .collect::<Vec<_>>()
        })
        .unwrap_or_default()
}

fn base_generation_config() -> Value {
    json!({
        "temperature": CONFIG.gemini_temperature,
        "topK": CONFIG.gemini_top_k,
        "topP": CONFIG.gemini_top_p,
        "maxOutputTokens": CONFIG.gemini_max_output_tokens,
    })
}

fn with_response_json_schema(config: Value, response_json_schema: Option<&Value>) -> Value {
    let Some(schema) = response_json_schema else {
        return config;
    };

    let mut config_object = config.as_object().cloned().unwrap_or_default();
    config_object.insert(
        "responseMimeType".to_string(),
        Value::String("application/json".to_string()),
    );
    config_object.insert("responseJsonSchema".to_string(), schema.clone());
    Value::Object(config_object)
}

fn build_function_response_part(function_call: &Value, result_json: &str) -> Value {
    let name = function_call
        .get("name")
        .and_then(Value::as_str)
        .unwrap_or("");
    let id = function_call
        .get("id")
        .and_then(Value::as_str)
        .map(|value| value.to_string());
    let response_value = serde_json::from_str::<Value>(result_json)
        .unwrap_or_else(|_| json!({ "result": result_json }));

    let mut function_response = json!({
        "name": name,
        "response": response_value,
    });
    if let Some(id) = id {
        function_response["id"] = Value::String(id);
    }

    json!({ "functionResponse": function_response })
}

#[allow(clippy::too_many_arguments)]
pub async fn call_gemini_with_tool_runtime(
    system_prompt: &str,
    user_content: &str,
    runtime: &mut ToolRuntime,
    use_pro_model: bool,
    media_files: Option<Vec<MediaFile>>,
    youtube_urls: Option<Vec<String>>,
    system_prompt_label: Option<&str>,
    final_response_json_schema: Option<Value>,
) -> Result<GeminiCallResult> {
    let content = user_content.to_string();
    let youtube_urls = youtube_urls.unwrap_or_default();
    let files = media_files.unwrap_or_default();
    let uploaded_files = if files.is_empty() {
        Vec::new()
    } else {
        upload_media_files(&files).await?
    };
    let text_after_media = !uploaded_files.is_empty() || !youtube_urls.is_empty();
    let parts = build_gemini_file_parts(&content, &uploaded_files, &youtube_urls, text_after_media);
    let mut contents = vec![json!({ "role": "user", "parts": parts })];

    let model = if use_pro_model {
        CONFIG.gemini_pro_model.as_str()
    } else {
        CONFIG.gemini_model.as_str()
    };
    let tool_limit_turns = runtime.max_total_successful_calls().saturating_add(2);
    let mut tools_enabled = true;

    for _ in 0..tool_limit_turns {
        let mut payload = json!({
            "systemInstruction": { "parts": [{ "text": system_prompt }] },
            "contents": contents.clone(),
            "generationConfig": base_generation_config(),
            "safetySettings": build_safety_settings(),
        });
        if tools_enabled {
            payload["tools"] = Value::Array(runtime.build_gemini_tools());
        }

        let response = call_gemini_api_value(model, payload, system_prompt_label).await?;
        let Some(content) = extract_candidate_content(&response) else {
            return Err(anyhow!(
                "Gemini tool response did not include candidate content"
            ));
        };
        let function_calls = if tools_enabled {
            extract_function_calls(&content)
        } else {
            Vec::new()
        };

        if function_calls.is_empty() {
            if final_response_json_schema.is_none() {
                return Ok(GeminiCallResult {
                    text: extract_text_from_response_value(&response),
                    model_used: model.to_string(),
                });
            }
            break;
        }

        contents.push(content);

        let mut response_parts = Vec::new();
        for function_call in function_calls {
            let tool_name = function_call
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or("");
            let args = function_call
                .get("args")
                .cloned()
                .unwrap_or_else(|| json!({}));
            let result_json = runtime.execute_tool(tool_name, &args).await;
            response_parts.push(build_function_response_part(&function_call, &result_json));
        }

        contents.push(json!({
            "role": "user",
            "parts": response_parts,
        }));

        if runtime.force_final_answer() {
            tools_enabled = false;
        }
    }

    let final_payload = json!({
        "systemInstruction": { "parts": [{ "text": system_prompt }] },
        "contents": contents.clone(),
        "generationConfig": with_response_json_schema(
            base_generation_config(),
            final_response_json_schema.as_ref()
        ),
        "safetySettings": build_safety_settings(),
    });
    let final_response = call_gemini_api_value(model, final_payload, system_prompt_label).await?;
    Ok(GeminiCallResult {
        text: extract_text_from_response_value(&final_response),
        model_used: model.to_string(),
    })
}

async fn call_gemini_lite_fallback(
    payload: &serde_json::Value,
    system_prompt_label: Option<&str>,
    previous_model: &str,
    previous_err: &anyhow::Error,
) -> Result<GeminiCallResult> {
    let lite_model = CONFIG.gemini_lite_model.trim();
    if lite_model.is_empty() {
        return Err(anyhow!(
            "Gemini request failed on model '{}' and GEMINI_LITE_MODEL is not configured. Previous error: {}",
            previous_model,
            previous_err
        ));
    }

    if lite_model.eq_ignore_ascii_case(previous_model) {
        return Err(anyhow!(
            "Gemini request failed on model '{}' and GEMINI_LITE_MODEL points to the same model. Previous error: {}",
            previous_model,
            previous_err
        ));
    }

    warn!(
        "Gemini model '{}' failed after retries; trying lite fallback model '{}' for up to {} attempts: {}",
        previous_model,
        lite_model,
        GEMINI_LITE_FALLBACK_MAX_ATTEMPTS,
        previous_err
    );

    let mut last_lite_err = None;
    for attempt in 1..=GEMINI_LITE_FALLBACK_MAX_ATTEMPTS {
        let result = log_llm_timing(
            "gemini",
            lite_model,
            "call_gemini_lite_fallback",
            Some(json!({
                "attempt": attempt,
                "max_attempts": GEMINI_LITE_FALLBACK_MAX_ATTEMPTS,
                "after_model": previous_model,
            })),
            || async {
                let response =
                    call_gemini_api(lite_model, payload.clone(), system_prompt_label).await?;
                Ok(extract_text_from_response(response))
            },
        )
        .await;

        match result {
            Ok(text) => {
                return Ok(GeminiCallResult {
                    text,
                    model_used: lite_model.to_string(),
                });
            }
            Err(err) => {
                warn!(
                    "Gemini lite fallback attempt {}/{} failed on model '{}': {}",
                    attempt, GEMINI_LITE_FALLBACK_MAX_ATTEMPTS, lite_model, err
                );
                last_lite_err = Some(err);
            }
        }
    }

    let lite_err = last_lite_err.unwrap_or_else(|| anyhow!("Unknown Gemini lite fallback failure"));
    Err(anyhow!(
        "Gemini request failed on model '{}' and lite fallback model '{}' after {} attempts. Previous error: {}. Lite fallback error: {}",
        previous_model,
        lite_model,
        GEMINI_LITE_FALLBACK_MAX_ATTEMPTS,
        previous_err,
        lite_err
    ))
}

#[allow(clippy::too_many_arguments)]
pub async fn call_gemini(
    system_prompt: &str,
    user_content: &str,
    use_search_grounding: bool,
    _use_url_context: bool,
    _thinking_level: Option<&str>,
    image_url: Option<&str>,
    use_pro_model: bool,
    media_files: Option<Vec<MediaFile>>,
    youtube_urls: Option<Vec<String>>,
    system_prompt_label: Option<&str>,
) -> Result<GeminiCallResult> {
    let content = user_content.to_string();

    let youtube_urls = youtube_urls.unwrap_or_default();

    let mut files = media_files.unwrap_or_default();
    if files.is_empty() {
        if let Some(url) = image_url {
            if let Some(data) = download_media(url).await {
                let mime_type = detect_mime_type(&data).unwrap_or_else(|| "image/png".to_string());
                files.push(MediaFile::new(
                    data,
                    mime_type.clone(),
                    kind_for_mime(&mime_type),
                    None,
                ));
            }
        }
    }

    let has_video_or_audio = files
        .iter()
        .any(|file| matches!(file.kind, MediaKind::Video | MediaKind::Audio));

    let uploaded_files = if files.is_empty() {
        Vec::new()
    } else {
        upload_media_files(&files).await?
    };

    let text_after_media = !uploaded_files.is_empty() || !youtube_urls.is_empty();
    let parts = build_gemini_file_parts(&content, &uploaded_files, &youtube_urls, text_after_media);
    let tools = {
        let mut tools = Vec::new();
        if !has_video_or_audio && youtube_urls.is_empty() {
            tools.push(json!({ "code_execution": {} }));
        }
        if use_search_grounding {
            tools.push(json!({ "google_search": {} }));
        }
        tools
    };

    let payload = json!({
        "systemInstruction": { "parts": [{ "text": system_prompt }] },
        "contents": [{ "role": "user", "parts": parts }],
        "generationConfig": {
            "temperature": CONFIG.gemini_temperature,
            "topK": CONFIG.gemini_top_k,
            "topP": CONFIG.gemini_top_p,
            "maxOutputTokens": CONFIG.gemini_max_output_tokens,
        },
        "safetySettings": build_safety_settings(),
        "tools": tools,
    });

    let primary_model = if use_pro_model {
        &CONFIG.gemini_pro_model
    } else {
        &CONFIG.gemini_model
    };
    let primary_operation = if use_pro_model {
        "call_gemini_pro"
    } else {
        "call_gemini"
    };

    let primary_attempt =
        log_llm_timing("gemini", primary_model, primary_operation, None, || async {
            let response =
                call_gemini_api(primary_model, payload.clone(), system_prompt_label).await?;
            Ok(extract_text_from_response(response))
        })
        .await;

    match primary_attempt {
        Ok(text) => Ok(GeminiCallResult {
            text,
            model_used: primary_model.to_string(),
        }),
        Err(primary_err) => {
            if !use_pro_model {
                return call_gemini_lite_fallback(
                    &payload,
                    system_prompt_label,
                    primary_model,
                    &primary_err,
                )
                .await;
            }

            let fallback_model = CONFIG.gemini_model.as_str();
            warn!(
                "Gemini Pro model '{}' failed after retries; falling back to default model '{}': {}",
                primary_model, fallback_model, primary_err
            );

            let fallback_text = log_llm_timing(
                "gemini",
                fallback_model,
                "call_gemini_fallback",
                None,
                || async {
                    let response =
                        call_gemini_api(fallback_model, payload.clone(), system_prompt_label)
                            .await?;
                    Ok(extract_text_from_response(response))
                },
            )
            .await;

            let fallback_text = match fallback_text {
                Ok(text) => text,
                Err(fallback_err) => {
                    return call_gemini_lite_fallback(
                        &payload,
                        system_prompt_label,
                        fallback_model,
                        &fallback_err,
                    )
                    .await
                    .map_err(|lite_err| {
                        anyhow!(
                            "Gemini request failed on primary model '{}' and fallback model '{}'. \
Primary error: {}. Fallback error: {}. Lite fallback error: {}",
                            primary_model,
                            fallback_model,
                            primary_err,
                            fallback_err,
                            lite_err
                        )
                    });
                }
            };

            Ok(GeminiCallResult {
                text: fallback_text,
                model_used: fallback_model.to_string(),
            })
        }
    }
}

pub async fn generate_image_with_gemini(
    prompt: &str,
    image_urls: &[String],
    image_config: Option<GeminiImageConfig>,
    upload_to_cwd: bool,
) -> Result<Vec<Vec<u8>>, ImageGenerationError> {
    let mut images = Vec::new();
    for url in image_urls {
        if let Some(data) = download_media(url).await {
            images.push(data);
        }
    }

    let base_instruction = if images.is_empty() {
        "Generate an image based on the prompt. CRITICAL: response be an image, NOT TEXT."
    } else {
        "Edit the images based on the prompt. CRITICAL: response be an image, NOT TEXT."
    };

    let system_instruction = base_instruction.to_string();
    let parts = build_gemini_parts(prompt, &images, None, None, &[], false);
    let mut generation_config = json!({
        "responseModalities": ["IMAGE"]
    });
    if let Some(image_config) = build_image_config(image_config.as_ref()) {
        if let Some(config_object) = generation_config.as_object_mut() {
            config_object.insert("imageConfig".to_string(), image_config);
        }
    }

    let payload = json!({
        "systemInstruction": { "parts": [{ "text": system_instruction }] },
        "contents": [{ "role": "user", "parts": parts }],
        "generationConfig": generation_config,
        "safetySettings": build_safety_settings(),
        "tools": [{ "google_search": {"searchTypes": {"webSearch": {}, "imageSearch": {}}} }],
    });

    let model = &CONFIG.gemini_image_model;
    let response = call_gemini_api(model, payload, Some("image_generation_system_prompt"))
        .await
        .map_err(|err| ImageGenerationError(err.to_string()))?;

    let images = extract_images_from_response(response);
    if images.is_empty() {
        return Err(ImageGenerationError(format!(
            "No images returned by Gemini (model: {})",
            model
        )));
    }

    if upload_to_cwd && !CONFIG.cwd_pw_api_key.trim().is_empty() {
        for image in &images {
            let mime_type = detect_mime_type(image).unwrap_or_else(|| "image/png".to_string());
            let _ = crate::tools::cwd_uploader::upload_image_bytes_to_cwd(
                image,
                &CONFIG.cwd_pw_api_key,
                &mime_type,
                Some(model.as_str()),
                Some(prompt),
            )
            .await;
        }
    }

    Ok(images)
}

pub async fn generate_music_with_lyria(
    prompt: &str,
) -> Result<GeminiMusicGenerationResult, anyhow::Error> {
    let prompt = prompt.trim();
    if prompt.is_empty() {
        return Err(anyhow!("Music prompt is empty"));
    }

    let model = CONFIG.gemini_music_model.trim();
    if model.is_empty() {
        return Err(anyhow!("GEMINI_MUSIC_MODEL is not configured"));
    }

    let payload = json!({
        "contents": [{
            "role": "user",
            "parts": [{ "text": prompt }]
        }],
        "generationConfig": {
            "responseModalities": ["AUDIO", "TEXT"]
        },
        "safetySettings": build_safety_settings(),
    });

    let response = log_llm_timing(
        "gemini",
        model,
        "lyria_generate_content",
        Some(json!({
            "response_modalities": ["AUDIO", "TEXT"]
        })),
        || async {
            call_gemini_api_with_timeout(
                model,
                payload.clone(),
                Some("lyria_music_generation"),
                Duration::from_secs(LYRIA_GENERATION_TIMEOUT_SECS),
            )
            .await
        },
    )
    .await?;

    extract_music_generation_result(response, model)
}

pub async fn generate_video_with_veo(
    user_prompt: &str,
) -> Result<(Option<Vec<u8>>, Option<String>), anyhow::Error> {
    let prompt = user_prompt.trim();
    if prompt.is_empty() {
        return Ok((None, None));
    }

    let model = CONFIG.gemini_video_model.trim();
    if model.is_empty() {
        return Err(anyhow!("GEMINI_VIDEO_MODEL is not configured"));
    }

    let mut instance = Map::new();
    instance.insert("prompt".to_string(), json!(prompt));

    let mut parameters = Map::new();
    parameters.insert("resolution".to_string(), json!(VEO_DEFAULT_RESOLUTION));
    parameters.insert(
        "durationSeconds".to_string(),
        json!(VEO_DEFAULT_DURATION_SECONDS),
    );
    parameters.insert("aspectRatio".to_string(), json!(VEO_DEFAULT_ASPECT_RATIO));

    let payload = json!({
        "instances": [Value::Object(instance)],
        "parameters": Value::Object(parameters),
    });

    let client = get_http_client();
    let url = format!(
        "https://generativelanguage.googleapis.com/v1beta/models/{}:predictLongRunning",
        model
    );
    let metadata = json!({
        "resolution": VEO_DEFAULT_RESOLUTION,
        "duration_seconds": VEO_DEFAULT_DURATION_SECONDS,
    });

    let operation = log_llm_timing(
        "gemini",
        model,
        "veo_predict_long_running",
        Some(metadata),
        || async {
            let response = client
                .post(&url)
                .header("x-goog-api-key", &CONFIG.gemini_api_key)
                .json(&payload)
                .send()
                .await?;

            if !response.status().is_success() {
                let status = response.status();
                let body = response.text().await.unwrap_or_default();
                let (message, body_summary) = summarize_error_body(&body);
                let detail = message.unwrap_or(body_summary);
                return Err(anyhow!(
                    "Veo predictLongRunning failed with status {}: {}",
                    status,
                    detail
                ));
            }

            decode_json_response::<Value>(response, "Veo predictLongRunning").await
        },
    )
    .await?;

    let operation_name = operation
        .get("name")
        .and_then(|value| value.as_str())
        .ok_or_else(|| anyhow!("Veo operation response missing name"))?
        .to_string();

    let operation_url = format!(
        "https://generativelanguage.googleapis.com/v1beta/{}",
        operation_name
    );

    let mut current_operation = operation;
    for attempt in 0..VEO_MAX_POLL_ATTEMPTS {
        let done = current_operation
            .get("done")
            .and_then(|value| value.as_bool())
            .unwrap_or(false);
        if done {
            if let Some(error) = current_operation.get("error") {
                let message = error
                    .get("message")
                    .and_then(|value| value.as_str())
                    .unwrap_or("unknown error");
                warn!("Veo operation failed: {}", message);
                return Ok((None, None));
            }

            let video = current_operation
                .pointer("/response/generateVideoResponse/generatedSamples/0/video");
            let video_uri = video
                .and_then(|value| value.get("uri"))
                .and_then(|value| value.as_str());
            let mut mime_type = video
                .and_then(|value| value.get("mimeType"))
                .and_then(|value| value.as_str())
                .map(|value| value.to_string());

            let Some(video_uri) = video_uri else {
                warn!("Veo operation completed without a video uri");
                return Ok((None, None));
            };

            let response = client
                .get(video_uri)
                .header("x-goog-api-key", &CONFIG.gemini_api_key)
                .send()
                .await?;
            if !response.status().is_success() {
                let status = response.status();
                let body = response.text().await.unwrap_or_default();
                let (message, body_summary) = summarize_error_body(&body);
                let detail = message.unwrap_or(body_summary);
                return Err(anyhow!(
                    "Veo video download failed with status {}: {}",
                    status,
                    detail
                ));
            }

            if mime_type.is_none() {
                mime_type = response
                    .headers()
                    .get(CONTENT_TYPE)
                    .and_then(|value| value.to_str().ok())
                    .map(|value| value.to_string());
            }

            let bytes = response.bytes().await?;
            info!(
                "Veo video download completed (bytes={}, mime={:?})",
                bytes.len(),
                mime_type
            );
            return Ok((Some(bytes.to_vec()), mime_type));
        }

        if attempt + 1 < VEO_MAX_POLL_ATTEMPTS {
            info!(
                "Polling Veo operation (attempt {}/{})",
                attempt + 1,
                VEO_MAX_POLL_ATTEMPTS
            );
            tokio::time::sleep(Duration::from_secs(VEO_POLL_INTERVAL_SECS)).await;
            let response = client
                .get(&operation_url)
                .header("x-goog-api-key", &CONFIG.gemini_api_key)
                .send()
                .await?;
            if !response.status().is_success() {
                let status = response.status();
                let body = response.text().await.unwrap_or_default();
                let (message, body_summary) = summarize_error_body(&body);
                let detail = message.unwrap_or(body_summary);
                return Err(anyhow!(
                    "Veo operation poll failed with status {}: {}",
                    status,
                    detail
                ));
            }
            current_operation =
                decode_json_response::<Value>(response, "Veo operation poll").await?;
        }
    }

    warn!("Veo operation timed out after polling");
    Ok((None, None))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_music_generation_result_collects_lyrics_audio_and_notes() {
        let response: GeminiResponse = serde_json::from_value(json!({
            "candidates": [{
                "content": {
                    "parts": [
                        { "text": "[Intro]\nA bright beginning" },
                        { "text": "Caption: upbeat indie pop with layered harmonies\nBPM: 112" },
                        {
                            "inlineData": {
                                "mimeType": "audio/mpeg",
                                "data": general_purpose::STANDARD.encode(b"song-bytes")
                            }
                        }
                    ]
                }
            }]
        }))
        .expect("valid music response");

        let result =
            extract_music_generation_result(response, "lyria-3-pro-preview").expect("music");

        assert_eq!(result.lyrics_text, "[Intro]\nA bright beginning");
        assert_eq!(
            result.notes_text.as_deref(),
            Some("Caption: upbeat indie pop with layered harmonies\nBPM: 112")
        );
        assert_eq!(result.audio_bytes, b"song-bytes");
        assert_eq!(result.audio_mime_type, "audio/mpeg");
        assert_eq!(result.model_used, "lyria-3-pro-preview");
    }

    #[test]
    fn extract_music_generation_result_errors_when_audio_is_missing() {
        let response: GeminiResponse = serde_json::from_value(json!({
            "candidates": [{
                "content": {
                    "parts": [
                        { "text": "[Verse]\nNo audio returned" }
                    ]
                }
            }]
        }))
        .expect("valid music response");

        let err = extract_music_generation_result(response, "lyria-3-pro-preview").unwrap_err();

        assert!(err
            .to_string()
            .contains("No audio returned by Lyria (model: lyria-3-pro-preview)"));
    }
}
