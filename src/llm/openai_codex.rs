use std::fs;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{anyhow, Context, Result};
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine as _};
use chrono::{DateTime, Utc};
use once_cell::sync::Lazy;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tokio::sync::Mutex;
use tracing::{error, info, warn};

use crate::config::CONFIG;
use crate::utils::http::get_http_client;

pub const OPENAI_CODEX_CLIENT_ID: &str = "app_EMoamEEZ73f0CkXaXp7hrann";
pub const OPENAI_CODEX_DEFAULT_ISSUER: &str = "https://auth.openai.com";
const OPENAI_CODEX_DEFAULT_ORIGINATOR: &str = "codex_cli_rs";
const TOKEN_REFRESH_INTERVAL_DAYS: i64 = 8;
const ACCESS_TOKEN_REFRESH_WINDOW_MINUTES: i64 = 5;

static TOKEN_REFRESH_LOCK: Lazy<Mutex<()>> = Lazy::new(|| Mutex::new(()));

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct OpenAICodexAuthFile {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub auth_mode: Option<String>,
    #[serde(
        rename = "OPENAI_API_KEY",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub openai_api_key: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tokens: Option<OpenAICodexTokenData>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_refresh: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct OpenAICodexTokenData {
    pub id_token: String,
    pub access_token: String,
    pub refresh_token: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub account_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub plan_type: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub user_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub email: Option<String>,
}

#[derive(Debug, Clone)]
pub struct OpenAICodexAuthContext {
    pub access_token: String,
    pub account_id: String,
}

#[derive(Debug, Clone)]
pub struct DeviceCodeStart {
    pub verification_url: String,
    pub user_code: String,
    device_auth_id: String,
    interval_seconds: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CodexModelVisibility {
    List,
    Hide,
    None,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CodexInputModality {
    Text,
    Image,
}

fn default_input_modalities() -> Vec<CodexInputModality> {
    vec![CodexInputModality::Text, CodexInputModality::Image]
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CodexReasoningEffortOption {
    pub effort: String,
    pub description: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum CodexWebSearchToolType {
    #[default]
    Text,
    TextAndImage,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CodexRemoteModel {
    pub slug: String,
    pub display_name: String,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub default_reasoning_level: Option<String>,
    #[serde(default)]
    pub supported_reasoning_levels: Vec<CodexReasoningEffortOption>,
    pub visibility: CodexModelVisibility,
    pub supported_in_api: bool,
    pub priority: i32,
    #[serde(default)]
    pub web_search_tool_type: CodexWebSearchToolType,
    #[serde(default = "default_input_modalities")]
    pub input_modalities: Vec<CodexInputModality>,
    #[serde(default)]
    pub supports_search_tool: bool,
}

#[derive(Debug, Clone)]
pub struct CodexModelList {
    pub models: Vec<CodexRemoteModel>,
    pub etag: Option<String>,
}

#[derive(Debug, Clone)]
pub struct OpenAICodexAuthSummary {
    pub auth_mode: Option<String>,
    pub plan_type: Option<String>,
    pub account_id: Option<String>,
    pub email: Option<String>,
    pub last_refresh: Option<DateTime<Utc>>,
    pub auth_file_exists: bool,
}

#[derive(Debug, Clone)]
pub struct CodexUsageSnapshot {
    pub plan_type: Option<String>,
    pub primary: Option<CodexUsageWindow>,
    pub secondary: Option<CodexUsageWindow>,
    pub credits: Option<CodexUsageCredits>,
    pub additional_limits: Vec<CodexAdditionalUsageLimit>,
}

#[derive(Debug, Clone)]
pub struct CodexUsageWindow {
    pub used_percent: f64,
    pub limit_window_seconds: Option<i64>,
    pub reset_at: Option<i64>,
}

#[derive(Debug, Clone)]
pub struct CodexUsageCredits {
    pub has_credits: bool,
    pub unlimited: bool,
    pub balance: Option<String>,
}

#[derive(Debug, Clone)]
pub struct CodexAdditionalUsageLimit {
    pub limit_name: String,
    pub metered_feature: String,
    pub primary: Option<CodexUsageWindow>,
    pub secondary: Option<CodexUsageWindow>,
}

#[derive(Debug, Clone)]
struct ParsedIdToken {
    email: Option<String>,
    plan_type: Option<String>,
    user_id: Option<String>,
    account_id: Option<String>,
}

#[derive(Debug, Deserialize)]
struct DeviceCodeUserCodeResponse {
    device_auth_id: String,
    #[serde(alias = "user_code", alias = "usercode")]
    user_code: String,
    #[serde(default, deserialize_with = "deserialize_interval_seconds")]
    interval: u64,
}

#[derive(Debug, Deserialize)]
struct DeviceCodeTokenResponse {
    authorization_code: String,
    #[serde(rename = "code_challenge")]
    _code_challenge: String,
    code_verifier: String,
}

#[derive(Debug, Deserialize)]
struct OAuthTokenResponse {
    id_token: String,
    access_token: String,
    refresh_token: String,
}

#[derive(Debug, Deserialize)]
struct RefreshTokenResponse {
    #[serde(default)]
    id_token: Option<String>,
    #[serde(default)]
    access_token: Option<String>,
    #[serde(default)]
    refresh_token: Option<String>,
}

#[derive(Debug, Deserialize)]
struct CodexModelsResponse {
    models: Vec<CodexRemoteModel>,
}

#[derive(Debug, Clone, Deserialize)]
struct CodexUsageResponse {
    #[serde(default)]
    plan_type: Option<String>,
    #[serde(default)]
    rate_limit: Option<CodexUsageRateLimitDetails>,
    #[serde(default)]
    additional_rate_limits: Option<Vec<CodexUsageAdditionalRateLimit>>,
    #[serde(default)]
    credits: Option<CodexUsageCreditsDetails>,
}

#[derive(Debug, Clone, Deserialize)]
struct CodexUsageRateLimitDetails {
    #[serde(default)]
    primary_window: Option<CodexUsageWindowSnapshot>,
    #[serde(default)]
    secondary_window: Option<CodexUsageWindowSnapshot>,
}

#[derive(Debug, Clone, Deserialize)]
struct CodexUsageWindowSnapshot {
    used_percent: f64,
    #[serde(default)]
    limit_window_seconds: Option<i64>,
    #[serde(default)]
    reset_at: Option<i64>,
}

#[derive(Debug, Clone, Deserialize)]
struct CodexUsageAdditionalRateLimit {
    limit_name: String,
    metered_feature: String,
    #[serde(default)]
    rate_limit: Option<CodexUsageRateLimitDetails>,
}

#[derive(Debug, Clone, Deserialize)]
struct CodexUsageCreditsDetails {
    #[serde(default)]
    has_credits: bool,
    #[serde(default)]
    unlimited: bool,
    #[serde(default)]
    balance: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CodexWebSearchMode {
    Disabled,
    Cached,
    Live,
}

impl CodexWebSearchMode {
    pub fn from_config(value: &str) -> Self {
        match value.trim().to_ascii_lowercase().as_str() {
            "cached" => Self::Cached,
            "live" => Self::Live,
            _ => Self::Disabled,
        }
    }
}

#[derive(Debug, Serialize)]
struct DeviceCodeUserCodeRequest<'a> {
    client_id: &'a str,
}

#[derive(Debug, Serialize)]
struct DeviceCodeTokenRequest<'a> {
    device_auth_id: &'a str,
    user_code: &'a str,
}

#[derive(Debug, Deserialize)]
struct StandardJwtClaims {
    #[serde(default)]
    exp: Option<i64>,
}

fn deserialize_interval_seconds<'de, D>(deserializer: D) -> Result<u64, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let value = String::deserialize(deserializer)?;
    value
        .trim()
        .parse::<u64>()
        .map_err(serde::de::Error::custom)
}

fn auth_file_path() -> &'static Path {
    Path::new(&CONFIG.openai_codex_auth_path)
}

fn current_originator() -> String {
    let configured = CONFIG.openai_codex_originator.trim();
    if configured.is_empty() {
        OPENAI_CODEX_DEFAULT_ORIGINATOR.to_string()
    } else {
        configured.to_string()
    }
}

fn current_user_agent() -> String {
    format!("{}/{}", env!("CARGO_PKG_NAME"), env!("CARGO_PKG_VERSION"))
}

fn truncate_for_log(value: &str, limit: usize) -> String {
    if value.chars().count() <= limit {
        return value.to_string();
    }
    let truncated: String = value.chars().take(limit).collect();
    format!("{truncated}... (truncated)")
}

fn summarize_error_body(body: &str) -> String {
    let trimmed = body.trim();
    if trimmed.is_empty() {
        return "empty response body".to_string();
    }

    if let Ok(value) = serde_json::from_str::<Value>(trimmed) {
        return truncate_for_log(&value.to_string(), 2000);
    }

    truncate_for_log(trimmed, 2000)
}

fn load_auth_file_internal() -> Result<Option<OpenAICodexAuthFile>> {
    let path = auth_file_path();
    let raw = match fs::read_to_string(path) {
        Ok(raw) => raw,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(err) => {
            return Err(anyhow!(
                "Failed to read Codex auth file {}: {}",
                path.display(),
                err
            ));
        }
    };

    Ok(Some(
        serde_json::from_str::<OpenAICodexAuthFile>(&raw)
            .with_context(|| format!("Failed to parse Codex auth file {}", path.display()))?,
    ))
}

fn save_auth_file(auth: &OpenAICodexAuthFile) -> Result<()> {
    let path = auth_file_path();
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, serde_json::to_string_pretty(auth)?)?;
    Ok(())
}

pub fn is_auth_ready() -> bool {
    load_auth_file_internal()
        .ok()
        .flatten()
        .and_then(|auth| auth.tokens)
        .is_some()
}

pub fn auth_summary() -> OpenAICodexAuthSummary {
    let path = auth_file_path();
    let auth = load_auth_file_internal().ok().flatten();
    let tokens = auth.as_ref().and_then(|auth| auth.tokens.as_ref());

    OpenAICodexAuthSummary {
        auth_mode: auth.as_ref().and_then(|value| value.auth_mode.clone()),
        plan_type: tokens.and_then(|value| value.plan_type.clone()),
        account_id: tokens.and_then(|value| value.account_id.clone()),
        email: tokens.and_then(|value| value.email.clone()),
        last_refresh: auth.and_then(|value| value.last_refresh),
        auth_file_exists: path.exists(),
    }
}

fn decode_jwt_payload<T: for<'de> Deserialize<'de>>(jwt: &str) -> Result<T> {
    let mut parts = jwt.split('.');
    let (_header, payload, _signature) = match (parts.next(), parts.next(), parts.next()) {
        (Some(header), Some(payload), Some(signature))
            if !header.is_empty() && !payload.is_empty() && !signature.is_empty() =>
        {
            (header, payload, signature)
        }
        _ => return Err(anyhow!("Invalid JWT format")),
    };

    let payload_bytes = URL_SAFE_NO_PAD
        .decode(payload)
        .context("Failed to decode JWT payload")?;
    serde_json::from_slice::<T>(&payload_bytes).context("Failed to parse JWT payload")
}

fn parse_access_token_expiration(jwt: &str) -> Result<Option<DateTime<Utc>>> {
    let claims = decode_jwt_payload::<StandardJwtClaims>(jwt)?;
    Ok(claims
        .exp
        .and_then(|value| DateTime::<Utc>::from_timestamp(value, 0)))
}

fn parse_id_token(jwt: &str) -> Result<ParsedIdToken> {
    #[derive(Debug, Deserialize)]
    struct IdClaims {
        #[serde(default)]
        email: Option<String>,
        #[serde(rename = "https://api.openai.com/profile", default)]
        profile: Option<ProfileClaims>,
        #[serde(rename = "https://api.openai.com/auth", default)]
        auth: Option<AuthClaims>,
    }

    #[derive(Debug, Deserialize)]
    struct ProfileClaims {
        #[serde(default)]
        email: Option<String>,
    }

    #[derive(Debug, Deserialize)]
    struct AuthClaims {
        #[serde(default)]
        chatgpt_plan_type: Option<String>,
        #[serde(default)]
        chatgpt_user_id: Option<String>,
        #[serde(default)]
        user_id: Option<String>,
        #[serde(default)]
        chatgpt_account_id: Option<String>,
    }

    let claims = decode_jwt_payload::<IdClaims>(jwt)?;
    let email = claims
        .email
        .or_else(|| claims.profile.and_then(|profile| profile.email));
    let (plan_type, user_id, account_id) = match claims.auth {
        Some(auth) => (
            auth.chatgpt_plan_type,
            auth.chatgpt_user_id.or(auth.user_id),
            auth.chatgpt_account_id,
        ),
        None => (None, None, None),
    };

    Ok(ParsedIdToken {
        email,
        plan_type,
        user_id,
        account_id,
    })
}

fn auth_requires_refresh(auth: &OpenAICodexAuthFile) -> bool {
    let Some(tokens) = auth.tokens.as_ref() else {
        return false;
    };

    if let Ok(Some(expires_at)) = parse_access_token_expiration(&tokens.access_token) {
        if expires_at <= Utc::now() + chrono::Duration::minutes(ACCESS_TOKEN_REFRESH_WINDOW_MINUTES)
        {
            return true;
        }
    }

    auth.last_refresh
        .is_some_and(|last| last < Utc::now() - chrono::Duration::days(TOKEN_REFRESH_INTERVAL_DAYS))
}

async fn refresh_auth_tokens(auth: &OpenAICodexAuthFile) -> Result<OpenAICodexAuthFile> {
    let refresh_token = auth
        .tokens
        .as_ref()
        .map(|value| value.refresh_token.clone())
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| anyhow!("Missing Codex refresh token"))?;

    let response = get_http_client()
        .post(format!("{}/oauth/token", OPENAI_CODEX_DEFAULT_ISSUER))
        .json(&serde_json::json!({
            "client_id": OPENAI_CODEX_CLIENT_ID,
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
        }))
        .send()
        .await
        .context("Failed to refresh Codex auth token")?;

    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        error!(
            "Codex token refresh failed: status={}, body={}",
            status,
            summarize_error_body(&body)
        );
        return Err(anyhow!(
            "Codex token refresh failed with status {}: {}",
            status,
            body
        ));
    }

    let refresh = response
        .json::<RefreshTokenResponse>()
        .await
        .context("Failed to parse Codex refresh response")?;
    let mut updated = auth.clone();
    let existing = auth
        .tokens
        .as_ref()
        .ok_or_else(|| anyhow!("Missing Codex token state"))?;

    let id_token = refresh
        .id_token
        .unwrap_or_else(|| existing.id_token.clone());
    let access_token = refresh
        .access_token
        .unwrap_or_else(|| existing.access_token.clone());
    let refresh_token = refresh
        .refresh_token
        .unwrap_or_else(|| existing.refresh_token.clone());
    let parsed = parse_id_token(&id_token)?;

    updated.tokens = Some(OpenAICodexTokenData {
        id_token,
        access_token,
        refresh_token,
        account_id: parsed
            .account_id
            .clone()
            .or_else(|| existing.account_id.clone()),
        plan_type: parsed
            .plan_type
            .clone()
            .or_else(|| existing.plan_type.clone()),
        user_id: parsed.user_id.clone().or_else(|| existing.user_id.clone()),
        email: parsed.email.clone().or_else(|| existing.email.clone()),
    });
    updated.last_refresh = Some(Utc::now());
    save_auth_file(&updated)?;
    Ok(updated)
}

pub async fn force_refresh_auth_tokens() -> Result<OpenAICodexAuthFile> {
    let _guard = TOKEN_REFRESH_LOCK.lock().await;
    let auth = load_auth_file_internal()?.ok_or_else(|| anyhow!("Codex is not logged in"))?;
    info!("Forcing Codex auth token refresh");
    refresh_auth_tokens(&auth).await
}

pub async fn get_valid_auth_context() -> Result<OpenAICodexAuthContext> {
    let _guard = TOKEN_REFRESH_LOCK.lock().await;
    let mut auth = load_auth_file_internal()?.ok_or_else(|| anyhow!("Codex is not logged in"))?;
    if auth_requires_refresh(&auth) {
        info!("Refreshing Codex auth tokens before request");
        auth = refresh_auth_tokens(&auth).await?;
    }

    let tokens = auth
        .tokens
        .ok_or_else(|| anyhow!("Codex auth file does not contain tokens"))?;
    let account_id = tokens
        .account_id
        .clone()
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| anyhow!("Codex auth token does not include a ChatGPT account id"))?;

    Ok(OpenAICodexAuthContext {
        access_token: tokens.access_token,
        account_id,
    })
}

pub fn codex_headers(
    auth: &OpenAICodexAuthContext,
    session_id: Option<&str>,
) -> Vec<(String, String)> {
    let mut headers = vec![
        (
            "Authorization".to_string(),
            format!("Bearer {}", auth.access_token),
        ),
        ("ChatGPT-Account-Id".to_string(), auth.account_id.clone()),
        ("originator".to_string(), current_originator()),
        ("User-Agent".to_string(), current_user_agent()),
    ];

    if let Some(session_id) = session_id.filter(|value| !value.trim().is_empty()) {
        headers.push(("session_id".to_string(), session_id.to_string()));
    }

    headers
}

pub fn codex_base_url() -> String {
    CONFIG
        .openai_codex_base_url
        .trim_end_matches('/')
        .to_string()
}

pub fn codex_response_url() -> String {
    format!("{}/responses", codex_base_url())
}

fn codex_usage_url_from_base_url(base_url: &str) -> String {
    let normalized = base_url.trim().trim_end_matches('/');
    if let Some((prefix, _)) = normalized.split_once("/backend-api") {
        format!("{prefix}/backend-api/wham/usage")
    } else if normalized.ends_with("/api/codex") || normalized.ends_with("/codex") {
        format!("{normalized}/usage")
    } else {
        format!("{normalized}/api/codex/usage")
    }
}

pub fn codex_usage_url() -> String {
    codex_usage_url_from_base_url(&codex_base_url())
}

pub fn native_web_search_mode() -> CodexWebSearchMode {
    CodexWebSearchMode::from_config(&CONFIG.openai_codex_web_search_mode)
}

pub fn build_native_web_search_tool_from_record(
    supports_search_tool: bool,
    web_search_tool_type: CodexWebSearchToolType,
    mode: CodexWebSearchMode,
    allowed_domains: &[String],
    context_size: Option<&str>,
) -> Option<Value> {
    if !supports_search_tool || mode == CodexWebSearchMode::Disabled {
        return None;
    }

    let external_web_access = match mode {
        CodexWebSearchMode::Cached => false,
        CodexWebSearchMode::Live => true,
        CodexWebSearchMode::Disabled => return None,
    };

    let mut tool = json!({
        "type": "web_search",
        "external_web_access": external_web_access,
    });

    if !allowed_domains.is_empty() {
        tool["filters"] = json!({
            "allowed_domains": allowed_domains,
        });
    }

    if let Some(context_size) = context_size.filter(|value| !value.trim().is_empty()) {
        tool["search_context_size"] = Value::String(context_size.to_string());
    }

    if web_search_tool_type == CodexWebSearchToolType::TextAndImage {
        tool["search_content_types"] = json!(["text", "image"]);
    }

    Some(tool)
}

fn map_usage_window(window: Option<CodexUsageWindowSnapshot>) -> Option<CodexUsageWindow> {
    let window = window?;
    Some(CodexUsageWindow {
        used_percent: window.used_percent,
        limit_window_seconds: window.limit_window_seconds,
        reset_at: window.reset_at,
    })
}

pub async fn fetch_usage_snapshot() -> Result<CodexUsageSnapshot> {
    let url = codex_usage_url();
    for attempt in 0..2 {
        let auth = get_valid_auth_context().await?;
        info!("Fetching Codex usage snapshot");
        let response = get_http_client()
            .get(&url)
            .headers({
                let mut headers = reqwest::header::HeaderMap::new();
                for (name, value) in codex_headers(&auth, None) {
                    headers.insert(
                        reqwest::header::HeaderName::from_bytes(name.as_bytes())?,
                        reqwest::header::HeaderValue::from_str(&value)?,
                    );
                }
                headers
            })
            .send()
            .await
            .context("Failed to fetch Codex usage snapshot")?;

        if response.status() == reqwest::StatusCode::UNAUTHORIZED && attempt == 0 {
            warn!("Codex usage request unauthorized; refreshing auth and retrying");
            let _ = force_refresh_auth_tokens().await?;
            continue;
        }

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            error!(
                "Codex usage request failed: status={}, body={}",
                status,
                summarize_error_body(&body)
            );
            return Err(anyhow!(
                "Codex usage request failed with status {}: {}",
                status,
                body
            ));
        }

        let body = response
            .text()
            .await
            .context("Failed to read Codex usage response body")?;
        let payload = match serde_json::from_str::<CodexUsageResponse>(&body) {
            Ok(payload) => payload,
            Err(err) => {
                error!(
                    "Failed to parse Codex usage response: body={}",
                    summarize_error_body(&body)
                );
                return Err(anyhow!("Failed to parse Codex usage response: {}", err));
            }
        };

        info!(
            "Fetched Codex usage snapshot successfully: plan_type={}",
            payload.plan_type.as_deref().unwrap_or("unknown")
        );

        let primary = payload
            .rate_limit
            .as_ref()
            .and_then(|details| map_usage_window(details.primary_window.clone()));
        let secondary = payload
            .rate_limit
            .as_ref()
            .and_then(|details| map_usage_window(details.secondary_window.clone()));
        let credits = payload.credits.map(|credits| CodexUsageCredits {
            has_credits: credits.has_credits,
            unlimited: credits.unlimited,
            balance: credits.balance,
        });
        let additional_limits = payload
            .additional_rate_limits
            .unwrap_or_default()
            .into_iter()
            .map(|limit| CodexAdditionalUsageLimit {
                limit_name: limit.limit_name,
                metered_feature: limit.metered_feature,
                primary: limit
                    .rate_limit
                    .as_ref()
                    .and_then(|details| map_usage_window(details.primary_window.clone())),
                secondary: limit
                    .rate_limit
                    .as_ref()
                    .and_then(|details| map_usage_window(details.secondary_window.clone())),
            })
            .collect::<Vec<_>>();

        return Ok(CodexUsageSnapshot {
            plan_type: payload.plan_type,
            primary,
            secondary,
            credits,
            additional_limits,
        });
    }

    Err(anyhow!("Codex usage request failed after refresh retry"))
}

pub async fn fetch_models() -> Result<CodexModelList> {
    let version = if CONFIG.openai_codex_client_version.trim().is_empty() {
        "0.99.0"
    } else {
        CONFIG.openai_codex_client_version.trim()
    };
    for attempt in 0..2 {
        let auth = get_valid_auth_context().await?;
        info!("Fetching Codex model catalog with client_version={version}");
        let response = get_http_client()
            .get(format!("{}/models", codex_base_url()))
            .query(&[("client_version", version)])
            .headers({
                let mut headers = reqwest::header::HeaderMap::new();
                for (name, value) in codex_headers(&auth, None) {
                    headers.insert(
                        reqwest::header::HeaderName::from_bytes(name.as_bytes())?,
                        reqwest::header::HeaderValue::from_str(&value)?,
                    );
                }
                headers
            })
            .send()
            .await
            .context("Failed to fetch Codex models")?;

        if response.status() == reqwest::StatusCode::UNAUTHORIZED && attempt == 0 {
            warn!(
                "Codex models request unauthorized with client_version={version}; refreshing auth and retrying"
            );
            let _ = force_refresh_auth_tokens().await?;
            continue;
        }

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            error!(
                "Codex models request failed: status={}, client_version={}, body={}",
                status,
                version,
                summarize_error_body(&body)
            );
            return Err(anyhow!(
                "Codex models request failed with status {}: {}",
                status,
                body
            ));
        }

        let etag = response
            .headers()
            .get(reqwest::header::ETAG)
            .and_then(|value| value.to_str().ok())
            .map(ToString::to_string);
        let body = response
            .text()
            .await
            .context("Failed to read Codex models response body")?;
        let payload = match serde_json::from_str::<CodexModelsResponse>(&body) {
            Ok(payload) => payload,
            Err(err) => {
                error!(
                    "Failed to parse Codex models response: client_version={}, body={}",
                    version,
                    summarize_error_body(&body)
                );
                return Err(anyhow!("Failed to parse Codex models response: {}", err));
            }
        };
        info!(
            "Fetched Codex model catalog successfully: count={}, client_version={version}",
            payload.models.len()
        );

        return Ok(CodexModelList {
            models: payload.models,
            etag,
        });
    }

    Err(anyhow!("Codex models request failed after refresh retry"))
}

pub async fn request_device_code() -> Result<DeviceCodeStart> {
    let response = get_http_client()
        .post(format!(
            "{}/api/accounts/deviceauth/usercode",
            OPENAI_CODEX_DEFAULT_ISSUER
        ))
        .json(&DeviceCodeUserCodeRequest {
            client_id: OPENAI_CODEX_CLIENT_ID,
        })
        .send()
        .await
        .context("Failed to start Codex device-code login")?;

    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        return Err(anyhow!(
            "Codex device-code request failed with status {}: {}",
            status,
            body
        ));
    }

    let data = response
        .json::<DeviceCodeUserCodeResponse>()
        .await
        .context("Failed to parse Codex device-code response")?;

    Ok(DeviceCodeStart {
        verification_url: format!("{}/codex/device", OPENAI_CODEX_DEFAULT_ISSUER),
        user_code: data.user_code,
        device_auth_id: data.device_auth_id,
        interval_seconds: data.interval.max(1),
    })
}

async fn exchange_authorization_code(
    authorization_code: &str,
    redirect_uri: &str,
    code_verifier: &str,
) -> Result<OAuthTokenResponse> {
    let body = serde_urlencoded::to_string([
        ("grant_type", "authorization_code"),
        ("code", authorization_code),
        ("redirect_uri", redirect_uri),
        ("client_id", OPENAI_CODEX_CLIENT_ID),
        ("code_verifier", code_verifier),
    ])?;
    let response = get_http_client()
        .post(format!("{}/oauth/token", OPENAI_CODEX_DEFAULT_ISSUER))
        .header(
            reqwest::header::CONTENT_TYPE,
            "application/x-www-form-urlencoded",
        )
        .body(body)
        .send()
        .await
        .context("Failed to exchange Codex authorization code")?;

    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        return Err(anyhow!(
            "Codex OAuth token exchange failed with status {}: {}",
            status,
            body
        ));
    }

    response
        .json::<OAuthTokenResponse>()
        .await
        .context("Failed to parse Codex OAuth token exchange response")
}

pub async fn complete_device_code_login(
    device_code: &DeviceCodeStart,
    cancel: Arc<AtomicBool>,
) -> Result<OpenAICodexAuthFile> {
    let poll_deadline = Instant::now() + Duration::from_secs(15 * 60);

    loop {
        if cancel.load(Ordering::SeqCst) {
            return Err(anyhow!("Codex device-code login was cancelled"));
        }
        if Instant::now() >= poll_deadline {
            return Err(anyhow!(
                "Codex device-code login timed out after 15 minutes"
            ));
        }

        let response = get_http_client()
            .post(format!(
                "{}/api/accounts/deviceauth/token",
                OPENAI_CODEX_DEFAULT_ISSUER
            ))
            .json(&DeviceCodeTokenRequest {
                device_auth_id: &device_code.device_auth_id,
                user_code: &device_code.user_code,
            })
            .send()
            .await
            .context("Failed to poll Codex device-code token")?;

        if response.status().is_success() {
            let code = response
                .json::<DeviceCodeTokenResponse>()
                .await
                .context("Failed to parse Codex device-code token response")?;
            let oauth = exchange_authorization_code(
                &code.authorization_code,
                &format!("{}/deviceauth/callback", OPENAI_CODEX_DEFAULT_ISSUER),
                &code.code_verifier,
            )
            .await?;

            let parsed = parse_id_token(&oauth.id_token)?;
            let auth = OpenAICodexAuthFile {
                auth_mode: Some("chatgpt".to_string()),
                openai_api_key: None,
                tokens: Some(OpenAICodexTokenData {
                    id_token: oauth.id_token,
                    access_token: oauth.access_token,
                    refresh_token: oauth.refresh_token,
                    account_id: parsed.account_id,
                    plan_type: parsed.plan_type,
                    user_id: parsed.user_id,
                    email: parsed.email,
                }),
                last_refresh: Some(Utc::now()),
            };
            if cancel.load(Ordering::SeqCst) {
                return Err(anyhow!("Codex device-code login was cancelled"));
            }
            save_auth_file(&auth)?;
            info!("Codex device-code login completed successfully");
            return Ok(auth);
        }

        let status = response.status();
        if status == reqwest::StatusCode::FORBIDDEN || status == reqwest::StatusCode::NOT_FOUND {
            tokio::time::sleep(Duration::from_secs(device_code.interval_seconds)).await;
            continue;
        }

        let body = response.text().await.unwrap_or_default();
        return Err(anyhow!(
            "Codex device-code login failed with status {}: {}",
            status,
            body
        ));
    }
}

pub fn logout() -> Result<bool> {
    let path = auth_file_path();
    match fs::remove_file(path) {
        Ok(()) => Ok(true),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(err) => Err(err.into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_id_token_extracts_openai_claims() {
        let header = URL_SAFE_NO_PAD.encode(r#"{"alg":"none"}"#);
        let payload = URL_SAFE_NO_PAD.encode(
            r#"{"email":"user@example.com","https://api.openai.com/auth":{"chatgpt_plan_type":"plus","chatgpt_user_id":"user_123","chatgpt_account_id":"acct_456"}}"#,
        );
        let jwt = format!("{header}.{payload}.sig");

        let parsed = parse_id_token(&jwt).expect("JWT should parse");

        assert_eq!(parsed.email.as_deref(), Some("user@example.com"));
        assert_eq!(parsed.plan_type.as_deref(), Some("plus"));
        assert_eq!(parsed.user_id.as_deref(), Some("user_123"));
        assert_eq!(parsed.account_id.as_deref(), Some("acct_456"));
    }

    #[test]
    fn parse_access_token_expiration_reads_exp_claim() {
        let now = Utc::now().timestamp() + 3600;
        let header = URL_SAFE_NO_PAD.encode(r#"{"alg":"none"}"#);
        let payload = URL_SAFE_NO_PAD.encode(format!(r#"{{"exp":{now}}}"#));
        let jwt = format!("{header}.{payload}.sig");

        let expiration = parse_access_token_expiration(&jwt)
            .expect("expiration should parse")
            .expect("expiration should exist");

        assert_eq!(expiration.timestamp(), now);
    }

    #[test]
    fn native_web_search_tool_supports_live_image_search() {
        let tool = build_native_web_search_tool_from_record(
            true,
            CodexWebSearchToolType::TextAndImage,
            CodexWebSearchMode::Live,
            &["example.com".to_string()],
            Some("high"),
        )
        .expect("native search tool should be built");

        assert_eq!(tool["type"], "web_search");
        assert_eq!(tool["external_web_access"], true);
        assert_eq!(tool["search_context_size"], "high");
        assert_eq!(tool["filters"]["allowed_domains"][0], "example.com");
        assert_eq!(tool["search_content_types"][0], "text");
        assert_eq!(tool["search_content_types"][1], "image");
    }

    #[test]
    fn codex_usage_url_uses_wham_for_chatgpt_backend_api() {
        assert_eq!(
            codex_usage_url_from_base_url("https://chatgpt.com/backend-api/codex"),
            "https://chatgpt.com/backend-api/wham/usage"
        );
    }

    #[test]
    fn codex_usage_url_uses_codex_api_path_for_codex_api_bases() {
        assert_eq!(
            codex_usage_url_from_base_url("https://example.com/api/codex"),
            "https://example.com/api/codex/usage"
        );
    }

    #[test]
    fn codex_usage_response_accepts_null_additional_rate_limits() {
        let raw = r#"{
            "plan_type": "plus",
            "rate_limit": {
                "primary_window": {
                    "used_percent": 66,
                    "limit_window_seconds": 18000,
                    "reset_at": 1774771659
                },
                "secondary_window": {
                    "used_percent": 45,
                    "limit_window_seconds": 604800,
                    "reset_at": 1775181504
                }
            },
            "additional_rate_limits": null,
            "credits": {
                "has_credits": false,
                "unlimited": false,
                "balance": "0"
            }
        }"#;

        let parsed: CodexUsageResponse =
            serde_json::from_str(raw).expect("usage response should parse");

        assert_eq!(parsed.plan_type.as_deref(), Some("plus"));
        assert!(parsed.additional_rate_limits.is_none());
        assert_eq!(
            parsed
                .rate_limit
                .as_ref()
                .and_then(|details| details.primary_window.as_ref())
                .map(|window| window.used_percent),
            Some(66.0)
        );
    }
}
