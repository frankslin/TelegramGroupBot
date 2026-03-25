use std::collections::HashMap;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::Result;
use once_cell::sync::Lazy;
use serde::Deserialize;
use tracing::{info, warn};

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct ModelCapabilities {
    pub images: bool,
    pub video: bool,
    pub audio: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ThirdPartyProvider {
    OpenRouter,
    Nvidia,
}

impl ThirdPartyProvider {
    pub fn as_str(&self) -> &'static str {
        match self {
            ThirdPartyProvider::OpenRouter => "openrouter",
            ThirdPartyProvider::Nvidia => "nvidia",
        }
    }
}

impl std::str::FromStr for ThirdPartyProvider {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self> {
        match value.trim().to_lowercase().as_str() {
            "openrouter" => Ok(ThirdPartyProvider::OpenRouter),
            "nvidia" => Ok(ThirdPartyProvider::Nvidia),
            other => Err(anyhow::anyhow!(
                "Unsupported third-party model provider '{}'",
                other
            )),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
struct ThirdPartyModelsFile {
    models: Vec<ThirdPartyModelEntry>,
}

#[derive(Debug, Clone, Deserialize)]
struct ThirdPartyModelEntry {
    provider: ThirdPartyProvider,
    name: String,
    model: String,
    #[serde(default)]
    image: Option<bool>,
    #[serde(default)]
    video: Option<bool>,
    #[serde(default)]
    audio: Option<bool>,
    #[serde(default)]
    tools: Option<bool>,
}

#[derive(Debug, Clone)]
pub struct ThirdPartyModelConfig {
    pub id: String,
    pub provider: ThirdPartyProvider,
    pub name: String,
    pub model: String,
    pub image: bool,
    pub video: bool,
    pub audio: bool,
    pub tools: bool,
}

impl ThirdPartyModelConfig {
    #[allow(dead_code)]
    pub fn capabilities(&self) -> ModelCapabilities {
        ModelCapabilities {
            images: self.image,
            video: self.video,
            audio: self.audio,
        }
    }
}

pub fn qualify_third_party_model_id(provider: ThirdPartyProvider, model: &str) -> String {
    format!("{}:{}", provider.as_str(), model.trim())
}

pub fn parse_third_party_model_id(identifier: &str) -> Option<(ThirdPartyProvider, &str)> {
    let (provider, model) = identifier.trim().split_once(':')?;
    let provider = provider.parse().ok()?;
    let model = model.trim();
    if model.is_empty() {
        None
    } else {
        Some((provider, model))
    }
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct Config {
    pub bot_token: String,
    pub log_level: String,
    pub database_url: String,
    pub publish_bot_commands: bool,
    pub gemini_api_key: String,
    pub gemini_model: String,
    pub gemini_lite_model: String,
    pub gemini_pro_model: String,
    pub gemini_image_model: String,
    pub gemini_music_model: String,
    pub gemini_video_model: String,
    pub gemini_temperature: f32,
    pub gemini_top_k: i32,
    pub gemini_top_p: f32,
    pub gemini_max_output_tokens: i32,
    pub gemini_thinking_level: String,
    pub gemini_safety_settings: String,
    pub enable_openrouter: bool,
    pub openrouter_api_key: String,
    pub openrouter_base_url: String,
    pub openrouter_alpha_base_url: String,
    pub openrouter_temperature: f32,
    pub openrouter_top_k: i32,
    pub openrouter_top_p: f32,
    pub enable_nvidia: bool,
    pub nvidia_api_key: String,
    pub nvidia_base_url: String,
    pub nvidia_temperature: f32,
    pub nvidia_top_k: i32,
    pub nvidia_top_p: f32,
    pub enable_jina_mcp: bool,
    pub jina_ai_api_key: String,
    pub jina_search_endpoint: String,
    pub jina_reader_endpoint: String,
    pub enable_brave_search: bool,
    pub brave_search_api_key: String,
    pub brave_search_endpoint: String,
    pub enable_exa_search: bool,
    pub exa_api_key: String,
    pub exa_search_endpoint: String,
    pub web_search_cache_ttl_seconds: u64,
    pub web_search_providers: Vec<String>,
    pub llama_model: String,
    pub grok_model: String,
    pub qwen_model: String,
    pub deepseek_model: String,
    pub gpt_model: String,
    pub rate_limit_seconds: u64,
    pub model_selection_timeout: u64,
    pub default_q_model: String,
    pub telegram_max_length: usize,
    pub telegraph_access_token: String,
    pub telegraph_author_name: String,
    pub telegraph_author_url: String,
    pub user_history_message_count: i64,
    pub cwd_pw_api_key: String,
    pub support_message: String,
    pub support_link: String,
    pub whitelist_file_path: String,
    pub access_controlled_commands: Vec<String>,
    pub third_party_models_config_path: PathBuf,
    pub third_party_models: Vec<ThirdPartyModelConfig>,
    pub third_party_models_by_id: HashMap<String, ThirdPartyModelConfig>,
}

pub static CONFIG: Lazy<Config> =
    Lazy::new(|| Config::load().expect("Failed to load configuration"));

fn env_bool(name: &str, default: bool) -> bool {
    env::var(name)
        .ok()
        .map(|value| value.trim().eq_ignore_ascii_case("true"))
        .unwrap_or(default)
}

fn env_string(name: &str, default: &str) -> String {
    env::var(name).unwrap_or_else(|_| default.to_string())
}

fn env_f32(name: &str, default: f32) -> f32 {
    env::var(name)
        .ok()
        .and_then(|value| value.parse::<f32>().ok())
        .unwrap_or(default)
}

fn env_i32(name: &str, default: i32) -> i32 {
    env::var(name)
        .ok()
        .and_then(|value| value.parse::<i32>().ok())
        .unwrap_or(default)
}

fn env_u64(name: &str, default: u64) -> u64 {
    env::var(name)
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(default)
}

fn env_usize(name: &str, default: usize) -> usize {
    env::var(name)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(default)
}

fn env_csv_lowercase(name: &str, default: &str) -> Vec<String> {
    env::var(name)
        .unwrap_or_else(|_| default.to_string())
        .split(',')
        .map(|value| value.trim().to_lowercase())
        .filter(|value| !value.is_empty())
        .collect()
}

fn normalize_database_url(value: String) -> String {
    if value.starts_with("sqlite+aiosqlite://") {
        return value.replacen("sqlite+aiosqlite://", "sqlite://", 1);
    }
    value
}

fn normalize_gemini_safety_settings(value: String) -> String {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return "permissive".to_string();
    }

    let lowered = trimmed.to_lowercase();
    match lowered.as_str() {
        "permissive" | "off" | "none" => "permissive".to_string(),
        "standard" => "standard".to_string(),
        _ => {
            warn!(
                "Unknown GEMINI_SAFETY_SETTINGS value '{}'; defaulting to permissive.",
                value
            );
            "permissive".to_string()
        }
    }
}

fn resolve_third_party_models_path() -> PathBuf {
    let mut candidates: Vec<PathBuf> = Vec::new();
    if let Ok(env_value) = env::var("THIRD_PARTY_MODELS_CONFIG_PATH") {
        let env_path = PathBuf::from(env_value);
        if env_path.is_absolute() {
            candidates.push(env_path);
        } else {
            candidates.push(
                PathBuf::from(env::current_dir().unwrap_or_else(|_| PathBuf::from(".")))
                    .join(env_path),
            );
        }
    }
    candidates.push(PathBuf::from("third_party_models.json"));
    candidates.push(PathBuf::from("bot").join("third_party_models.json"));

    for candidate in &candidates {
        if candidate.exists() {
            return candidate.to_path_buf();
        }
    }

    candidates
        .get(0)
        .cloned()
        .unwrap_or_else(|| PathBuf::from("third_party_models.json"))
}

fn build_third_party_model_config(
    provider: ThirdPartyProvider,
    name: &str,
    model: &str,
    image: bool,
    video: bool,
    audio: bool,
    tools: bool,
) -> ThirdPartyModelConfig {
    ThirdPartyModelConfig {
        id: qualify_third_party_model_id(provider, model),
        provider,
        name: name.to_string(),
        model: model.to_string(),
        image,
        video,
        audio,
        tools,
    }
}

fn parse_third_party_models_from_str(raw: &str) -> Vec<ThirdPartyModelConfig> {
    let parsed: ThirdPartyModelsFile = match serde_json::from_str(raw) {
        Ok(data) => data,
        Err(err) => {
            info!("Failed to parse third-party model config JSON: {}", err);
            return Vec::new();
        }
    };

    let mut models = Vec::new();
    for entry in parsed.models {
        let name = entry.name.trim();
        let model = entry.model.trim();
        if name.is_empty() || model.is_empty() {
            continue;
        }
        models.push(build_third_party_model_config(
            entry.provider,
            name,
            model,
            entry.image.unwrap_or(false),
            entry.video.unwrap_or(false),
            entry.audio.unwrap_or(false),
            entry.tools.unwrap_or(true),
        ));
    }
    models
}

fn load_third_party_models_from_path(path: &Path) -> Vec<ThirdPartyModelConfig> {
    if !path.exists() {
        info!("Third-party model config not found at {}", path.display());
        return Vec::new();
    }

    let raw = match fs::read_to_string(path) {
        Ok(content) => content,
        Err(err) => {
            info!(
                "Failed to read third-party model config at {}: {}",
                path.display(),
                err
            );
            return Vec::new();
        }
    };

    let models = parse_third_party_models_from_str(&raw);
    if models.is_empty() && !raw.trim().is_empty() {
        info!("Parsed zero third-party models from {}", path.display());
    }
    models
}

fn load_legacy_openrouter_models(config: &LegacyOpenRouterEnv) -> Vec<ThirdPartyModelConfig> {
    let legacy_entries: Vec<(&str, &str, bool, bool, bool, bool)> = vec![
        ("Llama 4", &config.llama_model, true, false, false, true),
        ("Grok 4", &config.grok_model, true, false, false, true),
        ("Qwen 3", &config.qwen_model, false, false, false, true),
        (
            "DeepSeek 3.1",
            &config.deepseek_model,
            false,
            false,
            false,
            false,
        ),
        ("GPT", &config.gpt_model, true, false, false, true),
    ];

    let mut models = Vec::new();
    for (name, model_id, image, video, audio, tools) in legacy_entries {
        if !model_id.trim().is_empty() {
            models.push(build_third_party_model_config(
                ThirdPartyProvider::OpenRouter,
                name,
                model_id.trim(),
                image,
                video,
                audio,
                tools,
            ));
        }
    }
    models
}

fn build_third_party_models(
    path: &Path,
    legacy_env: &LegacyOpenRouterEnv,
) -> Vec<ThirdPartyModelConfig> {
    let models = load_third_party_models_from_path(path);
    if !models.is_empty() {
        info!(
            "Loaded {} third-party model(s) from {}",
            models.len(),
            path.display()
        );
        return models;
    }
    let legacy_models = load_legacy_openrouter_models(legacy_env);
    if !legacy_models.is_empty() {
        info!(
            "Using legacy OpenRouter model configuration with {} model(s)",
            legacy_models.len()
        );
    } else {
        info!(
            "No third-party models configured via JSON or legacy OpenRouter environment variables"
        );
    }
    legacy_models
}

fn resolve_exact_model_identifier(
    value: &str,
    models: &[ThirdPartyModelConfig],
    default_provider: Option<ThirdPartyProvider>,
) -> String {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return String::new();
    }

    if trimmed.eq_ignore_ascii_case("gemini") {
        return "gemini".to_string();
    }

    if let Some((provider, model)) = parse_third_party_model_id(trimmed) {
        return qualify_third_party_model_id(provider, model);
    }

    let exact_matches = models
        .iter()
        .filter(|config_entry| config_entry.model == trimmed)
        .collect::<Vec<_>>();
    if exact_matches.len() == 1 {
        return exact_matches[0].id.clone();
    }

    if let Some(provider) = default_provider {
        return qualify_third_party_model_id(provider, trimmed);
    }

    trimmed.to_string()
}

fn resolve_model_by_keyword(
    value: &str,
    models: &[ThirdPartyModelConfig],
    keywords: &[&str],
    default_provider: Option<ThirdPartyProvider>,
) -> String {
    if !value.trim().is_empty() {
        return resolve_exact_model_identifier(value, models, default_provider);
    }

    let lowered: Vec<String> = keywords.iter().map(|k| k.to_lowercase()).collect();
    for config_entry in models {
        let haystack = format!(
            "{} {} {}",
            config_entry.provider.as_str(),
            config_entry.name,
            config_entry.model
        )
        .to_lowercase();
        if lowered.iter().all(|keyword| haystack.contains(keyword)) {
            return config_entry.id.clone();
        }
    }

    resolve_exact_model_identifier(value, models, default_provider)
}

#[derive(Debug, Clone)]
struct LegacyOpenRouterEnv {
    llama_model: String,
    grok_model: String,
    qwen_model: String,
    deepseek_model: String,
    gpt_model: String,
}

impl Config {
    pub fn load() -> Result<Self> {
        let bot_token = env::var("BOT_TOKEN").unwrap_or_default();
        if bot_token.trim().is_empty() {
            return Err(anyhow::anyhow!("BOT_TOKEN is required"));
        }

        let legacy_env = LegacyOpenRouterEnv {
            llama_model: env_string("LLAMA_MODEL", ""),
            grok_model: env_string("GROK_MODEL", ""),
            qwen_model: env_string("QWEN_MODEL", ""),
            deepseek_model: env_string("DEEPSEEK_MODEL", ""),
            gpt_model: env_string("GPT_MODEL", ""),
        };

        let third_party_models_config_path = resolve_third_party_models_path();
        let third_party_models =
            build_third_party_models(&third_party_models_config_path, &legacy_env);
        let third_party_models_by_id = third_party_models
            .iter()
            .cloned()
            .map(|model| (model.id.clone(), model))
            .collect::<HashMap<_, _>>();

        let llama_model = resolve_model_by_keyword(
            &legacy_env.llama_model,
            &third_party_models,
            &["llama"],
            Some(ThirdPartyProvider::OpenRouter),
        );
        let grok_model = resolve_model_by_keyword(
            &legacy_env.grok_model,
            &third_party_models,
            &["grok"],
            Some(ThirdPartyProvider::OpenRouter),
        );
        let qwen_model = resolve_model_by_keyword(
            &legacy_env.qwen_model,
            &third_party_models,
            &["qwen"],
            Some(ThirdPartyProvider::OpenRouter),
        );
        let deepseek_model = resolve_model_by_keyword(
            &legacy_env.deepseek_model,
            &third_party_models,
            &["deepseek"],
            Some(ThirdPartyProvider::OpenRouter),
        );
        let gpt_model = resolve_model_by_keyword(
            &legacy_env.gpt_model,
            &third_party_models,
            &["gpt"],
            Some(ThirdPartyProvider::OpenRouter),
        );

        let access_controlled_commands = env::var("ACCESS_CONTROLLED_COMMANDS")
            .ok()
            .map(|value| {
                value
                    .split(',')
                    .map(|entry| entry.trim().to_string())
                    .filter(|entry| !entry.is_empty())
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();

        let mut web_search_providers = env_csv_lowercase("WEB_SEARCH_PROVIDERS", "brave,exa,jina");
        if web_search_providers.is_empty() {
            web_search_providers = vec!["brave".to_string(), "exa".to_string(), "jina".to_string()];
        }

        Ok(Config {
            bot_token,
            log_level: env_string("LOG_LEVEL", "info").to_lowercase(),
            database_url: normalize_database_url(env_string(
                "DATABASE_URL",
                "sqlite+aiosqlite:///bot.db",
            )),
            publish_bot_commands: env_bool("PUBLISH_BOT_COMMANDS", false),
            gemini_api_key: env_string("GEMINI_API_KEY", ""),
            gemini_model: env_string("GEMINI_MODEL", "gemini-flash-latest"),
            gemini_lite_model: env_string("GEMINI_LITE_MODEL", "gemini-flash-lite-latest"),
            gemini_pro_model: env_string("GEMINI_PRO_MODEL", "gemini-2.5-pro"),
            gemini_image_model: env_string("GEMINI_IMAGE_MODEL", "gemini-3-pro-image-preview"),
            gemini_music_model: env_string("GEMINI_MUSIC_MODEL", "lyria-3-pro-preview"),
            gemini_video_model: env_string("GEMINI_VIDEO_MODEL", "veo-3.1-generate-preview"),
            gemini_temperature: env_f32("GEMINI_TEMPERATURE", 0.7),
            gemini_top_k: env_i32("GEMINI_TOP_K", 40),
            gemini_top_p: env_f32("GEMINI_TOP_P", 0.95),
            gemini_max_output_tokens: env_i32("GEMINI_MAX_OUTPUT_TOKENS", 2048),
            gemini_thinking_level: env_string("GEMINI_THINKING_LEVEL", "high"),
            gemini_safety_settings: normalize_gemini_safety_settings(env_string(
                "GEMINI_SAFETY_SETTINGS",
                "permissive",
            )),
            enable_openrouter: env_bool("ENABLE_OPENROUTER", true),
            openrouter_api_key: env_string("OPENROUTER_API_KEY", ""),
            openrouter_base_url: env_string("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
            openrouter_alpha_base_url: env_string(
                "OPENROUTER_ALPHA_BASE_URL",
                "https://openrouter.ai/api/alpha",
            ),
            openrouter_temperature: env_f32("OPENROUTER_TEMPERATURE", 0.7),
            openrouter_top_k: env_i32("OPENROUTER_TOP_K", 40),
            openrouter_top_p: env_f32("OPENROUTER_TOP_P", 0.95),
            enable_nvidia: env_bool("ENABLE_NVIDIA", true),
            nvidia_api_key: env_string("NVIDIA_API_KEY", ""),
            nvidia_base_url: env_string("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"),
            nvidia_temperature: env_f32("NVIDIA_TEMPERATURE", 0.7),
            nvidia_top_k: env_i32("NVIDIA_TOP_K", 40),
            nvidia_top_p: env_f32("NVIDIA_TOP_P", 0.95),
            enable_jina_mcp: env_bool("ENABLE_JINA_MCP", false),
            jina_ai_api_key: env_string("JINA_AI_API_KEY", ""),
            jina_search_endpoint: env_string("JINA_SEARCH_ENDPOINT", "https://s.jina.ai/search"),
            jina_reader_endpoint: env_string("JINA_READER_ENDPOINT", "https://r.jina.ai/"),
            enable_brave_search: env_bool("ENABLE_BRAVE_SEARCH", true),
            brave_search_api_key: env_string("BRAVE_SEARCH_API_KEY", ""),
            brave_search_endpoint: env_string(
                "BRAVE_SEARCH_ENDPOINT",
                "https://api.search.brave.com/res/v1/web/search",
            ),
            enable_exa_search: env_bool("ENABLE_EXA_SEARCH", true),
            exa_api_key: env_string("EXA_API_KEY", ""),
            exa_search_endpoint: env_string("EXA_SEARCH_ENDPOINT", "https://api.exa.ai/search"),
            web_search_cache_ttl_seconds: env_u64("WEB_SEARCH_CACHE_TTL_SECONDS", 900),
            web_search_providers,
            llama_model,
            grok_model,
            qwen_model,
            deepseek_model,
            gpt_model,
            rate_limit_seconds: env_u64("RATE_LIMIT_SECONDS", 15),
            model_selection_timeout: env_u64("MODEL_SELECTION_TIMEOUT", 30),
            default_q_model: env_string("DEFAULT_Q_MODEL", "gemini").to_lowercase(),
            telegram_max_length: env_usize("TELEGRAM_MAX_LENGTH", 4000),
            telegraph_access_token: env_string("TELEGRAPH_ACCESS_TOKEN", ""),
            telegraph_author_name: env_string("TELEGRAPH_AUTHOR_NAME", ""),
            telegraph_author_url: env_string("TELEGRAPH_AUTHOR_URL", ""),
            user_history_message_count: env_u64("USER_HISTORY_MESSAGE_COUNT", 200) as i64,
            cwd_pw_api_key: env_string("CWD_PW_API_KEY", ""),
            support_message: env_string(
                "SUPPORT_MESSAGE",
                "Thanks for supporting the bot! Tap the button below to open the support page.",
            ),
            support_link: env_string("SUPPORT_LINK", ""),
            whitelist_file_path: env_string("WHITELIST_FILE_PATH", "allowed_chat.txt"),
            access_controlled_commands,
            third_party_models_config_path,
            third_party_models,
            third_party_models_by_id,
        })
    }

    pub fn iter_third_party_models(&self) -> &[ThirdPartyModelConfig] {
        &self.third_party_models
    }

    pub fn get_third_party_model_config(&self, model_id: &str) -> Option<&ThirdPartyModelConfig> {
        self.third_party_models_by_id.get(model_id)
    }

    pub fn is_third_party_provider_ready(&self, provider: ThirdPartyProvider) -> bool {
        match provider {
            ThirdPartyProvider::OpenRouter => {
                self.enable_openrouter && !self.openrouter_api_key.trim().is_empty()
            }
            ThirdPartyProvider::Nvidia => {
                self.enable_nvidia && !self.nvidia_api_key.trim().is_empty()
            }
        }
    }
}

pub const TLDR_SYSTEM_PROMPT: &str = r#"你是一个AI助手，名叫{bot_name}，请用中文总结以下群聊内容。
请先汇总出群聊主要内容。
再依据发言数量依次列出主要发言用户的名字和观点但不要超过10位用户。
请尽量详细地表述每个人的对各个议题的观点和陈述，字数不限。
非常关键：如果群聊内容中出现投资相关信息，请在总结后再全文最后逐项列出。格式为：投资标的物：投资建议 [由哪位用户提出]。
"#;

pub const FACTCHECK_SYSTEM_PROMPT: &str = "You are an expert fact-checker that is unbiased, honest, and direct. Your job is to evaluate the factual accuracy of the text provided.\n\nFor each significant claim, verify using web search results:\n1. Analyze each claim objectively.\n2. Provide a judgment on its accuracy (True, False, Partially True, or Insufficient Evidence).\n3. Briefly explain your reasoning with citations to the sources found through web search.\n4. When a claim is not factually accurate, provide corrections.\n5. IMPORTANT: The current UTC date and time is {current_datetime}. Verify all temporal claims relative to this date and time.\n6. CRITICAL: List the sources you used to check the facts with links.\n7. Format your response in an easily readable way using Markdown where appropriate.\n8. CRITICAL: You must decide the response language yourself.\n9. Language policy:\n- Prefer the language of the user's actual fact-check request or the primary claim/content being fact-checked.\n- Ignore structural wrappers and system-generated boilerplate when deciding the response language, including tags such as `<reply_context>`, `<factcheck_target>`, and `<auto_factcheck_target ... />`.\n- Ignore links, usernames, slash commands, inline code, emojis, and other noise when deciding the response language.\n- If the current user request and replied-to content are in different languages, prioritize the current user request unless the user clearly wants the reply in another language.\n- If there is no reliable text signal but the attached image, video, audio, or document has a clear language signal, use that.\n- If the language is still ambiguous, use this Telegram user language hint: {telegram_user_language_hint}.\n- If that hint is missing, unknown, or still does not provide a reliable answer, default to Chinese.\n- When the user explicitly asks for a specific response language, follow that instruction.\n\nAlways cite your sources and only draw definitive conclusions when you have sufficient reliable evidence.\n";

pub const Q_SYSTEM_PROMPT: &str = "You are a helpful assistant in a Telegram group chat. You provide concise, factual, and helpful answers to users' questions.\n\nGuidelines for your responses:\n1. Provide a direct, clear answer to the question.\n2. Be concise but comprehensive.\n3. Fact-check your information using web search and include citations to reliable sources.\n4. When the question asks for technical information, provide accurate and up-to-date information.\n5. IMPORTANT: Use web search to verify all facts and information before answering.\n6. CRITICAL: The current UTC date and time is {current_datetime}. Always verify current political leadership, office holders, and recent events through web search based on this date and time.\n7. If there's uncertainty, acknowledge it and explain the limitations.\n8. Format your response in an easily readable way using Markdown where appropriate.\n9. Keep your response under 400 words unless a detailed explanation is necessary.\n10. If the answer requires multiple parts, use numbered or bulleted lists.\n11. CRITICAL: You must decide the response language yourself.\n12. Language policy:\n- Prefer the language of the user's actual question or request.\n- If the message includes quoted text, reply context, links, usernames, slash commands, inline code, emojis, or other noise, ignore those when deciding the response language.\n- If the replied-to content is in a different language from the user's current question, prioritize the current question unless the user explicitly asks you to answer in another language.\n- If the user's message is too short or ambiguous to infer reliably, use this Telegram user language hint: {telegram_user_language_hint}.\n- If that hint is missing, unknown, or still does not provide a reliable answer, default to Chinese.\n- When there is a clear instruction to answer in a specific language, follow that instruction.\n\nRemember to be helpful and accurate in your responses. But do not be too nice and agreeable. If necessary, do not be afraid to be critical.\n";

pub const PROFILEME_SYSTEM_PROMPT: &str = "You are an experienced professional profiler. Based on the following chat history of a user in a group chat, generate a concise and insightful user profile. The profile must highlight their communication style, potential interests, key personality traits, and how they typically interact in the group. Focus on patterns and recurring themes. Address the user directly (e.g., 'You seem to be...').Do not include any specific message content, timestamps or message IDs.The user is asking for their own profile.CRITICAL: Always reply in Chinese";

pub const PAINTME_SYSTEM_PROMPT: &str = r#"You are a Visionary Prompt Engineer and Data Alchemist specializing in the "Nano Banana Pro" generation architecture.

YOUR GOAL:
Analyze the user's chat history and persona provided in the conversation. Distill their personality, communication style, and recurring themes into a single, cohesive *visual metaphor*. Then, convert this into an EXTREMELY DETAILED JSON object.

### STEP 1: CONCEPTUALIZATION & VARIANCE
1.  **Metaphorical Representation:** Do not depict the user physically. Focus on abstract concepts (e.g., "a geometric ice sculpture," "a clockwork garden").
2.  **Stochastic Art Style (CRITICAL):** To prevent visual repetition, you must RANDOMLY select a distinct art style (e.g., Baroque, Synthwave, Ukiyo-e, Bauhaus, Glitch Art) for every new request. Do *not* default to "Cinematic" or "Hyper-realistic" unless it strictly fits.
3.  **The "Twist":** You must inject one "Visual Twist"—an element that contrasts with the main theme (e.g., if the theme is "Ancient Ruins," add "Neon Cables").

### STEP 2: JSON STRUCTURE GUIDELINES
You must output a single valid JSON object.

1.  **Dynamic Taxonomy:** Invent keys that match your metaphor (e.g., if "Ocean," use `waves`, `depth`, `bioluminescence`).
2.  **Visual Twist:** Include a specific field called `visual_twist` describing the contrasting element.
3.  **Technical Specs:** You must define `lighting`, `color_palette`, and `medium` (e.g., "oil on canvas," "3D render").
4.  **Standard Fields:** Include `subject_summary`, `art_style`, `constraints`, and `negative_prompt`.

### ONE-SHOT EXAMPLE:
{
  "subject_summary": "A fragile glass heart suspended in a storm of iron filings",
  "art_style": "Surrealist macro photography mixed with charcoal sketching",
  "visual_twist": "The iron filings are magnetic and forming digital circuit patterns",
  "subject_details": {
    "core": "Translucent blown glass, cracking slightly under pressure",
    "particles": "Jagged, matte black iron dust swirling violently",
    "suspension": "Levitating in a zero-gravity void"
  },
  "technical_specs": {
    "lighting": "Single harsh strobe light from above, deep shadows",
    "color_palette": "Monochrome black and white with a single strike of crimson",
    "medium": "Photorealistic 8K render with film grain"
  },
  "constraints": {
    "must_keep": ["cracks in glass", "magnetic patterns"],
    "avoid": ["blood", "romantic imagery", "soft lighting"]
  },
  "negative_prompt": "cartoon, low res, blurry, happy, text, watermark"
}

### OUTPUT
Return ONLY the raw JSON string."#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_third_party_models_supports_mixed_providers() {
        let raw = r#"{
            "models": [
                {
                    "provider": "openrouter",
                    "name": "Qwen 3",
                    "model": "qwen/qwen3-next-80b-a3b-instruct:free",
                    "tools": true
                },
                {
                    "provider": "nvidia",
                    "name": "Gemma 3n",
                    "model": "google/gemma-3n-e4b-it",
                    "image": true,
                    "audio": true,
                    "tools": false
                }
            ]
        }"#;

        let models = parse_third_party_models_from_str(raw);

        assert_eq!(models.len(), 2);
        assert_eq!(
            models[0].id,
            "openrouter:qwen/qwen3-next-80b-a3b-instruct:free"
        );
        assert_eq!(models[0].provider, ThirdPartyProvider::OpenRouter);
        assert_eq!(models[1].id, "nvidia:google/gemma-3n-e4b-it");
        assert_eq!(models[1].provider, ThirdPartyProvider::Nvidia);
        assert!(models[1].image);
        assert!(models[1].audio);
        assert!(!models[1].tools);
    }

    #[test]
    fn provider_qualified_ids_disambiguate_duplicate_raw_model_ids() {
        let raw = r#"{
            "models": [
                {
                    "provider": "openrouter",
                    "name": "Shared OpenRouter",
                    "model": "shared/model"
                },
                {
                    "provider": "nvidia",
                    "name": "Shared NVIDIA",
                    "model": "shared/model"
                }
            ]
        }"#;

        let models = parse_third_party_models_from_str(raw);
        let model_map = models
            .iter()
            .cloned()
            .map(|model| (model.id.clone(), model))
            .collect::<HashMap<_, _>>();

        assert_eq!(models.len(), 2);
        assert!(model_map.contains_key("openrouter:shared/model"));
        assert!(model_map.contains_key("nvidia:shared/model"));
        assert_eq!(
            resolve_exact_model_identifier("shared/model", &models, None),
            "shared/model"
        );
    }

    #[test]
    fn resolve_model_by_keyword_returns_provider_qualified_id() {
        let models = vec![
            build_third_party_model_config(
                ThirdPartyProvider::OpenRouter,
                "Llama 4",
                "meta-llama/llama-4",
                true,
                false,
                false,
                true,
            ),
            build_third_party_model_config(
                ThirdPartyProvider::Nvidia,
                "Nemotron Super 49B",
                "nvidia/llama-3.3-nemotron-super-49b-v1.5",
                false,
                false,
                false,
                true,
            ),
        ];

        let resolved = resolve_model_by_keyword(
            "",
            &models,
            &["nemotron"],
            Some(ThirdPartyProvider::OpenRouter),
        );
        assert_eq!(resolved, "nvidia:nvidia/llama-3.3-nemotron-super-49b-v1.5");

        let legacy_exact = resolve_model_by_keyword(
            "meta-llama/llama-4",
            &models,
            &["llama"],
            Some(ThirdPartyProvider::OpenRouter),
        );
        assert_eq!(legacy_exact, "openrouter:meta-llama/llama-4");
    }

    #[test]
    fn legacy_openrouter_models_are_used_when_json_is_missing() {
        let path = std::env::temp_dir().join(format!(
            "missing-third-party-models-{}-{}.json",
            std::process::id(),
            chrono::Utc::now().timestamp_nanos_opt().unwrap_or_default()
        ));
        let legacy_env = LegacyOpenRouterEnv {
            llama_model: "meta-llama/llama-4".to_string(),
            grok_model: String::new(),
            qwen_model: String::new(),
            deepseek_model: String::new(),
            gpt_model: String::new(),
        };

        let models = build_third_party_models(&path, &legacy_env);

        assert_eq!(models.len(), 1);
        assert_eq!(models[0].provider, ThirdPartyProvider::OpenRouter);
        assert_eq!(models[0].id, "openrouter:meta-llama/llama-4");
    }
}

pub const PORTRAIT_SYSTEM_PROMPT: &str = r#"You are a Master Character Designer and Cinematic Portrait Photographer specializing in "Nano Banana Pro" prompts.

YOUR GOAL:
Analyze the user's chat history to construct a hyper-detailed "environmental portrait." Since you do not have a photo, you must INFER a plausible physical persona and style.

### STEP 1: PROFILING & RANDOMIZATION
1.  **The Persona:** Infer demographics and "vibe" from the text (vocabulary, interests, profession).
2.  **Randomized Composition (CRITICAL):** To avoid repetitive "passport style" photos, you must RANDOMLY select a camera angle and framing for each request.
    * *Options:* Low angle (hero shot), High angle (vulnerable), Profile, Reflection in a mirror, Wide shot (environment focus), Extreme close-up.
3.  **Lighting RNG:** Randomly select a lighting scenario that is NOT standard studio lighting (e.g., "Streetlights through blinds," "Bioluminescent glow," "Candlelight only").

### STEP 2: JSON STRUCTURE GUIDELINES
You must output a single valid JSON object.

1.  **Subject Specificity:** Use keys for `physical_appearance`, `attire`, and `expression`.
2.  **Composition Data:** You must include a `composition` object defining the angle and framing chosen in Step 1.
3.  **Environment:** Details on `setting`, `lighting`, and `props`.
4.  **Standard Fields:** Include `subject_summary`, `art_style`, `constraints`, and `negative_prompt`.

### ONE-SHOT EXAMPLE:
{
  "subject_summary": "A weary cyber-security analyst reflected in a rainy window",
  "art_style": "Neo-noir cinematic still, Blade Runner aesthetic",
  "physical_appearance": {
    "demographics": "Male, early 50s, greying beard",
    "expression": "Distant, contemplating the city outside",
    "wear": "Dark circles under eyes, slight stubble"
  },
  "attire": {
    "clothing": "Worn leather bomber jacket over a hoodie",
    "accessories": "Augmented reality contact lenses (glowing faint blue)"
  },
  "composition": {
    "angle": "Shot through glass looking in (reflection + subject)",
    "framing": "Medium shot, rule of thirds",
    "focus": "Raindrops on glass in focus, subject slightly soft"
  },
  "environment": {
    "setting": "Cramped server room in Tokyo",
    "lighting": "Neon pink and blue signage bleeding in from outside",
    "props": "Empty ramen bowl, tangles of ethernet cables"
  },
  "technical_specs": {
    "camera": "Leica M10, 35mm Summilux",
    "film_stock": "Kodak Vision3 500T (high grain)"
  },
  "constraints": {
    "must_keep": ["reflection", "neon colors", "rain texture"],
    "avoid": ["looking at camera", "clean environment", "daylight"]
  },
  "negative_prompt": "sunny, happy, clean, 3d render, plastic, smooth skin"
}

### OUTPUT
Return ONLY the raw JSON string."#;
