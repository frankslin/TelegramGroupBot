use std::collections::HashMap;
use std::fs;
use std::path::Path;

use anyhow::Result;
use chrono::{DateTime, Utc};
use once_cell::sync::Lazy;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use tracing::{info, warn};

use crate::config::{
    qualify_third_party_model_id, ThirdPartyModelConfig, ThirdPartyProvider, CONFIG,
};
use crate::llm::openai_codex::CodexReasoningEffortOption;

pub const OPENAI_CODEX_SELECTED_MODEL_ID: &str = "openai-codex:selected";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CodexSelectedModelRecord {
    pub slug: String,
    pub display_name: String,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub input_modalities: Vec<String>,
    #[serde(default)]
    pub priority: i32,
    #[serde(default)]
    pub etag: Option<String>,
    #[serde(default)]
    pub default_reasoning_level: Option<String>,
    #[serde(default)]
    pub supported_reasoning_levels: Vec<CodexReasoningEffortOption>,
    #[serde(default)]
    pub selected_reasoning_level: Option<String>,
    pub fetched_at: DateTime<Utc>,
}

#[derive(Debug, Clone)]
struct RuntimeModelsState {
    models: Vec<ThirdPartyModelConfig>,
    models_by_id: HashMap<String, ThirdPartyModelConfig>,
    codex_selected_model: Option<CodexSelectedModelRecord>,
}

static RUNTIME_MODELS: Lazy<RwLock<RuntimeModelsState>> =
    Lazy::new(|| RwLock::new(build_runtime_models_state()));

fn selected_model_path() -> &'static Path {
    Path::new(&CONFIG.openai_codex_model_path)
}

fn load_selected_codex_model_record() -> Option<CodexSelectedModelRecord> {
    let path = selected_model_path();
    let raw = match fs::read_to_string(path) {
        Ok(raw) => raw,
        Err(err) => {
            if err.kind() != std::io::ErrorKind::NotFound {
                warn!(
                    "Failed to read Codex selected model file {}: {}",
                    path.display(),
                    err
                );
            }
            return None;
        }
    };

    match serde_json::from_str::<CodexSelectedModelRecord>(&raw) {
        Ok(record) => Some(record),
        Err(err) => {
            warn!(
                "Failed to parse Codex selected model file {}: {}",
                path.display(),
                err
            );
            None
        }
    }
}

fn dynamic_codex_model_config(record: &CodexSelectedModelRecord) -> ThirdPartyModelConfig {
    let supports_images = record
        .input_modalities
        .iter()
        .any(|value| value.eq_ignore_ascii_case("image"));

    ThirdPartyModelConfig {
        id: OPENAI_CODEX_SELECTED_MODEL_ID.to_string(),
        provider: ThirdPartyProvider::OpenAICodex,
        name: record.display_name.clone(),
        model: record.slug.clone(),
        image: supports_images,
        video: false,
        audio: false,
        tools: true,
    }
}

fn build_runtime_models_state() -> RuntimeModelsState {
    let mut models = CONFIG.third_party_models.clone();
    let codex_selected_model = load_selected_codex_model_record();
    if let Some(record) = codex_selected_model.as_ref() {
        models.push(dynamic_codex_model_config(record));
    }
    let models_by_id = models
        .iter()
        .cloned()
        .map(|model| (model.id.clone(), model))
        .collect::<HashMap<_, _>>();

    RuntimeModelsState {
        models,
        models_by_id,
        codex_selected_model,
    }
}

pub fn reload_runtime_models() {
    let mut state = RUNTIME_MODELS.write();
    *state = build_runtime_models_state();
}

pub fn runtime_models() -> Vec<ThirdPartyModelConfig> {
    RUNTIME_MODELS.read().models.clone()
}

pub fn runtime_model_count() -> usize {
    RUNTIME_MODELS.read().models.len()
}

pub fn runtime_model_config(model_id: &str) -> Option<ThirdPartyModelConfig> {
    if model_id.trim().eq_ignore_ascii_case("openai-codex") {
        return RUNTIME_MODELS
            .read()
            .models_by_id
            .get(OPENAI_CODEX_SELECTED_MODEL_ID)
            .cloned();
    }

    let state = RUNTIME_MODELS.read();
    if let Some(model) = state.models_by_id.get(model_id).cloned() {
        return Some(model);
    }

    if let Some((provider, slug)) = crate::config::parse_third_party_model_id(model_id) {
        if provider == ThirdPartyProvider::OpenAICodex {
            if let Some(record) = state.codex_selected_model.as_ref() {
                if record.slug == slug {
                    return state
                        .models_by_id
                        .get(OPENAI_CODEX_SELECTED_MODEL_ID)
                        .cloned();
                }
            }
        }
    }

    None
}

pub fn resolve_runtime_model_identifier(identifier: &str) -> Option<String> {
    let trimmed = identifier.trim();
    if trimmed.eq_ignore_ascii_case("openai-codex") {
        return runtime_model_config(OPENAI_CODEX_SELECTED_MODEL_ID).map(|_| {
            qualify_third_party_model_id(ThirdPartyProvider::OpenAICodex, "selected")
        });
    }

    if let Some((provider, slug)) = crate::config::parse_third_party_model_id(trimmed) {
        if provider == ThirdPartyProvider::OpenAICodex {
            if let Some(record) = selected_codex_model_record() {
                if record.slug == slug {
                    return Some(OPENAI_CODEX_SELECTED_MODEL_ID.to_string());
                }
            }
        }
    }

    None
}

pub fn selected_codex_model_record() -> Option<CodexSelectedModelRecord> {
    RUNTIME_MODELS.read().codex_selected_model.clone()
}

pub fn save_selected_codex_model(record: &CodexSelectedModelRecord) -> Result<ThirdPartyModelConfig> {
    let path = selected_model_path();
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, serde_json::to_string_pretty(record)?)?;
    info!(
        "Saved selected Codex model {} ({}) to {}",
        record.display_name,
        record.slug,
        path.display()
    );
    reload_runtime_models();
    runtime_model_config(OPENAI_CODEX_SELECTED_MODEL_ID)
        .ok_or_else(|| anyhow::anyhow!("Selected Codex model did not load after save"))
}

pub fn save_selected_codex_reasoning_level(level: Option<String>) -> Result<CodexSelectedModelRecord> {
    let mut record = selected_codex_model_record()
        .ok_or_else(|| anyhow::anyhow!("No Codex model is currently selected"))?;
    record.selected_reasoning_level = level;
    save_selected_codex_model(&record)?;
    Ok(record)
}

pub fn current_codex_reasoning_level() -> Option<String> {
    selected_codex_model_record().and_then(|record| record.selected_reasoning_level)
}

pub fn clear_selected_codex_model() -> Result<bool> {
    let path = selected_model_path();
    let removed = match fs::remove_file(path) {
        Ok(()) => true,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => false,
        Err(err) => return Err(err.into()),
    };
    reload_runtime_models();
    Ok(removed)
}

pub fn is_runtime_provider_ready(provider: ThirdPartyProvider) -> bool {
    match provider {
        ThirdPartyProvider::OpenRouter => {
            CONFIG.enable_openrouter && !CONFIG.openrouter_api_key.trim().is_empty()
        }
        ThirdPartyProvider::Nvidia => {
            CONFIG.enable_nvidia && !CONFIG.nvidia_api_key.trim().is_empty()
        }
        ThirdPartyProvider::OpenAI => {
            CONFIG.enable_openai && !CONFIG.openai_api_key.trim().is_empty()
        }
        ThirdPartyProvider::OpenAICodex => {
            CONFIG.enable_openai_codex
                && crate::llm::openai_codex::is_auth_ready()
                && selected_codex_model_record().is_some()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selected_model_config_maps_image_capability_from_modalities() {
        let record = CodexSelectedModelRecord {
            slug: "gpt-5.4".to_string(),
            display_name: "GPT-5.4".to_string(),
            description: None,
            input_modalities: vec!["text".to_string(), "image".to_string()],
            priority: 1,
            etag: None,
            default_reasoning_level: Some("medium".to_string()),
            supported_reasoning_levels: vec![CodexReasoningEffortOption {
                effort: "medium".to_string(),
                description: "medium".to_string(),
            }],
            selected_reasoning_level: None,
            fetched_at: Utc::now(),
        };

        let config = dynamic_codex_model_config(&record);

        assert_eq!(config.id, OPENAI_CODEX_SELECTED_MODEL_ID);
        assert_eq!(config.provider, ThirdPartyProvider::OpenAICodex);
        assert!(config.image);
        assert!(!config.audio);
        assert!(!config.video);
        assert!(config.tools);
    }
}
