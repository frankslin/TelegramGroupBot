use std::collections::HashMap;
use std::time::{Duration, Instant};

use anyhow::{anyhow, Result};
use once_cell::sync::Lazy;
use tokio::sync::Mutex;
use tracing::{info, warn};

use crate::config::CONFIG;
use crate::llm::brave_search::brave_search;
use crate::llm::exa_search::exa_search;
use crate::llm::jina_search::search_jina_web;

const DEFAULT_MAX_RESULTS: usize = 5;
const MAX_RESULTS_LIMIT: usize = 10;
const SNIPPET_LIMIT: usize = 240;

#[derive(Debug, Clone)]
pub struct SearchResult {
    pub title: String,
    pub url: String,
    pub snippet: String,
}

#[derive(Debug, Clone)]
struct CacheEntry {
    stored_at: Instant,
    results: Vec<SearchResult>,
}

static SEARCH_CACHE: Lazy<Mutex<HashMap<String, CacheEntry>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WebSearchProvider {
    Brave,
    Exa,
    Jina,
}

impl WebSearchProvider {
    fn from_str(value: &str) -> Option<Self> {
        match value.trim().to_lowercase().as_str() {
            "brave" => Some(Self::Brave),
            "exa" => Some(Self::Exa),
            "jina" => Some(Self::Jina),
            _ => None,
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Brave => "brave",
            Self::Exa => "exa",
            Self::Jina => "jina",
        }
    }
}

fn normalize_snippet(value: &str) -> String {
    let snippet = value.replace('\n', " ");
    let snippet = snippet.trim();
    if snippet.chars().count() > SNIPPET_LIMIT {
        let truncated: String = snippet.chars().take(SNIPPET_LIMIT).collect();
        format!("{truncated}...")
    } else {
        snippet.to_string()
    }
}

fn normalize_result(mut result: SearchResult) -> Option<SearchResult> {
    if result.url.trim().is_empty() {
        return None;
    }

    if result.title.trim().is_empty() {
        result.title = result.url.clone();
    }

    result.snippet = normalize_snippet(&result.snippet);
    Some(result)
}

fn cache_key(query: &str, max_results: usize) -> String {
    format!(
        "{}::{}",
        query.trim().to_lowercase().replace('\n', " "),
        max_results
    )
}

fn cache_ttl() -> Duration {
    Duration::from_secs(CONFIG.web_search_cache_ttl_seconds)
}

async fn get_cached(query: &str, max_results: usize) -> Option<Vec<SearchResult>> {
    let ttl = cache_ttl();
    if ttl.is_zero() {
        return None;
    }

    let key = cache_key(query, max_results);
    let mut cache = SEARCH_CACHE.lock().await;
    if let Some(entry) = cache.get(&key) {
        if entry.stored_at.elapsed() < ttl {
            return Some(entry.results.clone());
        }
    }
    cache.remove(&key);
    None
}

async fn set_cached(query: &str, max_results: usize, results: Vec<SearchResult>) {
    if cache_ttl().is_zero() {
        return;
    }

    let key = cache_key(query, max_results);
    let entry = CacheEntry {
        stored_at: Instant::now(),
        results,
    };
    let mut cache = SEARCH_CACHE.lock().await;
    cache.insert(key, entry);
}

fn provider_order() -> Vec<WebSearchProvider> {
    let mut providers = Vec::new();
    for entry in &CONFIG.web_search_providers {
        if let Some(provider) = WebSearchProvider::from_str(entry) {
            providers.push(provider);
        } else {
            warn!(
                "Unknown web search provider '{}' in WEB_SEARCH_PROVIDERS",
                entry
            );
        }
    }

    if providers.is_empty() {
        providers = vec![
            WebSearchProvider::Brave,
            WebSearchProvider::Exa,
            WebSearchProvider::Jina,
        ];
    }

    providers
}

fn provider_enabled(provider: WebSearchProvider) -> bool {
    match provider {
        WebSearchProvider::Brave => {
            CONFIG.enable_brave_search && !CONFIG.brave_search_api_key.trim().is_empty()
        }
        WebSearchProvider::Exa => CONFIG.enable_exa_search && !CONFIG.exa_api_key.trim().is_empty(),
        WebSearchProvider::Jina => CONFIG.enable_jina_mcp,
    }
}

pub fn is_search_enabled() -> bool {
    provider_order().into_iter().any(provider_enabled)
}

async fn search_with_provider(
    provider: WebSearchProvider,
    query: &str,
    max_results: usize,
) -> Result<Vec<SearchResult>> {
    match provider {
        WebSearchProvider::Brave => {
            let results = brave_search(query, max_results).await?;
            Ok(results
                .into_iter()
                .filter_map(|item| {
                    normalize_result(SearchResult {
                        title: item.title,
                        url: item.url,
                        snippet: item.snippet,
                    })
                })
                .collect())
        }
        WebSearchProvider::Exa => {
            let results = exa_search(query, Some(max_results)).await?;
            Ok(results
                .into_iter()
                .filter_map(|(title, url, snippet)| {
                    normalize_result(SearchResult {
                        title,
                        url,
                        snippet,
                    })
                })
                .collect())
        }
        WebSearchProvider::Jina => {
            let response = search_jina_web(query, max_results).await?;
            Ok(response
                .results
                .into_iter()
                .filter_map(|item| {
                    normalize_result(SearchResult {
                        title: item.title,
                        url: item.url,
                        snippet: item.snippet,
                    })
                })
                .collect())
        }
    }
}

pub async fn search_web(query: &str, max_results: Option<usize>) -> Result<Vec<SearchResult>> {
    if query.trim().is_empty() {
        return Err(anyhow!("query must not be empty"));
    }

    let max_results = max_results
        .unwrap_or(DEFAULT_MAX_RESULTS)
        .clamp(1, MAX_RESULTS_LIMIT);

    if let Some(results) = get_cached(query, max_results).await {
        return Ok(results);
    }

    let providers = provider_order();
    if !providers.iter().any(|provider| provider_enabled(*provider)) {
        return Err(anyhow!("No web search providers are enabled"));
    }

    let mut last_error: Option<String> = None;
    let mut had_success = false;
    for provider in providers {
        if !provider_enabled(provider) {
            continue;
        }
        info!("Trying web search provider '{}'", provider.as_str());
        match search_with_provider(provider, query, max_results).await {
            Ok(results) => {
                had_success = true;
                let mut trimmed = results;
                if trimmed.len() > max_results {
                    trimmed.truncate(max_results);
                }
                if !trimmed.is_empty() {
                    set_cached(query, max_results, trimmed.clone()).await;
                    return Ok(trimmed);
                }
            }
            Err(err) => {
                last_error = Some(format!("{}: {}", provider.as_str(), err));
            }
        }
    }

    let empty = Vec::new();
    if had_success {
        if let Some(message) = last_error {
            warn!("Web search had partial failures: {}", message);
        }
        set_cached(query, max_results, empty.clone()).await;
        return Ok(empty);
    }

    if let Some(message) = last_error {
        warn!("Web search failed: {}", message);
        return Err(anyhow!("Web search failed: {}", message));
    }

    set_cached(query, max_results, empty.clone()).await;
    Ok(empty)
}

pub fn format_results_markdown(query: &str, results: &[SearchResult]) -> String {
    if results.is_empty() {
        return format!("No web results found for query: {}", query);
    }

    let mut lines = vec![format!("Search results for **{}**:", query)];
    for (idx, result) in results.iter().enumerate() {
        lines.push(format!("{}. [{}]({})", idx + 1, result.title, result.url));
        if !result.snippet.is_empty() {
            lines.push(format!("   {}", result.snippet));
        }
    }
    lines.join("\n")
}

pub async fn web_search_tool(query: &str, max_results: Option<usize>) -> Result<String> {
    let results = search_web(query, max_results).await?;
    Ok(format_results_markdown(query, &results))
}
