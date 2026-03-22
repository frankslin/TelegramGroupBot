use std::collections::BTreeSet;

use jieba_rs::Jieba;
use once_cell::sync::Lazy;
use regex::Regex;
use serde::{Deserialize, Serialize};
use url::Url;

pub const CURRENT_SEARCH_SCHEMA_VERSION: i64 = 1;
pub const SEARCH_INDEX_REBUILDING_ERROR: &str = "search_index_rebuilding";
const MAX_SEARCH_TEXT_CHARS: usize = 4_000;
const MAX_SNIPPET_SOURCE_CHARS: usize = 2_000;

static JIEBA: Lazy<Jieba> = Lazy::new(Jieba::new);
static URL_REGEX: Lazy<Regex> =
    Lazy::new(|| Regex::new(r#"https?://[^\s<>"'()\[\]]+"#).expect("valid url regex"));
static HTML_TAG_REGEX: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?is)<[^>]+>").expect("valid html regex"));
static ZERO_WIDTH_REGEX: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"[\u{200B}\u{200C}\u{200D}\u{2060}\u{FEFF}]").expect("valid zero width regex")
});
static MODEL_LINE_REGEX: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?im)(?:^|\s)Model:\s*[^\n]+$").expect("valid model regex"));
static TELEGRAPH_WRAPPER_REGEX: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?i)\[(?:Telegraph|Twitter) content extracted from [^\]]+\]")
        .expect("valid telegraph wrapper regex")
});
static VIEW_IT_HERE_REGEX: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?i)I have too much to say\.?\s*View it here").expect("valid telegraph view regex")
});
static ASK_PREFIX_REGEX: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?i)^\s*Ask(?: about chat)? [^:]{0,100}:\s*").expect("valid ask prefix regex")
});
static REPLY_CONTEXT_LABEL_REGEX: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?i)\bContext from replied message\s*:").expect("valid reply label regex")
});
static QUESTION_LABEL_REGEX: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?i)\bQuestion\s*:").expect("valid question label regex"));
static COMMAND_REGEX: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r#"(?i)^\s*/([a-z0-9_]+)(?:@\w+)?(?:\s|$)"#).expect("valid command regex")
});
static WHITESPACE_REGEX: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"\s+").expect("valid whitespace regex"));
static MARKDOWN_FORMATTING_REGEX: Lazy<Regex> =
    Lazy::new(|| Regex::new(r#"[*_`~]+"#).expect("valid markdown regex"));
static NON_TOKEN_EDGE_REGEX: Lazy<Regex> =
    Lazy::new(|| Regex::new(r#"^[^[:alnum:]_#@]+|[^[:alnum:]_#@]+$"#).expect("valid edge regex"));
static HAN_REGEX: Lazy<Regex> = Lazy::new(|| Regex::new(r"\p{Han}").expect("valid Han regex"));
static QUERY_TOKEN_REGEX: Lazy<Regex> =
    Lazy::new(|| Regex::new(r#"[\p{L}\p{N}_#@.]+"#).expect("valid query token regex"));

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum SearchMatchStage {
    Phrase,
    And,
    OrPrefix,
}

impl SearchMatchStage {
    pub fn label(self) -> &'static str {
        match self {
            Self::Phrase => "phrase",
            Self::And => "and",
            Self::OrPrefix => "or_prefix",
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SearchProvenance {
    pub asks_ai: bool,
    pub ai_command: Option<String>,
    pub is_command: bool,
    pub is_synthetic_record: bool,
}

#[derive(Debug, Clone)]
pub struct SearchDocument {
    pub search_text: Option<String>,
    pub search_tags: Option<String>,
    pub provenance: SearchProvenance,
}

#[derive(Debug, Clone)]
pub struct SearchQuery {
    pub phrase_text: Option<String>,
    pub phrase_eligible: bool,
    pub semantic_tokens: Vec<String>,
    pub tag_tokens: Vec<String>,
    pub snippet_terms: Vec<String>,
}

pub fn derive_search_provenance(text: &str) -> SearchProvenance {
    let trimmed = text.trim();
    let command = COMMAND_REGEX
        .captures(trimmed)
        .and_then(|captures| captures.get(1))
        .map(|capture| capture.as_str().to_lowercase());
    let ai_command = command
        .as_deref()
        .filter(|value| matches!(*value, "q" | "qc" | "qq" | "factcheck"))
        .map(str::to_string);
    let synthetic_lower = trimmed.to_lowercase();
    let synthetic_command = if synthetic_lower.starts_with("ask about chat ") {
        Some("qc".to_string())
    } else if synthetic_lower.starts_with("ask ") {
        Some("q".to_string())
    } else {
        None
    };

    SearchProvenance {
        asks_ai: ai_command.is_some() || synthetic_command.is_some(),
        ai_command: ai_command.or(synthetic_command),
        is_command: trimmed.starts_with('/'),
        is_synthetic_record: synthetic_lower.starts_with("ask "),
    }
}

pub fn normalize_message_document(
    raw_text: Option<&str>,
    search_source_text: Option<&str>,
    explicit: &SearchProvenance,
) -> SearchDocument {
    let derived = raw_text.map(derive_search_provenance).unwrap_or_default();
    let provenance = merge_provenance(explicit, &derived);
    let semantic_source = search_source_text.unwrap_or_else(|| raw_text.unwrap_or_default());
    let normalized_semantic =
        normalize_semantic_text(semantic_source, provenance.is_synthetic_record);

    let mut tag_tokens = BTreeSet::new();
    let tag_source = search_source_text.unwrap_or_else(|| raw_text.unwrap_or_default());
    let _ = strip_urls_and_collect_tags(tag_source, &mut tag_tokens);
    add_provenance_tags(&mut tag_tokens, &provenance);

    let semantic_tokens = tokenize_search_text(&normalized_semantic);
    let search_text = build_search_text(&normalized_semantic, &semantic_tokens);
    let search_tags = join_tokens(tag_tokens);

    SearchDocument {
        search_text,
        search_tags,
        provenance,
    }
}

pub fn normalize_search_query(query: &str) -> SearchQuery {
    let query_lower = query.to_lowercase();
    let normalized_semantic = normalize_semantic_text(query, false);
    let semantic_tokens = tokenize_search_text(&normalized_semantic);
    let mut tag_tokens = BTreeSet::new();
    let _ = strip_urls_and_collect_tags(query, &mut tag_tokens);
    expand_alias_tags(&query_lower, &semantic_tokens, &mut tag_tokens);
    expand_command_tags(&query_lower, &mut tag_tokens);

    let phrase_text = (!normalized_semantic.is_empty()).then_some(normalized_semantic.clone());
    let han_chars = normalized_semantic.chars().filter(|ch| is_han(*ch)).count();
    let phrase_eligible = semantic_tokens.len() >= 2 || han_chars >= 4;

    SearchQuery {
        phrase_text,
        phrase_eligible,
        semantic_tokens: semantic_tokens.clone(),
        tag_tokens: tag_tokens.into_iter().collect(),
        snippet_terms: semantic_tokens,
    }
}

pub fn clean_text_for_display(text: &str, is_synthetic_record: bool) -> String {
    clean_display_text(text, is_synthetic_record)
}

fn merge_provenance(explicit: &SearchProvenance, derived: &SearchProvenance) -> SearchProvenance {
    SearchProvenance {
        asks_ai: explicit.asks_ai || derived.asks_ai,
        ai_command: explicit
            .ai_command
            .clone()
            .or_else(|| derived.ai_command.clone()),
        is_command: explicit.is_command || derived.is_command,
        is_synthetic_record: explicit.is_synthetic_record || derived.is_synthetic_record,
    }
}

fn normalize_semantic_text(text: &str, is_synthetic_record: bool) -> String {
    let mut normalized = ZERO_WIDTH_REGEX.replace_all(text, " ").to_string();
    normalized = HTML_TAG_REGEX.replace_all(&normalized, " ").to_string();
    normalized = MODEL_LINE_REGEX.replace_all(&normalized, " ").to_string();
    normalized = TELEGRAPH_WRAPPER_REGEX
        .replace_all(&normalized, " ")
        .to_string();
    normalized = VIEW_IT_HERE_REGEX.replace_all(&normalized, " ").to_string();
    if is_synthetic_record {
        normalized = ASK_PREFIX_REGEX.replace(&normalized, " ").to_string();
    }
    normalized = REPLY_CONTEXT_LABEL_REGEX
        .replace_all(&normalized, " ")
        .to_string();
    normalized = QUESTION_LABEL_REGEX
        .replace_all(&normalized, " ")
        .to_string();
    let mut url_tags = BTreeSet::new();
    normalized = strip_urls_and_collect_tags(&normalized, &mut url_tags);
    normalized = MARKDOWN_FORMATTING_REGEX
        .replace_all(&normalized, " ")
        .to_string();
    normalized = normalized
        .chars()
        .map(|ch| match ch {
            '[' | ']' | '(' | ')' | '{' | '}' | '"' | '\'' | '|' | '\\' | '>' => ' ',
            _ => ch,
        })
        .collect::<String>();
    normalized = WHITESPACE_REGEX
        .replace_all(&normalized, " ")
        .trim()
        .to_string();
    truncate_chars(&normalized.to_lowercase(), MAX_SEARCH_TEXT_CHARS)
}

fn clean_display_text(text: &str, is_synthetic_record: bool) -> String {
    let mut cleaned = ZERO_WIDTH_REGEX.replace_all(text, " ").to_string();
    cleaned = HTML_TAG_REGEX.replace_all(&cleaned, " ").to_string();
    cleaned = MODEL_LINE_REGEX.replace_all(&cleaned, " ").to_string();
    cleaned = TELEGRAPH_WRAPPER_REGEX
        .replace_all(&cleaned, " ")
        .to_string();
    cleaned = VIEW_IT_HERE_REGEX.replace_all(&cleaned, " ").to_string();
    if is_synthetic_record {
        cleaned = ASK_PREFIX_REGEX.replace(&cleaned, " ").to_string();
    }
    cleaned = REPLY_CONTEXT_LABEL_REGEX
        .replace_all(&cleaned, " ")
        .to_string();
    cleaned = QUESTION_LABEL_REGEX.replace_all(&cleaned, " ").to_string();
    let mut url_tags = BTreeSet::new();
    cleaned = strip_urls_and_collect_tags(&cleaned, &mut url_tags);
    cleaned = MARKDOWN_FORMATTING_REGEX
        .replace_all(&cleaned, " ")
        .to_string();
    cleaned = WHITESPACE_REGEX
        .replace_all(&cleaned, " ")
        .trim()
        .to_string();
    truncate_chars(&cleaned, MAX_SNIPPET_SOURCE_CHARS)
}

fn strip_urls_and_collect_tags(text: &str, tags: &mut BTreeSet<String>) -> String {
    URL_REGEX
        .replace_all(text, |captures: &regex::Captures<'_>| {
            if let Some(url_match) = captures.get(0) {
                collect_url_tags(url_match.as_str(), tags);
            }
            " "
        })
        .to_string()
}

fn collect_url_tags(url_text: &str, tags: &mut BTreeSet<String>) {
    let Ok(parsed) = Url::parse(url_text) else {
        return;
    };
    let Some(host) = parsed.host_str() else {
        return;
    };
    let normalized_host = host.trim_start_matches("www.").to_lowercase();
    if normalized_host.is_empty() {
        return;
    }

    tags.insert("has_url".to_string());
    tags.insert(domain_tag(&normalized_host));

    match normalized_host.as_str() {
        "x.com" => {
            tags.insert("twitter_link".to_string());
            tags.insert("x_link".to_string());
            tags.insert(domain_tag("twitter.com"));
        }
        "twitter.com" => {
            tags.insert("twitter_link".to_string());
            tags.insert("x_link".to_string());
            tags.insert(domain_tag("x.com"));
        }
        "youtube.com" => {
            tags.insert("youtube_link".to_string());
            tags.insert(domain_tag("youtu.be"));
        }
        "youtu.be" => {
            tags.insert("youtube_link".to_string());
            tags.insert(domain_tag("youtube.com"));
        }
        "telegra.ph" => {
            tags.insert("telegraph_link".to_string());
        }
        _ => {}
    }
}

fn add_provenance_tags(tags: &mut BTreeSet<String>, provenance: &SearchProvenance) {
    if provenance.asks_ai {
        tags.insert("asks_ai".to_string());
    }
    if let Some(command) = provenance.ai_command.as_deref() {
        tags.insert(command_tag(command));
    }
}

fn expand_alias_tags(
    query_lower: &str,
    semantic_tokens: &[String],
    tag_tokens: &mut BTreeSet<String>,
) {
    let raw_tokens = QUERY_TOKEN_REGEX
        .find_iter(query_lower)
        .map(|capture| capture.as_str().trim_matches('.').to_string())
        .collect::<BTreeSet<_>>();
    let has_link_word = semantic_tokens
        .iter()
        .any(|token| matches!(token.as_str(), "link" | "links" | "url" | "urls"))
        || query_lower.contains("\u{94fe}\u{63a5}")
        || query_lower.contains("\u{9023}\u{7d50}");

    if raw_tokens.contains("twitter")
        || raw_tokens.contains("twitter.com")
        || raw_tokens.contains("x.com")
        || raw_tokens.contains("tweet")
        || raw_tokens.contains("tweets")
        || raw_tokens.contains("x")
    {
        tag_tokens.insert("twitter_link".to_string());
        tag_tokens.insert("x_link".to_string());
        tag_tokens.insert(domain_tag("twitter.com"));
        tag_tokens.insert(domain_tag("x.com"));
    }

    if raw_tokens.contains("youtube")
        || raw_tokens.contains("youtube.com")
        || raw_tokens.contains("youtu.be")
        || raw_tokens.contains("video")
        || raw_tokens.contains("videos")
    {
        tag_tokens.insert("youtube_link".to_string());
        tag_tokens.insert(domain_tag("youtube.com"));
        tag_tokens.insert(domain_tag("youtu.be"));
    }

    if raw_tokens.contains("telegraph") || raw_tokens.contains("telegra.ph") {
        tag_tokens.insert("telegraph_link".to_string());
        tag_tokens.insert(domain_tag("telegra.ph"));
    }

    if has_link_word {
        tag_tokens.insert("has_url".to_string());
    }
}

fn expand_command_tags(query_lower: &str, tag_tokens: &mut BTreeSet<String>) {
    for command in ["q", "qc", "qq", "factcheck"] {
        if query_lower.contains(&format!("/{command}"))
            || query_lower.contains(&format!("command {command}"))
        {
            tag_tokens.insert(command_tag(command));
        }
    }
}

fn tokenize_search_text(text: &str) -> Vec<String> {
    if text.trim().is_empty() {
        return Vec::new();
    }

    let mut tokens = Vec::new();
    let mut seen = BTreeSet::new();

    for segment in JIEBA.cut(text, false) {
        let normalized = normalize_token(segment);
        if normalized.is_empty() || !token_should_be_indexed(&normalized) {
            continue;
        }
        if seen.insert(normalized.clone()) {
            tokens.push(normalized);
        }
    }

    let existing_tokens = tokens.clone();
    for token in existing_tokens {
        add_han_bigrams(&token, &mut tokens, &mut seen);
    }
    add_han_span_tokens(text, &mut tokens, &mut seen);

    tokens
}

fn normalize_token(token: &str) -> String {
    let trimmed = NON_TOKEN_EDGE_REGEX
        .replace_all(token.trim(), "")
        .to_string();
    let lowered = trimmed
        .trim_matches(|ch: char| ch == '#' || ch == '@')
        .to_lowercase();
    lowered
        .chars()
        .filter(|ch| ch.is_alphanumeric() || *ch == '_' || is_han(*ch))
        .collect()
}

fn token_should_be_indexed(token: &str) -> bool {
    if token.is_empty() {
        return false;
    }
    if token.chars().any(is_han) {
        return true;
    }
    if token.len() == 1 && token.chars().all(|ch| ch.is_ascii_alphabetic()) {
        return false;
    }
    true
}

fn add_han_bigrams(token: &str, tokens: &mut Vec<String>, seen: &mut BTreeSet<String>) {
    if !token.chars().all(is_han) {
        return;
    }

    let chars = token.chars().collect::<Vec<_>>();
    if chars.len() < 2 {
        return;
    }

    for window in chars.windows(2) {
        let bigram = window.iter().collect::<String>();
        if seen.insert(bigram.clone()) {
            tokens.push(bigram);
        }
    }
}

fn add_han_span_tokens(text: &str, tokens: &mut Vec<String>, seen: &mut BTreeSet<String>) {
    let mut current = String::new();
    for ch in text.chars() {
        if is_han(ch) {
            current.push(ch);
            continue;
        }
        flush_han_span(&mut current, tokens, seen);
    }
    flush_han_span(&mut current, tokens, seen);
}

fn flush_han_span(current: &mut String, tokens: &mut Vec<String>, seen: &mut BTreeSet<String>) {
    if current.chars().count() < 2 {
        current.clear();
        return;
    }

    if seen.insert(current.clone()) {
        tokens.push(current.clone());
    }
    add_han_bigrams(current, tokens, seen);
    current.clear();
}

fn build_search_text(normalized_semantic: &str, semantic_tokens: &[String]) -> Option<String> {
    let semantic = normalized_semantic.trim();
    let token_block = semantic_tokens.join(" ");
    let combined = match (semantic.is_empty(), token_block.is_empty()) {
        (true, true) => String::new(),
        (false, true) => semantic.to_string(),
        (true, false) => token_block,
        (false, false) => format!("{semantic}\n{token_block}"),
    };
    let combined = truncate_chars(combined.trim(), MAX_SEARCH_TEXT_CHARS);
    (!combined.is_empty()).then_some(combined)
}

fn join_tokens(tokens: BTreeSet<String>) -> Option<String> {
    let joined = tokens.into_iter().collect::<Vec<_>>().join(" ");
    (!joined.trim().is_empty()).then_some(joined)
}

fn domain_tag(domain: &str) -> String {
    format!("domain_{}", domain.replace('.', "_"))
}

fn command_tag(command: &str) -> String {
    format!("command_{command}")
}

fn truncate_chars(value: &str, max_chars: usize) -> String {
    if value.chars().count() <= max_chars {
        return value.to_string();
    }
    let mut truncated: String = value.chars().take(max_chars).collect();
    truncated.push_str("...");
    truncated
}

fn is_han(ch: char) -> bool {
    HAN_REGEX.is_match(&ch.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn synthetic_records_drop_bot_wrappers_from_semantic_search_text() {
        let provenance = SearchProvenance {
            asks_ai: true,
            ai_command: Some("qc".to_string()),
            is_command: true,
            is_synthetic_record: true,
        };
        let document = normalize_message_document(
            Some(
                "Ask about chat AI bot: Context from replied message: \"\u{65e7}\u{6d88}\u{606f}\"\n\nQuestion: \u{80a1}\u{7968}\u{600e}\u{4e48}\u{4e86}",
            ),
            Some(
                "Context from replied message: \"\u{65e7}\u{6d88}\u{606f}\"\n\nQuestion: \u{80a1}\u{7968}\u{600e}\u{4e48}\u{4e86}",
            ),
            &provenance,
        );

        let search_text = document.search_text.expect("search text should exist");
        assert!(!search_text.contains("ask about chat"));
        assert!(!search_text.contains("question:"));
        assert!(search_text.contains("\u{80a1}\u{7968}\u{600e}\u{4e48}\u{4e86}"));
    }

    #[test]
    fn normalization_extracts_url_tags_and_ai_provenance() {
        let document = normalize_message_document(
            Some("/qc look at this https://x.com/example/status/123"),
            None,
            &SearchProvenance::default(),
        );

        assert!(document.provenance.asks_ai);
        assert_eq!(document.provenance.ai_command.as_deref(), Some("qc"));
        let tags = document.search_tags.expect("tags should exist");
        assert!(tags.contains("has_url"));
        assert!(tags.contains("twitter_link"));
        assert!(tags.contains("x_link"));
        assert!(tags.contains("command_qc"));
    }

    #[test]
    fn chinese_queries_emit_segmented_tokens() {
        let query =
            normalize_search_query("\u{5173}\u{4e8e}\u{80a1}\u{7968}\u{7684}\u{5185}\u{5bb9}");
        assert!(query.phrase_eligible);
        assert!(query
            .semantic_tokens
            .iter()
            .any(|token| token.contains("\u{80a1}\u{7968}")));
    }

    #[test]
    fn query_aliases_expand_x_and_twitter_links() {
        let query = normalize_search_query("who posted the most twitter/x links");
        assert!(query.tag_tokens.iter().any(|token| token == "twitter_link"));
        assert!(query.tag_tokens.iter().any(|token| token == "x_link"));
        assert!(query.tag_tokens.iter().any(|token| token == "has_url"));
    }

    #[test]
    fn display_cleaning_removes_telegraph_wrappers() {
        let cleaned = clean_text_for_display(
            "Ask AI bot: [Telegraph content extracted from https://telegra.ph/x] I have too much to say. View it here. Model: gemini",
            true,
        );
        assert!(!cleaned.contains("Telegraph content extracted"));
        assert!(!cleaned.contains("View it here"));
        assert!(!cleaned.contains("Model:"));
    }
}
