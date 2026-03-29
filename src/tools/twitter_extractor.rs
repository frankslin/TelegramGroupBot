use std::collections::HashSet;
use std::time::Duration;

use anyhow::{anyhow, Result};
use regex::Regex;
use tracing::{debug, info};
use url::Url;

use crate::utils::http::get_http_client;

#[derive(Debug, Clone)]
pub struct TwitterContent {
    #[allow(dead_code)]
    pub url: String,
    pub text_content: String,
    pub image_urls: Vec<String>,
    pub video_urls: Vec<String>,
    pub formatted_content: String,
}

const REQUEST_TIMEOUT: u64 = 20;
const USER_AGENT: &str =
    "TelegramGroupHelperBot/0.1 (+https://github.com/sailself/TelegramGroupHelperBot)";

fn is_supported_host(host: &str) -> bool {
    let mut host = host.to_lowercase();
    if host.starts_with("www.") {
        host = host.trim_start_matches("www.").to_string();
    }
    if host.ends_with("x.com") {
        return true;
    }
    let tokens = [
        "twitter.com",
        "fxtwitter.com",
        "vxtwitter.com",
        "fixupx.com",
        "fixvx.com",
        "twittpr.com",
        "pxtwitter.com",
        "tweetpik.com",
    ];
    tokens.iter().any(|token| host.ends_with(token))
}

fn normalize_status_url(raw_url: &str) -> Result<String> {
    if raw_url.trim().is_empty() {
        return Err(anyhow!("Empty URL provided for Twitter extraction"));
    }

    let mut candidate = raw_url.trim().to_string();
    if !candidate.starts_with("http://") && !candidate.starts_with("https://") {
        candidate = format!("https://{}", candidate);
    }

    let parsed = Url::parse(&candidate)?;
    let host = parsed.host_str().unwrap_or_default();
    if !is_supported_host(host) {
        return Err(anyhow!("Unsupported Twitter/X host: {}", host));
    }

    if !parsed.path().contains("/status/") {
        return Err(anyhow!("Twitter/X URL does not reference a status update"));
    }

    let mut canonical = parsed.clone();
    canonical.set_scheme("https").ok();
    canonical.set_host(Some("x.com")).ok();

    Ok(canonical.to_string())
}

fn build_proxy_url(normalized_url: &str) -> String {
    let stripped = normalized_url.trim_start_matches("https://");
    format!("https://r.jina.ai/https://{}", stripped)
}

fn normalize_media_url(url: &str) -> String {
    let parsed = Url::parse(url)
        .or_else(|_| Url::parse(&format!("https:{}", url)))
        .unwrap_or_else(|_| Url::parse("https://x.com").unwrap());
    let mut query_pairs = parsed.query_pairs().collect::<Vec<_>>();
    if parsed
        .domain()
        .map(|d| d.ends_with("twimg.com"))
        .unwrap_or(false)
    {
        for pair in &mut query_pairs {
            if pair.0 == "name" {
                pair.1 = "orig".into();
            }
        }
    }
    let mut normalized = parsed.clone();
    if !query_pairs.is_empty() {
        normalized.set_query(Some(
            &serde_urlencoded::to_string(&query_pairs).unwrap_or_default(),
        ));
    }
    normalized.to_string()
}

fn looks_like_timestamp(text: &str) -> bool {
    let timestamp_pattern = Regex::new(r"\d{1,2}:\d{2}\s?[AP]M").unwrap();
    if timestamp_pattern.is_match(text) {
        return true;
    }
    let lowered = text.to_lowercase();
    if lowered.contains("am") || lowered.contains("pm") {
        let months = [
            "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "sept", "oct", "nov",
            "dec",
        ];
        return months.iter().any(|month| lowered.contains(month));
    }
    false
}

fn collect_relevant_lines(markdown_block: &str) -> Vec<String> {
    let mut relevant: Vec<String> = Vec::new();
    let mut collecting = false;
    let mut seen_content = false;

    for line in markdown_block.lines() {
        let stripped = line.trim();
        let lowered = stripped.to_lowercase();

        if !collecting {
            if lowered == "conversation" {
                collecting = true;
            }
            continue;
        }

        if !seen_content {
            if stripped.is_empty() || stripped.chars().all(|c| c == '-') {
                continue;
            }
            seen_content = true;
        }

        if stripped.is_empty() {
            if relevant
                .last()
                .map(|line| !line.is_empty())
                .unwrap_or(false)
            {
                relevant.push(String::new());
            }
            continue;
        }

        let stop_markers = [
            "new to x?",
            "join x today",
            "sign up now to get your own personalized timeline!",
            "sign up",
            "log in",
            "tweet your reply",
            "trending now",
            "what's happening",
            "terms of service",
            "privacy policy",
            "cookie policy",
            "accessibility",
            "ads info",
        ];
        let stop_prefixes = [
            "watch on",
            "show more",
            "related",
            "more replies",
            "explore",
            "tweet your reply",
        ];

        if stop_markers.iter().any(|marker| marker == &lowered)
            || stop_prefixes
                .iter()
                .any(|prefix| lowered.starts_with(prefix))
        {
            break;
        }

        relevant.push(line.to_string());
    }

    if relevant.is_empty() {
        for line in markdown_block.lines() {
            let stripped = line.trim();
            if stripped.is_empty() {
                continue;
            }
            let lowered = stripped.to_lowercase();
            let stop_markers = [
                "new to x?",
                "join x today",
                "sign up now to get your own personalized timeline!",
                "sign up",
                "log in",
                "tweet your reply",
                "trending now",
                "what's happening",
                "terms of service",
                "privacy policy",
                "cookie policy",
                "accessibility",
                "ads info",
            ];
            let stop_prefixes = [
                "watch on",
                "show more",
                "related",
                "more replies",
                "explore",
                "tweet your reply",
            ];
            if stop_markers.iter().any(|marker| marker == &lowered)
                || stop_prefixes
                    .iter()
                    .any(|prefix| lowered.starts_with(prefix))
            {
                break;
            }
            relevant.push(line.to_string());
        }
    }

    relevant
}

fn clean_lines_and_media(lines: &[String]) -> (Vec<String>, Vec<String>, Vec<String>) {
    let media_pattern = Regex::new(r"!\[[^\]]*?\]\((https?://[^\)]+)\)").unwrap();
    let link_pattern = Regex::new(r"\[([^\]]*?)\]\((https?://[^\)]+)\)").unwrap();
    let empty_link_pattern = Regex::new(r"\[\s*\]\((https?://[^\)]+)\)").unwrap();
    let whitespace_pattern = Regex::new(r"\s+").unwrap();

    let mut cleaned: Vec<String> = Vec::new();
    let mut image_urls = Vec::new();
    let mut video_urls = Vec::new();

    let profile_media_tokens = [
        "profile_images",
        "profile_banners",
        "semantic_core_img",
        "/emoji/",
        "responsive-web/client-web",
    ];
    let video_extensions = [".mp4", ".m3u8", ".mpd"];
    let punct_prefixes = [".", ",", ";", ":", ")", "]", "}", "!", "?"];

    for line in lines {
        let mut working = line.clone();
        for caps in media_pattern.captures_iter(&working) {
            let media_url = normalize_media_url(&caps[1]);
            debug!(
                target: "content.extract",
                source = "twitter",
                media_url = %media_url,
                line = %line,
                "Found Twitter media candidate"
            );
            if media_url.to_ascii_lowercase().contains(".svg") {
                debug!(
                    target: "content.extract",
                    source = "twitter",
                    media_url = %media_url,
                    line = %line,
                    reason = "svg",
                    "Skipping Twitter media candidate"
                );
                continue;
            }
            if profile_media_tokens
                .iter()
                .any(|token| media_url.contains(token))
            {
                debug!(
                    target: "content.extract",
                    source = "twitter",
                    media_url = %media_url,
                    line = %line,
                    reason = "profile_media",
                    "Skipping Twitter media candidate"
                );
                continue;
            }
            if video_extensions.iter().any(|ext| media_url.ends_with(ext))
                || media_url.contains("video.twimg.com")
            {
                if !video_urls.contains(&media_url) {
                    debug!(
                        target: "content.extract",
                        source = "twitter",
                        media_url = %media_url,
                        line = %line,
                        media_kind = "video",
                        "Added Twitter video"
                    );
                    video_urls.push(media_url);
                } else {
                    debug!(
                        target: "content.extract",
                        source = "twitter",
                        media_url = %media_url,
                        line = %line,
                        media_kind = "video",
                        reason = "duplicate",
                        "Skipping duplicate Twitter video"
                    );
                }
            } else if !image_urls.contains(&media_url) {
                debug!(
                    target: "content.extract",
                    source = "twitter",
                    media_url = %media_url,
                    line = %line,
                    media_kind = "image",
                    "Added Twitter image"
                );
                image_urls.push(media_url);
            } else {
                debug!(
                    target: "content.extract",
                    source = "twitter",
                    media_url = %media_url,
                    line = %line,
                    media_kind = "image",
                    reason = "duplicate",
                    "Skipping duplicate Twitter image"
                );
            }
        }
        working = media_pattern.replace_all(&working, "").to_string();
        working = empty_link_pattern.replace_all(&working, "").to_string();

        working = link_pattern
            .replace_all(&working, |caps: &regex::Captures| {
                caps[1].trim().to_string()
            })
            .to_string();
        working = whitespace_pattern
            .replace_all(&working, " ")
            .trim()
            .to_string();

        if working.is_empty() {
            continue;
        }

        if let Some(last) = cleaned.last_mut() {
            if !last.ends_with(['.', '!', '?', ':'])
                && (working.starts_with('@') || working.starts_with('#'))
            {
                *last = format!("{} {}", last, working);
                continue;
            }
            if punct_prefixes
                .iter()
                .any(|prefix| working.starts_with(prefix))
            {
                *last = format!("{}{}", last, working);
                continue;
            }
        }

        cleaned.push(working);
    }

    (cleaned, image_urls, video_urls)
}

fn extract_metadata(cleaned_lines: &[String]) -> (Option<String>, Option<String>, Option<usize>) {
    let mut display_name = None;
    let mut handle = None;
    let mut handle_index = None;

    for (idx, line) in cleaned_lines.iter().take(6).enumerate() {
        if line.starts_with('@') && !line.contains(' ') {
            handle = Some(line.to_string());
            handle_index = Some(idx);
            if let Some(prev) = cleaned_lines[..idx]
                .iter()
                .rev()
                .find(|line| !line.is_empty())
            {
                display_name = Some(prev.to_string());
            }
            break;
        }
    }

    (display_name, handle, handle_index)
}

fn strip_indices(lines: &[String], indexes: &[Option<usize>]) -> Vec<String> {
    let mut skip = HashSet::new();
    for value in indexes.iter().flatten() {
        skip.insert(*value);
    }
    lines
        .iter()
        .enumerate()
        .filter_map(|(idx, line)| {
            if skip.contains(&idx) {
                None
            } else {
                Some(line.clone())
            }
        })
        .collect()
}

pub async fn extract_twitter_content(url: &str) -> Result<TwitterContent> {
    let normalized_url = normalize_status_url(url)?;
    let proxy_url = build_proxy_url(&normalized_url);
    info!("Fetching Twitter/X content via proxy: {}", proxy_url);

    let client = get_http_client();
    let response = client
        .get(proxy_url)
        .timeout(Duration::from_secs(REQUEST_TIMEOUT))
        .header("User-Agent", USER_AGENT)
        .send()
        .await?;

    if !response.status().is_success() {
        return Err(anyhow!(
            "Twitter proxy request failed with status {}",
            response.status()
        ));
    }

    let raw_text = response.text().await?;
    let marker = "Markdown Content:\n";
    let marker_idx = raw_text
        .find(marker)
        .ok_or_else(|| anyhow!("Unable to locate Twitter/X markdown content in response"))?;
    let markdown_block = &raw_text[marker_idx + marker.len()..];

    let relevant_lines = collect_relevant_lines(markdown_block);
    let (cleaned_lines, image_urls, video_urls) = clean_lines_and_media(&relevant_lines);

    if cleaned_lines.is_empty() && image_urls.is_empty() && video_urls.is_empty() {
        return Err(anyhow!("No content extracted from Twitter/X response"));
    }

    let (display_name, handle, handle_idx) = extract_metadata(&cleaned_lines);
    let mut timestamp_idx = None;
    let mut timestamp_text = None;
    for (idx, line) in cleaned_lines.iter().enumerate() {
        if looks_like_timestamp(line) {
            timestamp_idx = Some(idx);
            timestamp_text = Some(line.clone());
            break;
        }
    }

    let display_index = display_name
        .as_ref()
        .and_then(|name| cleaned_lines.iter().position(|line| line == name));
    let body_lines = strip_indices(&cleaned_lines, &[handle_idx, display_index, timestamp_idx]);
    let body_text = body_lines.join("\n").trim().to_string();

    let mut header_parts = Vec::new();
    if let Some(name) = display_name.clone() {
        header_parts.push(name);
    }
    if let Some(handle_value) = handle.clone() {
        if !header_parts.contains(&handle_value) {
            header_parts.push(handle_value);
        }
    }

    let mut header_text = String::new();
    if !header_parts.is_empty() {
        header_text = format!("Tweet by {}", header_parts.join(" "));
    }
    if let Some(timestamp) = timestamp_text.clone() {
        if header_text.is_empty() {
            header_text = format!("Tweet at {}", timestamp);
        } else {
            header_text = format!("{} at {}", header_text, timestamp);
        }
    }

    let mut sections = Vec::new();
    if !header_text.trim().is_empty() {
        sections.push(header_text.trim().to_string());
    }
    if !body_text.trim().is_empty() {
        sections.push(body_text.clone());
    }
    sections.push(format!("Original link: {}", normalized_url));

    let text_content = sections.join("\n\n");
    let mut formatted_content = format!("\n\n--- Twitter Content ---\n{}", text_content);
    if !image_urls.is_empty() {
        formatted_content.push_str(&format!(
            "\n\nImages attached: {} image(s)",
            image_urls.len()
        ));
    }
    if !video_urls.is_empty() {
        formatted_content.push_str(&format!("\nVideos attached: {} video(s)", video_urls.len()));
    }
    formatted_content.push_str("\n--- End Twitter Content ---\n\n");

    Ok(TwitterContent {
        url: normalized_url,
        text_content,
        image_urls,
        video_urls,
        formatted_content,
    })
}
