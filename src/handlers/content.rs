use std::time::Duration;

use pulldown_cmark::{Event, HeadingLevel, Options, Parser, Tag};
use regex::Regex;
use reqwest::header::CONTENT_TYPE;
use serde::Deserialize;
use serde_json::json;
use teloxide::types::{MessageEntityKind, MessageEntityRef};
use tracing::{debug, warn};

use crate::config::CONFIG;
use crate::llm::media::{detect_mime_type, download_media, MediaFile, MediaKind};
use crate::tools::telegraph_extractor::{extract_telegraph_content, TelegraphContent};
use crate::tools::twitter_extractor::{extract_twitter_content, TwitterContent};
use crate::utils::http::get_http_client;

fn truncate_for_log(value: &str, limit: usize) -> String {
    if value.chars().count() <= limit {
        return value.to_string();
    }
    let truncated: String = value.chars().take(limit).collect();
    format!("{}...", truncated)
}

fn log_extracted_content(
    source: &str,
    url: &str,
    text: &str,
    images: usize,
    videos: usize,
    audios: usize,
) {
    debug!(
        target: "content.extract",
        source = source,
        url = url,
        images = images,
        videos = videos,
        audios = audios,
        text = %truncate_for_log(text, 200)
    );
}

fn markdown_to_telegraph_nodes(content: &str) -> Vec<serde_json::Value> {
    if content.trim().is_empty() {
        return Vec::new();
    }

    #[derive(Debug)]
    struct NodeBuilder {
        tag: String,
        attrs: Option<serde_json::Map<String, serde_json::Value>>,
        children: Vec<serde_json::Value>,
    }

    enum StackEntry {
        Node(NodeBuilder),
        Image { src: String, alt: String },
    }

    fn push_text(children: &mut Vec<serde_json::Value>, text: &str) {
        if text.is_empty() {
            return;
        }
        if let Some(serde_json::Value::String(existing)) = children.last_mut() {
            existing.push_str(text);
            return;
        }
        children.push(serde_json::Value::String(text.to_string()));
    }

    fn push_value(
        stack: &mut [StackEntry],
        root: &mut Vec<serde_json::Value>,
        value: serde_json::Value,
    ) {
        if let Some(StackEntry::Node(parent)) = stack.last_mut() {
            parent.children.push(value);
        } else {
            root.push(value);
        }
    }

    fn close_node(stack: &mut Vec<StackEntry>, root: &mut Vec<serde_json::Value>) {
        let Some(entry) = stack.pop() else {
            return;
        };
        match entry {
            StackEntry::Node(node) => {
                let mut obj = serde_json::Map::new();
                obj.insert("tag".to_string(), serde_json::Value::String(node.tag));
                if let Some(attrs) = node.attrs {
                    obj.insert("attrs".to_string(), serde_json::Value::Object(attrs));
                }
                if !node.children.is_empty() {
                    obj.insert(
                        "children".to_string(),
                        serde_json::Value::Array(node.children),
                    );
                }
                push_value(stack, root, serde_json::Value::Object(obj));
            }
            StackEntry::Image { src, alt } => {
                if !src.is_empty() {
                    push_value(
                        stack,
                        root,
                        json!({
                            "tag": "img",
                            "attrs": { "src": src }
                        }),
                    );
                }
                if !alt.trim().is_empty() {
                    push_value(
                        stack,
                        root,
                        json!({
                            "tag": "figcaption",
                            "children": [alt.trim()]
                        }),
                    );
                }
            }
        }
    }

    let mut options = Options::empty();
    options.insert(Options::ENABLE_STRIKETHROUGH);
    options.insert(Options::ENABLE_TASKLISTS);

    let parser = Parser::new_ext(content, options);
    let mut root: Vec<serde_json::Value> = Vec::new();
    let mut stack: Vec<StackEntry> = Vec::new();

    for event in parser {
        match event {
            Event::Start(tag) => match tag {
                Tag::Paragraph => stack.push(StackEntry::Node(NodeBuilder {
                    tag: "p".to_string(),
                    attrs: None,
                    children: Vec::new(),
                })),
                Tag::Heading(level, _, _) => {
                    let tag_name = match level {
                        HeadingLevel::H1 | HeadingLevel::H2 | HeadingLevel::H3 => "h3",
                        _ => "h4",
                    };
                    stack.push(StackEntry::Node(NodeBuilder {
                        tag: tag_name.to_string(),
                        attrs: None,
                        children: Vec::new(),
                    }))
                }
                Tag::BlockQuote => stack.push(StackEntry::Node(NodeBuilder {
                    tag: "blockquote".to_string(),
                    attrs: None,
                    children: Vec::new(),
                })),
                Tag::List(Some(_)) => stack.push(StackEntry::Node(NodeBuilder {
                    tag: "ol".to_string(),
                    attrs: None,
                    children: Vec::new(),
                })),
                Tag::List(None) => stack.push(StackEntry::Node(NodeBuilder {
                    tag: "ul".to_string(),
                    attrs: None,
                    children: Vec::new(),
                })),
                Tag::Item => stack.push(StackEntry::Node(NodeBuilder {
                    tag: "li".to_string(),
                    attrs: None,
                    children: Vec::new(),
                })),
                Tag::Emphasis => stack.push(StackEntry::Node(NodeBuilder {
                    tag: "em".to_string(),
                    attrs: None,
                    children: Vec::new(),
                })),
                Tag::Strong => stack.push(StackEntry::Node(NodeBuilder {
                    tag: "strong".to_string(),
                    attrs: None,
                    children: Vec::new(),
                })),
                Tag::Strikethrough => stack.push(StackEntry::Node(NodeBuilder {
                    tag: "s".to_string(),
                    attrs: None,
                    children: Vec::new(),
                })),
                Tag::CodeBlock(_kind) => stack.push(StackEntry::Node(NodeBuilder {
                    tag: "pre".to_string(),
                    attrs: None,
                    children: Vec::new(),
                })),
                Tag::Link(_, dest, _) => {
                    let mut attrs = serde_json::Map::new();
                    attrs.insert(
                        "href".to_string(),
                        serde_json::Value::String(dest.to_string()),
                    );
                    stack.push(StackEntry::Node(NodeBuilder {
                        tag: "a".to_string(),
                        attrs: Some(attrs),
                        children: Vec::new(),
                    }))
                }
                Tag::Image(_, dest, _) => stack.push(StackEntry::Image {
                    src: dest.to_string(),
                    alt: String::new(),
                }),
                _ => {}
            },
            Event::End(tag) => match tag {
                Tag::Image(_, _, _) => close_node(&mut stack, &mut root),
                Tag::Paragraph
                | Tag::Heading(..)
                | Tag::BlockQuote
                | Tag::List(_)
                | Tag::Item
                | Tag::Emphasis
                | Tag::Strong
                | Tag::Strikethrough
                | Tag::Link(_, _, _)
                | Tag::CodeBlock(_) => close_node(&mut stack, &mut root),
                _ => {}
            },
            Event::Text(text) => {
                if let Some(StackEntry::Image { alt, .. }) = stack.last_mut() {
                    alt.push_str(&text);
                } else if let Some(StackEntry::Node(parent)) = stack.last_mut() {
                    push_text(&mut parent.children, &text);
                } else {
                    push_text(&mut root, &text);
                }
            }
            Event::Code(text) => {
                if let Some(StackEntry::Image { alt, .. }) = stack.last_mut() {
                    alt.push_str(&text);
                } else {
                    push_value(
                        &mut stack,
                        &mut root,
                        json!({
                            "tag": "code",
                            "children": [text.as_ref()]
                        }),
                    );
                }
            }
            Event::SoftBreak | Event::HardBreak => {
                push_value(&mut stack, &mut root, json!({ "tag": "br" }));
            }
            Event::Rule => {
                push_value(&mut stack, &mut root, json!({ "tag": "hr" }));
            }
            Event::Html(html) => {
                if let Some(StackEntry::Image { alt, .. }) = stack.last_mut() {
                    alt.push_str(&html);
                } else if let Some(StackEntry::Node(parent)) = stack.last_mut() {
                    push_text(&mut parent.children, &html);
                } else {
                    push_text(&mut root, &html);
                }
            }
            Event::TaskListMarker(checked) => {
                let marker = if checked { "[x] " } else { "[ ] " };
                if let Some(StackEntry::Image { alt, .. }) = stack.last_mut() {
                    alt.push_str(marker);
                } else if let Some(StackEntry::Node(parent)) = stack.last_mut() {
                    push_text(&mut parent.children, marker);
                } else {
                    push_text(&mut root, marker);
                }
            }
            Event::FootnoteReference(_) => {}
        }
    }

    while !stack.is_empty() {
        close_node(&mut stack, &mut root);
    }

    root
}

#[derive(Debug, Deserialize)]
struct TelegraphCreateResponse {
    ok: bool,
    result: Option<TelegraphCreateResult>,
    error: Option<String>,
}

#[derive(Debug, Deserialize)]
struct TelegraphCreateResult {
    url: String,
}

pub async fn create_telegraph_page(title: &str, content: &str) -> Option<String> {
    if CONFIG.telegraph_access_token.trim().is_empty() {
        warn!("Telegraph access token missing; skipping page creation");
        return None;
    }

    let nodes = markdown_to_telegraph_nodes(content);
    let content_json = serde_json::to_string(&nodes).unwrap_or_else(|_| "[]".to_string());
    let form = vec![
        (
            "access_token".to_string(),
            CONFIG.telegraph_access_token.clone(),
        ),
        (
            "author_name".to_string(),
            CONFIG.telegraph_author_name.clone(),
        ),
        (
            "author_url".to_string(),
            CONFIG.telegraph_author_url.clone(),
        ),
        ("title".to_string(), title.to_string()),
        ("content".to_string(), content_json),
        ("return_content".to_string(), "false".to_string()),
    ];

    let client = get_http_client();
    let response = client
        .post("https://api.telegra.ph/createPage")
        .timeout(Duration::from_secs(10))
        .form(&form)
        .send()
        .await
        .ok()?;

    if !response.status().is_success() {
        warn!(
            "Telegraph API call failed with status {}",
            response.status()
        );
        return None;
    }

    let payload = response.json::<TelegraphCreateResponse>().await.ok()?;
    if payload.ok {
        return payload.result.map(|result| result.url);
    }

    warn!("Telegraph API error: {}", payload.error.unwrap_or_default());
    None
}

pub fn extract_youtube_urls(text: &str, max_urls: usize) -> (String, Vec<String>) {
    if text.is_empty() {
        return (text.to_string(), Vec::new());
    }

    let pattern = Regex::new(
        r"((?:https?://)?(?:www\.|m\.)?(?:youtube\.com/(?:watch\?v=|shorts/)|youtu\.be/)([\w-]{11})(?:[\?&][^\s]*)?)",
    )
    .unwrap();
    let mut matches = pattern.captures_iter(text).collect::<Vec<_>>();
    let mut urls = Vec::new();
    let mut new_text = text.to_string();
    let mut count = 0;

    matches.reverse();
    for caps in matches {
        if count >= max_urls {
            break;
        }
        let vid_id = caps.get(2).map(|m| m.as_str()).unwrap_or("");
        if vid_id.is_empty() {
            continue;
        }
        let url = format!("https://www.youtube.com/watch?v={}", vid_id);
        urls.insert(0, url.clone());
        if let Some(m) = caps.get(0) {
            let start = m.start();
            let end = m.end();
            new_text.replace_range(start..end, &format!("YouTube_{}", vid_id));
        }
        count += 1;
    }

    (new_text, urls)
}

fn clean_url_candidate(url: &str) -> &str {
    url.trim_end_matches(|ch: char| {
        matches!(
            ch,
            ')' | ']' | '}' | '>' | '"' | '\'' | ',' | '.' | ';' | ':'
        )
    })
}

fn is_telegraph_url(url: &str) -> bool {
    let lowered = url.to_lowercase();
    lowered.contains("telegra.ph") || lowered.contains("t.me/")
}

pub async fn extract_telegraph_urls_and_content(
    text: &str,
    message_entities: Option<&[MessageEntityRef<'_>]>,
    max_urls: usize,
) -> (String, Vec<TelegraphContent>) {
    if text.is_empty() {
        return (text.to_string(), Vec::new());
    }

    let url_pattern = Regex::new(r#"https?://(?:telegra\.ph|t\.me)/[^\s\)>"]+"#).unwrap();
    let markdown_link_pattern = Regex::new(r#"\[[^\]]*\]\((https?://[^)]+)\)"#).unwrap();
    let html_link_pattern = Regex::new(r#"href=["'](https?://[^"']+)["']"#).unwrap();
    let mut urls = Vec::new();

    if let Some(entities) = message_entities {
        for entity in entities {
            if urls.len() >= max_urls {
                break;
            }
            let candidate = match entity.kind() {
                MessageEntityKind::Url => entity.text(),
                MessageEntityKind::TextLink { url } => url.as_str(),
                _ => continue,
            };
            let candidate = clean_url_candidate(candidate);
            if is_telegraph_url(candidate) {
                urls.push(candidate.to_string());
            }
        }
    }

    for m in url_pattern.find_iter(text) {
        urls.push(m.as_str().to_string());
    }
    for caps in markdown_link_pattern.captures_iter(text) {
        if let Some(url) = caps.get(1) {
            let candidate = clean_url_candidate(url.as_str());
            if is_telegraph_url(candidate) {
                urls.push(candidate.to_string());
            }
        }
    }
    for caps in html_link_pattern.captures_iter(text) {
        if let Some(url) = caps.get(1) {
            let candidate = clean_url_candidate(url.as_str());
            if is_telegraph_url(candidate) {
                urls.push(candidate.to_string());
            }
        }
    }

    urls.sort();
    urls.dedup();

    let mut new_text = text.to_string();
    let mut extracted = Vec::new();
    for url in urls.into_iter().take(max_urls) {
        match extract_telegraph_content(&url).await {
            Ok(content) => {
                log_extracted_content(
                    "telegraph",
                    &url,
                    &content.text_content,
                    content.image_urls.len(),
                    content.video_urls.len(),
                    0,
                );
                let formatted = format!(
                    "\n[Telegraph content extracted from {}]\n{}\n",
                    url, content.text_content
                );
                new_text.push_str(&formatted);
                extracted.push(content);
            }
            Err(err) => {
                warn!("Telegraph extraction failed for {}: {}", url, err);
                new_text.push_str(&format!(
                    "\n[Telegraph content extraction failed for {}]\n",
                    url
                ));
            }
        }
    }

    (new_text, extracted)
}

pub async fn extract_twitter_urls_and_content(
    text: &str,
    message_entities: Option<&[MessageEntityRef<'_>]>,
    max_urls: usize,
) -> (String, Vec<TwitterContent>) {
    if text.is_empty() {
        return (text.to_string(), Vec::new());
    }

    let url_pattern = Regex::new(
        r#"(https?://(?:www\.)?(?:x\.com|twitter\.com|mobile\.twitter\.com|m\.twitter\.com|fxtwitter\.com|vxtwitter\.com|fixupx\.com|fixvx\.com|twittpr\.com|pxtwitter\.com|tweetpik\.com)/[^\s\)>"]+)"#,
    )
    .unwrap();
    let markdown_link_pattern = Regex::new(r#"\[[^\]]*\]\((https?://[^)]+)\)"#).unwrap();
    let html_link_pattern = Regex::new(r#"href=["'](https?://[^"']+)["']"#).unwrap();

    let mut urls = Vec::new();
    if let Some(entities) = message_entities {
        for entity in entities {
            if urls.len() >= max_urls {
                break;
            }
            let candidate = match entity.kind() {
                MessageEntityKind::Url => entity.text(),
                MessageEntityKind::TextLink { url } => url.as_str(),
                _ => continue,
            };
            let candidate = clean_url_candidate(candidate);
            if url_pattern.is_match(candidate) {
                urls.push(candidate.to_string());
            }
        }
    }
    for m in url_pattern.find_iter(text) {
        urls.push(m.as_str().to_string());
    }
    for caps in markdown_link_pattern.captures_iter(text) {
        if let Some(url) = caps.get(1) {
            let candidate = clean_url_candidate(url.as_str());
            if url_pattern.is_match(candidate) {
                urls.push(candidate.to_string());
            }
        }
    }
    for caps in html_link_pattern.captures_iter(text) {
        if let Some(url) = caps.get(1) {
            let candidate = clean_url_candidate(url.as_str());
            if url_pattern.is_match(candidate) {
                urls.push(candidate.to_string());
            }
        }
    }
    urls.sort();
    urls.dedup();

    let mut new_text = text.to_string();
    let mut extracted = Vec::new();

    for url in urls.into_iter().take(max_urls) {
        match extract_twitter_content(&url).await {
            Ok(content) => {
                log_extracted_content(
                    "twitter",
                    &url,
                    &content.text_content,
                    content.image_urls.len(),
                    content.video_urls.len(),
                    0,
                );
                if !content.formatted_content.is_empty() {
                    new_text.push_str(&content.formatted_content);
                }
                extracted.push(content);
            }
            Err(err) => {
                warn!("Twitter extraction failed for {}: {}", url, err);
                new_text.push_str(&format!(
                    "\n[Twitter content extraction failed for {}]\n",
                    url
                ));
            }
        }
    }

    (new_text, extracted)
}

fn display_name_from_url(url: &str) -> Option<String> {
    let trimmed = url.split('?').next().unwrap_or(url);
    trimmed
        .rsplit('/')
        .next()
        .filter(|value| !value.is_empty())
        .map(|value| value.to_string())
}

fn image_mime_from_url(url: &str) -> Option<&'static str> {
    let lowered = url.to_ascii_lowercase();
    if lowered.contains("format=png") || lowered.ends_with(".png") {
        Some("image/png")
    } else if lowered.contains("format=jpg")
        || lowered.contains("format=jpeg")
        || lowered.ends_with(".jpg")
        || lowered.ends_with(".jpeg")
    {
        Some("image/jpeg")
    } else if lowered.contains("format=webp") || lowered.ends_with(".webp") {
        Some("image/webp")
    } else if lowered.ends_with(".heic") {
        Some("image/heic")
    } else if lowered.ends_with(".heif") {
        Some("image/heif")
    } else {
        None
    }
}

async fn download_image_with_content_type(url: &str, source: &str) -> Option<(Vec<u8>, String)> {
    let client = get_http_client();
    let response = match client.get(url).send().await {
        Ok(resp) => resp,
        Err(err) => {
            warn!(
                target: "content.extract",
                source = source,
                media_url = %url,
                error = %err,
                "Failed to fetch image"
            );
            return None;
        }
    };

    if !response.status().is_success() {
        warn!(
            target: "content.extract",
            source = source,
            media_url = %url,
            status = %response.status(),
            "Image download failed"
        );
        return None;
    }

    let content_type = response
        .headers()
        .get(CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .map(|value| {
            value
                .split(';')
                .next()
                .unwrap_or(value)
                .trim()
                .to_ascii_lowercase()
        })
        .filter(|value| !value.is_empty())
        .or_else(|| {
            let inferred = image_mime_from_url(url).map(|value| value.to_string());
            if let Some(mime_type) = inferred.as_deref() {
                debug!(
                    target: "content.extract",
                    source = source,
                    media_url = %url,
                    mime_type = %mime_type,
                    "Inferred image Content-Type from URL"
                );
            }
            inferred
        });

    let Some(content_type) = content_type else {
        warn!(
            target: "content.extract",
            source = source,
            media_url = %url,
            reason = "missing_content_type",
            "Skipping image without Content-Type or URL MIME hint"
        );
        return None;
    };

    if !content_type.starts_with("image/") {
        warn!(
            target: "content.extract",
            source = source,
            media_url = %url,
            content_type = %content_type,
            reason = "non_image_content_type",
            "Skipping non-image content"
        );
        return None;
    }

    let bytes = match response.bytes().await {
        Ok(bytes) => bytes.to_vec(),
        Err(err) => {
            warn!(
                target: "content.extract",
                source = source,
                media_url = %url,
                error = %err,
                "Failed to read image bytes"
            );
            return None;
        }
    };

    Some((bytes, content_type))
}

fn video_mime_from_url(url: &str) -> Option<&'static str> {
    let lowered = url.to_ascii_lowercase();
    if lowered.ends_with(".m3u8") {
        Some("application/x-mpegURL")
    } else if lowered.ends_with(".mpd") {
        Some("application/dash+xml")
    } else if lowered.ends_with(".webm") {
        Some("video/webm")
    } else if lowered.ends_with(".mp4") {
        Some("video/mp4")
    } else {
        None
    }
}

pub async fn download_telegraph_media(
    contents: &[TelegraphContent],
    max_files: usize,
) -> Vec<MediaFile> {
    let mut files = Vec::new();
    if max_files == 0 {
        return files;
    }

    for content in contents {
        for url in &content.image_urls {
            if files.len() >= max_files {
                return files;
            }
            if url.to_ascii_lowercase().contains(".svg") {
                warn!(target: "content.extract", source = "twitter", media_url = %url, reason = "svg", "Skipping Twitter image");
                continue;
            }
            if let Some((bytes, mime_type)) = download_image_with_content_type(url, "twitter").await
            {
                if mime_type == "image/svg+xml" {
                    warn!(
                        target: "content.extract",
                        source = "twitter",
                        media_url = %url,
                        content_type = %mime_type,
                        reason = "svg",
                        "Skipping Twitter image"
                    );
                    continue;
                }
                files.push(MediaFile::new(
                    bytes,
                    mime_type,
                    MediaKind::Image,
                    display_name_from_url(url),
                ));
            }
        }

        for url in &content.video_urls {
            if files.len() >= max_files {
                return files;
            }
            if let Some(bytes) = download_media(url).await {
                let mime_type = video_mime_from_url(url)
                    .map(|value| value.to_string())
                    .or_else(|| detect_mime_type(&bytes))
                    .unwrap_or_else(|| "video/mp4".to_string());
                files.push(MediaFile::new(
                    bytes,
                    mime_type,
                    MediaKind::Video,
                    display_name_from_url(url),
                ));
            }
        }
    }

    files
}

pub async fn download_twitter_media(
    contents: &[TwitterContent],
    max_files: usize,
) -> Vec<MediaFile> {
    let mut files = Vec::new();
    if max_files == 0 {
        return files;
    }

    for content in contents {
        for url in &content.image_urls {
            if files.len() >= max_files {
                return files;
            }
            if let Some(bytes) = download_media(url).await {
                let mime_type = detect_mime_type(&bytes).unwrap_or_else(|| "image/png".to_string());
                files.push(MediaFile::new(
                    bytes,
                    mime_type,
                    MediaKind::Image,
                    display_name_from_url(url),
                ));
            }
        }

        for url in &content.video_urls {
            if files.len() >= max_files {
                return files;
            }
            if let Some(bytes) = download_media(url).await {
                let mime_type = video_mime_from_url(url)
                    .map(|value| value.to_string())
                    .or_else(|| detect_mime_type(&bytes))
                    .unwrap_or_else(|| "video/mp4".to_string());
                files.push(MediaFile::new(
                    bytes,
                    mime_type,
                    MediaKind::Video,
                    display_name_from_url(url),
                ));
            }
        }
    }

    files
}
