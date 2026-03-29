use std::time::Duration;

use anyhow::{anyhow, Result};
use serde::Deserialize;
use tracing::info;
use url::Url;

use crate::utils::http::get_http_client;

#[derive(Debug, Deserialize)]
struct TelegraphResponse {
    ok: bool,
    result: Option<TelegraphResult>,
    error: Option<String>,
}

#[derive(Debug, Deserialize)]
struct TelegraphResult {
    content: Option<Vec<serde_json::Value>>,
}

#[derive(Debug, Default, Clone)]
pub struct TelegraphContent {
    pub text_content: String,
    pub image_urls: Vec<String>,
    pub video_urls: Vec<String>,
}

pub async fn extract_telegraph_content(url: &str) -> Result<TelegraphContent> {
    info!("Starting Telegraph extraction for url: {}", url);
    let parsed = Url::parse(url)?;
    let path = parsed.path().trim_start_matches('/');
    if path.is_empty() {
        return Err(anyhow!("Invalid Telegraph URL: Missing path component"));
    }

    let api_url = format!(
        "https://api.telegra.ph/getPage/{}?return_content=true",
        path
    );
    let client = get_http_client();
    let response = client
        .get(&api_url)
        .timeout(Duration::from_secs(15))
        .send()
        .await?;

    if !response.status().is_success() {
        return Err(anyhow!(
            "Telegraph API request failed with status {}",
            response.status()
        ));
    }

    let data = response.json::<TelegraphResponse>().await?;
    if !data.ok {
        return Err(anyhow!(
            "Telegraph API error: {}",
            data.error.unwrap_or_else(|| "Unknown error".to_string())
        ));
    }

    let nodes = data
        .result
        .and_then(|result| result.content)
        .unwrap_or_default();

    let mut content = TelegraphContent::default();
    let mut image_counter = 0;
    let mut video_counter = 0;

    fn process_nodes(
        nodes: &[serde_json::Value],
        content: &mut TelegraphContent,
        image_counter: &mut usize,
        video_counter: &mut usize,
    ) -> String {
        let mut current_text = String::new();
        for node in nodes {
            if let Some(text) = node.as_str() {
                current_text.push_str(text);
                continue;
            }

            let Some(obj) = node.as_object() else {
                continue;
            };

            let tag = obj.get("tag").and_then(|v| v.as_str()).unwrap_or("");
            let children = obj
                .get("children")
                .and_then(|v| v.as_array())
                .cloned()
                .unwrap_or_default();

            match tag {
                "img" => {
                    *image_counter += 1;
                    if let Some(src) = obj
                        .get("attrs")
                        .and_then(|v| v.get("src"))
                        .and_then(|v| v.as_str())
                    {
                        let url = if src.starts_with('/') {
                            format!("https://telegra.ph{}", src)
                        } else {
                            src.to_string()
                        };
                        content.image_urls.push(url.clone());
                        current_text.push_str(&format!("[image_{}]", image_counter));
                    }
                }
                "video" | "iframe" => {
                    *video_counter += 1;
                    if let Some(src) = obj
                        .get("attrs")
                        .and_then(|v| v.get("src"))
                        .and_then(|v| v.as_str())
                    {
                        let url = if src.starts_with("/embed/youtube") {
                            format!("https://www.youtube.com{}", src)
                        } else if src.starts_with("/embed/vimeo") {
                            format!("https://player.vimeo.com{}", src)
                        } else if src.starts_with('/') {
                            format!("https://telegra.ph{}", src)
                        } else {
                            src.to_string()
                        };
                        content.video_urls.push(url.clone());
                        current_text.push_str(&format!("[video_{}]", video_counter));
                    }
                }
                "figure" => {
                    current_text.push_str(&process_nodes(
                        &children,
                        content,
                        image_counter,
                        video_counter,
                    ));
                    current_text.push('\n');
                }
                "p" | "a" | "li" | "h3" | "h4" | "em" | "strong" | "figcaption" | "blockquote"
                | "code" | "span" => {
                    current_text.push_str(&process_nodes(
                        &children,
                        content,
                        image_counter,
                        video_counter,
                    ));
                    if matches!(tag, "p" | "h3" | "h4" | "li" | "blockquote") {
                        current_text.push('\n');
                    }
                }
                "br" => current_text.push('\n'),
                "hr" => current_text.push_str("\n---\n"),
                "ul" | "ol" => {
                    for child in &children {
                        if let Some(child_tag) = child.get("tag").and_then(|v| v.as_str()) {
                            if child_tag == "li" {
                                let child_nodes = child
                                    .get("children")
                                    .and_then(|v| v.as_array())
                                    .cloned()
                                    .unwrap_or_default();
                                current_text.push_str("- ");
                                current_text.push_str(&process_nodes(
                                    &child_nodes,
                                    content,
                                    image_counter,
                                    video_counter,
                                ));
                                current_text.push('\n');
                                continue;
                            }
                        }
                        if let Some(child_nodes) = child.as_array() {
                            current_text.push_str(&process_nodes(
                                child_nodes,
                                content,
                                image_counter,
                                video_counter,
                            ));
                        } else {
                            current_text.push_str(&process_nodes(
                                std::slice::from_ref(child),
                                content,
                                image_counter,
                                video_counter,
                            ));
                        }
                    }
                    current_text.push('\n');
                }
                "pre" => {
                    let mut code_block = String::new();
                    for child in &children {
                        if child.get("tag").and_then(|v| v.as_str()) == Some("code") {
                            let child_nodes = child
                                .get("children")
                                .and_then(|v| v.as_array())
                                .cloned()
                                .unwrap_or_default();
                            code_block.push_str(&process_nodes(
                                &child_nodes,
                                content,
                                image_counter,
                                video_counter,
                            ));
                        } else {
                            code_block.push_str(&process_nodes(
                                std::slice::from_ref(child),
                                content,
                                image_counter,
                                video_counter,
                            ));
                        }
                    }
                    current_text.push_str(&format!("\n```\n{}\n```\n", code_block.trim()));
                }
                _ => {
                    if !children.is_empty() {
                        current_text.push_str(&process_nodes(
                            &children,
                            content,
                            image_counter,
                            video_counter,
                        ));
                    }
                }
            }
        }
        current_text
    }

    content.text_content =
        process_nodes(&nodes, &mut content, &mut image_counter, &mut video_counter)
            .trim()
            .to_string();

    Ok(content)
}
