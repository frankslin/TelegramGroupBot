use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use once_cell::sync::Lazy;
use parking_lot::Mutex;
use teloxide::prelude::*;
use teloxide::types::ReplyParameters;
use tracing::{info, warn};

use crate::config::CONFIG;

static RATE_LIMITS: Lazy<Mutex<HashMap<i64, Instant>>> = Lazy::new(|| Mutex::new(HashMap::new()));
static WHITELIST_CACHE: Lazy<Mutex<Option<Vec<String>>>> = Lazy::new(|| Mutex::new(None));
static WHITELIST_LOADED: AtomicBool = AtomicBool::new(false);

pub fn is_rate_limited(user_id: i64) -> bool {
    let mut limits = RATE_LIMITS.lock();
    let now = Instant::now();

    if let Some(last) = limits.get(&user_id) {
        if now.duration_since(*last) < Duration::from_secs(CONFIG.rate_limit_seconds) {
            return true;
        }
    }

    limits.insert(user_id, now);
    false
}

pub fn load_whitelist() {
    if WHITELIST_LOADED.swap(true, Ordering::SeqCst) {
        return;
    }

    let path = &CONFIG.whitelist_file_path;
    let file = std::fs::read_to_string(path);
    let mut cache = WHITELIST_CACHE.lock();

    match file {
        Ok(content) => {
            let ids = content
                .lines()
                .map(|line| line.trim())
                .filter(|line| !line.is_empty() && !line.starts_with('#'))
                .map(|line| line.to_string())
                .collect::<Vec<_>>();
            *cache = Some(ids);
            info!("Loaded whitelist file {}", path);
        }
        Err(err) => {
            warn!(
                "Whitelist file {} not found or failed to read: {}",
                path, err
            );
            *cache = None;
        }
    }
}

pub fn is_user_whitelisted(user_id: i64) -> bool {
    if !WHITELIST_LOADED.load(Ordering::SeqCst) {
        load_whitelist();
    }
    let cache = WHITELIST_CACHE.lock();
    match &*cache {
        None => true,
        Some(list) => list.contains(&user_id.to_string()),
    }
}

pub fn is_chat_whitelisted(chat_id: i64) -> bool {
    if !WHITELIST_LOADED.load(Ordering::SeqCst) {
        load_whitelist();
    }
    let cache = WHITELIST_CACHE.lock();
    match &*cache {
        None => true,
        Some(list) => list.contains(&chat_id.to_string()),
    }
}

pub fn is_access_allowed(user_id: i64, chat_id: i64) -> bool {
    is_user_whitelisted(user_id) || is_chat_whitelisted(chat_id)
}

fn normalize_command_name(command: &str) -> String {
    command.trim().trim_start_matches('/').to_ascii_lowercase()
}

pub fn requires_access_control(command: &str) -> bool {
    if CONFIG.access_controlled_commands.is_empty() {
        return false;
    }
    let command = normalize_command_name(command);
    CONFIG
        .access_controlled_commands
        .iter()
        .any(|entry| normalize_command_name(entry) == command)
}

pub async fn check_access_control(bot: &Bot, message: &Message, command: &str) -> bool {
    if !requires_access_control(command) {
        return true;
    }

    let user_id = message
        .from
        .as_ref()
        .and_then(|user| i64::try_from(user.id.0).ok())
        .unwrap_or_default();
    let chat_id = message.chat.id.0;

    if !is_access_allowed(user_id, chat_id) {
        let _ = bot
            .send_message(
                message.chat.id,
                "You are not authorized to use this command. Please contact the administrator.",
            )
            .reply_parameters(ReplyParameters::new(message.id))
            .await;
        return false;
    }

    true
}

#[cfg(test)]
#[allow(clippy::items_after_test_module)]
mod tests {
    use super::normalize_command_name;

    #[test]
    fn normalize_command_name_trims_slash_and_case() {
        assert_eq!(normalize_command_name("/ProfileMe "), "profileme");
        assert_eq!(normalize_command_name("mysong"), "mysong");
    }
}

pub async fn check_admin_access(bot: &Bot, message: &Message, command: &str) -> bool {
    if !WHITELIST_LOADED.load(Ordering::SeqCst) {
        load_whitelist();
    }

    let whitelist = {
        let cache = WHITELIST_CACHE.lock();
        cache.clone()
    };

    let Some(whitelist) = whitelist else {
        let _ = bot
            .send_message(
                message.chat.id,
                "Admin command is unavailable because no whitelist is configured. Add trusted user/chat IDs to the whitelist file and try again.",
            )
            .reply_parameters(ReplyParameters::new(message.id))
            .await;
        warn!(
            "Admin command '{}' denied because whitelist file is unavailable",
            command
        );
        return false;
    };

    let user_id = message
        .from
        .as_ref()
        .and_then(|user| i64::try_from(user.id.0).ok())
        .unwrap_or_default();
    let chat_id = message.chat.id.0;

    let allowed =
        whitelist.contains(&user_id.to_string()) || whitelist.contains(&chat_id.to_string());

    if !allowed {
        let _ = bot
            .send_message(
                message.chat.id,
                "This command is restricted to administrators.",
            )
            .reply_parameters(ReplyParameters::new(message.id))
            .await;
        return false;
    }

    true
}
