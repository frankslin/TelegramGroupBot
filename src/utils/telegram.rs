use std::time::Duration;

use teloxide::prelude::*;
use teloxide::types::ChatAction;
use tokio::task::JoinHandle;
use tracing::warn;

const CHAT_ACTION_HEARTBEAT_INTERVAL: Duration = Duration::from_secs(4);

pub struct ChatActionHeartbeat {
    task_handle: Option<JoinHandle<()>>,
}

impl Drop for ChatActionHeartbeat {
    fn drop(&mut self) {
        if let Some(handle) = self.task_handle.take() {
            handle.abort();
        }
    }
}

pub fn start_chat_action_heartbeat(
    bot: Bot,
    chat_id: ChatId,
    action: ChatAction,
) -> ChatActionHeartbeat {
    let task_handle = tokio::spawn(async move {
        loop {
            if let Err(err) = bot.send_chat_action(chat_id, action).await {
                warn!("send_chat_action failed: {err}");
            }
            tokio::time::sleep(CHAT_ACTION_HEARTBEAT_INTERVAL).await;
        }
    });

    ChatActionHeartbeat {
        task_handle: Some(task_handle),
    }
}

pub fn normalize_supergroup_chat_id_for_link(chat_id: i64) -> Option<String> {
    let raw = chat_id.to_string();
    if let Some(normalized) = raw.strip_prefix("-100") {
        if normalized.is_empty() {
            None
        } else {
            Some(normalized.to_string())
        }
    } else {
        None
    }
}

pub fn build_message_link(chat_id: i64, message_id: i64) -> Option<String> {
    if let Some(normalized_chat_id) = normalize_supergroup_chat_id_for_link(chat_id) {
        return Some(format!(
            "https://t.me/c/{}/{}",
            normalized_chat_id, message_id
        ));
    }

    if chat_id > 0 {
        return Some(format!(
            "tg://openmessage?user_id={}&message_id={}",
            chat_id, message_id
        ));
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalizes_supergroup_chat_ids_for_links() {
        assert_eq!(
            normalize_supergroup_chat_id_for_link(-1001374348669),
            Some("1374348669".to_string())
        );
        assert_eq!(
            build_message_link(-1001374348669, 2229136),
            Some("https://t.me/c/1374348669/2229136".to_string())
        );
    }

    #[test]
    fn rejects_non_supergroup_chat_ids_for_links() {
        assert_eq!(normalize_supergroup_chat_id_for_link(-4679676827), None);
        assert_eq!(build_message_link(-4679676827, 42), None);
    }

    #[test]
    fn builds_private_chat_deep_links_for_user_chats() {
        assert_eq!(
            build_message_link(351987360, 42),
            Some("tg://openmessage?user_id=351987360&message_id=42".to_string())
        );
    }
}
