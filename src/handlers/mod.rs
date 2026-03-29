pub mod access;
pub mod codex_admin;
pub mod commands;
pub mod content;
pub mod media;
pub mod qa;
pub mod responses;

use std::collections::HashMap;

/// Build a mapping from `user_id` to a unique display label.
///
/// When every display name in the batch is already unique no suffix is added.
/// When two or more *different* `user_id` values share the same display name,
/// a stable ordinal suffix `(1)`, `(2)`, … is appended so that the LLM (and
/// any reader) can tell them apart.
///
/// The ordinal is assigned in ascending `user_id` order so labels are
/// deterministic across calls with the same data.
///
/// # Arguments
/// * `entries` – iterator of `(user_id, display_name)` pairs.  When a
///   `user_id` appears more than once the *last* display name wins (display
///   names may change over time; the most recent one is the best label).
pub fn build_display_label_map<'a>(
    entries: impl IntoIterator<Item = (i64, &'a str)>,
) -> HashMap<i64, String> {
    // Collect all unique (user_id → display_name) pairs, keeping the last
    // display name seen for each user_id.
    let mut uid_to_name: HashMap<i64, &str> = HashMap::new();
    for (uid, name) in entries {
        uid_to_name.insert(uid, name);
    }

    // Group user_ids by their display name.
    let mut name_to_uids: HashMap<&str, Vec<i64>> = HashMap::new();
    for (&uid, &name) in &uid_to_name {
        name_to_uids.entry(name).or_default().push(uid);
    }
    // Sort each group by user_id for deterministic ordinals.
    for uids in name_to_uids.values_mut() {
        uids.sort_unstable();
    }

    // Build the final label map.
    let mut label_map: HashMap<i64, String> = HashMap::new();
    for (&uid, &name) in &uid_to_name {
        let uids = &name_to_uids[name];
        if uids.len() > 1 {
            // Safe: uid is guaranteed to be in uids because we built uids from uid_to_name.
            let ordinal = uids
                .iter()
                .position(|id| *id == uid)
                .expect("uid must exist in its own group")
                + 1;
            label_map.insert(uid, format!("{} ({})", name, ordinal));
        } else {
            label_map.insert(uid, name.to_string());
        }
    }
    label_map
}

pub fn format_tldr_chat_content(messages: &[crate::db::models::MessageRow]) -> String {
    let label_map = build_display_label_map(messages.iter().filter_map(|m| {
        m.user_id
            .map(|uid| (uid, m.username.as_deref().unwrap_or("Anonymous")))
    }));

    let mut chat_content = String::new();
    for msg in messages {
        let timestamp = msg.date.format("%Y-%m-%d %H:%M:%S");
        let username = msg
            .user_id
            .and_then(|uid| label_map.get(&uid).cloned())
            // Fallback for messages without a user_id (e.g. channel posts).
            .unwrap_or_else(|| {
                msg.username
                    .clone()
                    .unwrap_or_else(|| "Anonymous".to_string())
            });
        let text = msg.text.as_deref().unwrap_or_default();
        let reply_context = msg
            .reply_to_message_id
            .map(|reply_to| format!(" reply_to_message_id={reply_to}"))
            .unwrap_or_default();

        chat_content.push_str(&format!(
            "{} [message_id={}{}] {}: {}\n",
            timestamp, msg.message_id, reply_context, username, text
        ));
    }

    chat_content
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{TimeZone, Utc};

    use crate::db::models::MessageRow;

    #[test]
    fn unique_names_are_unchanged() {
        let entries = vec![(1, "Alice"), (2, "Bob"), (3, "Carol")];
        let map = build_display_label_map(entries);
        assert_eq!(map[&1], "Alice");
        assert_eq!(map[&2], "Bob");
        assert_eq!(map[&3], "Carol");
    }

    #[test]
    fn colliding_names_get_ordinal_suffix() {
        let entries = vec![(10, "John"), (20, "John"), (30, "Alice")];
        let map = build_display_label_map(entries);
        // user_id 10 < 20, so 10 gets ordinal 1 and 20 gets ordinal 2.
        assert_eq!(map[&10], "John (1)");
        assert_eq!(map[&20], "John (2)");
        // Alice is unique — no suffix.
        assert_eq!(map[&30], "Alice");
    }

    #[test]
    fn three_way_collision() {
        let entries = vec![(300, "王小明"), (100, "王小明"), (200, "王小明")];
        let map = build_display_label_map(entries);
        // Ordinals assigned by ascending user_id: 100→1, 200→2, 300→3.
        assert_eq!(map[&100], "王小明 (1)");
        assert_eq!(map[&200], "王小明 (2)");
        assert_eq!(map[&300], "王小明 (3)");
    }

    #[test]
    fn multiple_independent_collisions() {
        let entries = vec![
            (1, "Alice"),
            (2, "Alice"),
            (3, "Bob"),
            (4, "Bob"),
            (5, "Carol"),
        ];
        let map = build_display_label_map(entries);
        assert_eq!(map[&1], "Alice (1)");
        assert_eq!(map[&2], "Alice (2)");
        assert_eq!(map[&3], "Bob (1)");
        assert_eq!(map[&4], "Bob (2)");
        assert_eq!(map[&5], "Carol");
    }

    #[test]
    fn empty_input_returns_empty_map() {
        let entries: Vec<(i64, &str)> = vec![];
        let map = build_display_label_map(entries);
        assert!(map.is_empty());
    }

    #[test]
    fn single_user_no_suffix() {
        let entries = vec![(42, "Solo")];
        let map = build_display_label_map(entries);
        assert_eq!(map[&42], "Solo");
    }

    #[test]
    fn duplicate_entries_for_same_user_keeps_last_name() {
        // Simulates a user who changed their display name mid-conversation.
        let entries = vec![(1, "OldName"), (2, "Bob"), (1, "NewName")];
        let map = build_display_label_map(entries);
        // user_id 1 should use "NewName" (the last entry).
        assert_eq!(map[&1], "NewName");
        assert_eq!(map[&2], "Bob");
    }

    #[test]
    fn name_change_creates_collision() {
        // user 1 was "Alice", then changed to "Bob" — now collides with user 2.
        let entries = vec![(1, "Alice"), (2, "Bob"), (1, "Bob")];
        let map = build_display_label_map(entries);
        assert_eq!(map[&1], "Bob (1)");
        assert_eq!(map[&2], "Bob (2)");
    }

    #[test]
    fn anonymous_display_names_are_disambiguated() {
        let entries = vec![(10, "Anonymous"), (20, "Anonymous")];
        let map = build_display_label_map(entries);
        assert_eq!(map[&10], "Anonymous (1)");
        assert_eq!(map[&20], "Anonymous (2)");
    }

    #[test]
    fn ordinals_are_deterministic_regardless_of_input_order() {
        let map_a = build_display_label_map(vec![(50, "X"), (10, "X"), (30, "X")]);
        let map_b = build_display_label_map(vec![(10, "X"), (30, "X"), (50, "X")]);
        assert_eq!(map_a, map_b);
        assert_eq!(map_a[&10], "X (1)");
        assert_eq!(map_a[&30], "X (2)");
        assert_eq!(map_a[&50], "X (3)");
    }

    #[test]
    fn tldr_format_includes_message_and_reply_context() {
        let messages = vec![
            MessageRow {
                id: 1,
                message_id: 10,
                chat_id: -100,
                user_id: Some(1),
                username: Some("Alice".to_string()),
                text: Some("Root message".to_string()),
                language: Some("en".to_string()),
                date: Utc
                    .with_ymd_and_hms(2026, 3, 29, 12, 0, 0)
                    .single()
                    .unwrap(),
                reply_to_message_id: None,
                asks_ai: false,
                ai_command: None,
                is_synthetic_record: false,
            },
            MessageRow {
                id: 2,
                message_id: 11,
                chat_id: -100,
                user_id: Some(2),
                username: Some("Bob".to_string()),
                text: Some("Reply message".to_string()),
                language: Some("en".to_string()),
                date: Utc
                    .with_ymd_and_hms(2026, 3, 29, 12, 1, 0)
                    .single()
                    .unwrap(),
                reply_to_message_id: Some(10),
                asks_ai: false,
                ai_command: None,
                is_synthetic_record: false,
            },
        ];

        let content = format_tldr_chat_content(&messages);

        assert!(content.contains("2026-03-29 12:00:00 [message_id=10] Alice: Root message"));
        assert!(content.contains(
            "2026-03-29 12:01:00 [message_id=11 reply_to_message_id=10] Bob: Reply message"
        ));
    }
}
