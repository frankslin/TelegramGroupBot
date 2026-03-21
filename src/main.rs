use std::error::Error;

use dotenvy::dotenv;
use teloxide::dispatching::UpdateFilterExt;
use teloxide::prelude::*;
use teloxide::utils::command::BotCommands;
use tracing::{error, info};

mod config;
mod db;
mod handlers;
mod llm;
mod state;
mod tools;
mod utils;

use config::CONFIG;
use db::database::Database;
use handlers::qa::MODEL_CALLBACK_PREFIX;
use handlers::{commands, qa};
use state::AppState;
use utils::logging::init_logging;

#[derive(BotCommands, Clone)]
#[command(rename_rule = "lowercase")]
enum Command {
    Start,
    Help,
    Tldr(String),
    Factcheck(String),
    Q(String),
    Qc(String),
    Qq(String),
    S(String),
    Img(String),
    Image(String),
    Vid(String),
    Profileme(String),
    Paintme,
    Portraitme,
    Status,
    Diagnose,
    Support,
}

type HandlerResult = Result<(), Box<dyn Error + Send + Sync>>;

#[tokio::main]
async fn main() -> HandlerResult {
    dotenv().ok();
    let _guards = init_logging();

    let bot = Bot::new(CONFIG.bot_token.clone());
    let me = bot.get_me().await?;
    let bot_user_id = i64::try_from(me.id.0).unwrap_or_default();
    let bot_username_lower = me
        .username
        .as_ref()
        .map(|username| username.to_lowercase())
        .unwrap_or_default();
    info!("Starting TelegramGroupHelperBot (Rust)");

    let db = Database::init(&CONFIG.database_url).await?;
    let state = AppState::new(db, bot_user_id, bot_username_lower);

    handlers::access::load_whitelist();

    let command_handler = dptree::entry()
        .filter_command::<Command>()
        .endpoint(handle_command);

    let message_handler = Update::filter_message()
        .branch(command_handler)
        .branch(
            dptree::filter(|msg: Message| msg.media_group_id().is_some())
                .endpoint(handle_media_group),
        )
        .branch(
            dptree::filter(|msg: Message| msg.text().is_some() || msg.caption().is_some())
                .endpoint(handle_text_message),
        )
        .endpoint(ignore_message);

    let callback_state = state.clone();
    let callback_handler =
        Update::filter_callback_query().endpoint(move |bot: Bot, query: CallbackQuery| {
            let state = callback_state.clone();
            async move { handle_callback_query(bot, state, query).await }
        });

    let handler = dptree::entry()
        .branch(message_handler)
        .branch(callback_handler);

    Dispatcher::builder(bot, handler)
        .dependencies(dptree::deps![state])
        .enable_ctrlc_handler()
        .build()
        .dispatch()
        .await;

    Ok(())
}

async fn handle_command(
    bot: Bot,
    state: AppState,
    message: Message,
    command: Command,
) -> HandlerResult {
    fn optional_arg(arg: String) -> Option<String> {
        if arg.trim().is_empty() {
            None
        } else {
            Some(arg)
        }
    }

    match command {
        Command::Start => commands::start_handler(bot, message).await?,
        Command::Help => commands::help_handler(bot, message).await?,
        Command::Tldr(arg) => {
            let bot = bot.clone();
            let state = state.clone();
            let message = message.clone();
            let arg = optional_arg(arg);
            tokio::spawn(async move {
                if let Err(err) = commands::tldr_handler(bot, state, message, arg).await {
                    error!("tldr handler failed: {err}");
                }
            });
        }
        Command::Factcheck(arg) => {
            let bot = bot.clone();
            let state = state.clone();
            let message = message.clone();
            let arg = optional_arg(arg);
            tokio::spawn(async move {
                if let Err(err) = commands::factcheck_handler(bot, state, message, arg).await {
                    error!("factcheck handler failed: {err}");
                }
            });
        }
        Command::Q(arg) => {
            let bot = bot.clone();
            let state = state.clone();
            let message = message.clone();
            let arg = optional_arg(arg);
            tokio::spawn(async move {
                if let Err(err) = qa::q_handler(bot, state, message, arg, false, "q").await {
                    error!("q handler failed: {err}");
                }
            });
        }
        Command::Qc(arg) => {
            let bot = bot.clone();
            let state = state.clone();
            let message = message.clone();
            let arg = optional_arg(arg);
            tokio::spawn(async move {
                if let Err(err) = qa::qc_handler(bot, state, message, arg).await {
                    error!("qc handler failed: {err}");
                }
            });
        }
        Command::Qq(arg) => {
            let bot = bot.clone();
            let state = state.clone();
            let message = message.clone();
            let arg = optional_arg(arg);
            tokio::spawn(async move {
                if let Err(err) = qa::qq_handler(bot, state, message, arg).await {
                    error!("qq handler failed: {err}");
                }
            });
        }
        Command::S(arg) => {
            let bot = bot.clone();
            let state = state.clone();
            let message = message.clone();
            let arg = optional_arg(arg);
            tokio::spawn(async move {
                if let Err(err) = qa::s_handler(bot, state, message, arg).await {
                    error!("s handler failed: {err}");
                }
            });
        }
        Command::Img(arg) => {
            let bot = bot.clone();
            let state = state.clone();
            let message = message.clone();
            let arg = optional_arg(arg);
            tokio::spawn(async move {
                if let Err(err) = commands::img_handler(bot, state, message, arg).await {
                    error!("img handler failed: {err}");
                }
            });
        }
        Command::Image(arg) => {
            let bot = bot.clone();
            let state = state.clone();
            let message = message.clone();
            let arg = optional_arg(arg);
            tokio::spawn(async move {
                if let Err(err) = commands::image_handler(bot, state, message, arg).await {
                    error!("image handler failed: {err}");
                }
            });
        }
        Command::Vid(arg) => {
            let bot = bot.clone();
            let message = message.clone();
            let arg = optional_arg(arg);
            tokio::spawn(async move {
                if let Err(err) = commands::vid_handler(bot, message, arg).await {
                    error!("vid handler failed: {err}");
                }
            });
        }
        Command::Profileme(arg) => {
            let bot = bot.clone();
            let state = state.clone();
            let message = message.clone();
            let arg = optional_arg(arg);
            tokio::spawn(async move {
                if let Err(err) = commands::profileme_handler(bot, state, message, arg).await {
                    error!("profileme handler failed: {err}");
                }
            });
        }
        Command::Paintme => {
            let bot = bot.clone();
            let state = state.clone();
            let message = message.clone();
            tokio::spawn(async move {
                if let Err(err) = commands::paintme_handler(bot, state, message, false).await {
                    error!("paintme handler failed: {err}");
                }
            });
        }
        Command::Portraitme => {
            let bot = bot.clone();
            let state = state.clone();
            let message = message.clone();
            tokio::spawn(async move {
                if let Err(err) = commands::paintme_handler(bot, state, message, true).await {
                    error!("portraitme handler failed: {err}");
                }
            });
        }
        Command::Status => {
            let bot = bot.clone();
            let state = state.clone();
            let message = message.clone();
            tokio::spawn(async move {
                if let Err(err) = commands::status_handler(bot, state, message).await {
                    error!("status handler failed: {err}");
                }
            });
        }
        Command::Diagnose => {
            let bot = bot.clone();
            let state = state.clone();
            let message = message.clone();
            tokio::spawn(async move {
                if let Err(err) = commands::diagnose_handler(bot, state, message).await {
                    error!("diagnose handler failed: {err}");
                }
            });
        }
        Command::Support => commands::support_handler(bot, message).await?,
    }
    Ok(())
}

async fn handle_callback_query(bot: Bot, state: AppState, query: CallbackQuery) -> HandlerResult {
    let Some(data) = query.data.clone() else {
        return Ok(());
    };
    if data.starts_with(MODEL_CALLBACK_PREFIX) {
        let bot = bot.clone();
        let state = state.clone();
        tokio::spawn(async move {
            if let Err(err) = qa::model_selection_callback(bot, state, query).await {
                error!("model selection callback failed: {err}");
            }
        });
        return Ok(());
    }
    if data.starts_with("image_res:") || data.starts_with("image_aspect:") {
        let bot = bot.clone();
        let state = state.clone();
        tokio::spawn(async move {
            if let Err(err) = commands::image_selection_callback(bot, state, query).await {
                error!("image selection callback failed: {err}");
            }
        });
    }
    Ok(())
}

async fn handle_media_group(state: AppState, message: Message) -> HandlerResult {
    commands::handle_media_group(state, message).await;
    Ok(())
}

async fn handle_text_message(bot: Bot, state: AppState, message: Message) -> HandlerResult {
    if let Some(text) = message.text().or_else(|| message.caption()) {
        if text.trim_start().starts_with('/') {
            return Ok(());
        }
    }

    if qa::should_auto_q_trigger(&message, state.bot_user_id, &state.bot_username_lower) {
        let query = qa::build_auto_q_query(&message, state.bot_user_id, &state.bot_username_lower);
        let bot = bot.clone();
        let state = state.clone();
        let message = message.clone();
        tokio::spawn(async move {
            if let Err(err) = qa::q_handler(bot, state, message, query, false, "q").await {
                error!("auto q handler failed: {err}");
            }
        });
        return Ok(());
    }

    handlers::responses::log_message(&state, &message).await;
    Ok(())
}

async fn ignore_message(_message: Message) -> HandlerResult {
    Ok(())
}
