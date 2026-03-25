use std::error::Error;

use anyhow::anyhow;
use dotenvy::dotenv;
use serde::{Deserialize, Serialize};
use teloxide::dispatching::UpdateFilterExt;
use teloxide::prelude::*;
use teloxide::types::BotCommand;
use teloxide::utils::command::BotCommands;
use tracing::{error, info, warn};

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
use utils::http::get_http_client;
use utils::logging::init_logging;

#[derive(BotCommands, Clone)]
#[command(rename_rule = "lowercase")]
enum Command {
    #[command(description = "介绍AI小喵和可用指令")]
    Start,
    #[command(description = "查看帮助与指令说明")]
    Help,
    #[command(description = "汇总最近 N 条消息（默认 100 条，可用 /tldr 50 指定数量）")]
    Tldr(String),
    #[command(
        description = "回复一条文字/图片/视频/音频消息进行事实核查，支持消息内的 Telegraph/Twitter/YouTube 链接"
    )]
    Factcheck(String),
    #[command(
        description = "提问或分析媒体，弹出模型选择（默认 Gemini，自动隐藏不支持当前媒体的模型）"
    )]
    Q(String),
    #[command(description = "询问本群聊里的历史内容，可检索当前聊天记录并在需要时联网搜索")]
    Qc(String),
    #[command(
        description = "Quick Question（快问快答），小喵会用Gemini的低思考级别尽量快捷地回答你的问题"
    )]
    Qq(String),
    #[command(description = "搜索本群聊相关消息，返回命中的消息摘要和直达链接")]
    S(String),
    #[command(
        description = "用 Gemini（或已配置的 Vertex）生成/编辑图片，可直接描述或回复图片/贴纸"
    )]
    Img(String),
    #[command(description = "与 /img 相同，但附带分辨率与长宽比按钮")]
    Image(String),
    #[command(description = "用 Veo 生成视频")]
    Vid(String),
    #[command(description = "基于你在本群的聊天记录生成你的主题歌")]
    Mysong(String),
    #[command(description = "基于你在本群的聊天记录生成个人简介")]
    Profileme(String),
    #[command(description = "基于你在本群的聊天记录生成艺术形象")]
    Paintme,
    #[command(description = "基于你在本群的聊天记录生成肖像")]
    Portraitme,
    #[command(description = "查看机器人状态（管理员）")]
    Status,
    #[command(description = "查看诊断信息（管理员）")]
    Diagnose,
    #[command(description = "投喂AI小喵")]
    Support,
}

type HandlerResult = Result<(), Box<dyn Error + Send + Sync>>;

#[derive(Debug, Deserialize)]
struct TelegramApiResponse<T> {
    ok: bool,
    result: Option<T>,
    description: Option<String>,
    error_code: Option<u16>,
}

#[derive(Debug, Serialize)]
struct SetMyCommandsRequest<'a> {
    commands: &'a [BotCommand],
}

async fn publish_bot_commands(bot_token: &str, commands: &[BotCommand]) -> anyhow::Result<()> {
    let url = format!("https://api.telegram.org/bot{bot_token}/setMyCommands");
    let response = get_http_client()
        .post(url)
        .json(&SetMyCommandsRequest { commands })
        .send()
        .await
        .map_err(|_| anyhow!("Telegram setMyCommands HTTP request failed"))?;
    let status = response.status();
    let payload: TelegramApiResponse<bool> = response
        .json()
        .await
        .map_err(|_| anyhow!("Telegram setMyCommands response decode failed"))?;

    if !status.is_success() || !payload.ok || payload.result != Some(true) {
        let error_code = payload.error_code.unwrap_or(status.as_u16());
        let description = payload
            .description
            .unwrap_or_else(|| "unknown Telegram API error".to_string());
        return Err(anyhow!(
            "Telegram setMyCommands failed (status {}): {}",
            error_code,
            description
        ));
    }

    let published_commands = fetch_bot_commands(bot_token).await?;
    if published_commands.len() != commands.len() {
        warn!(
            "Telegram reported {} published default-scope command(s) after setMyCommands; expected {}",
            published_commands.len(),
            commands.len()
        );
    } else {
        info!(
            "Telegram now reports {} published default-scope command(s)",
            published_commands.len()
        );
    }

    Ok(())
}

async fn fetch_bot_commands(bot_token: &str) -> anyhow::Result<Vec<BotCommand>> {
    let url = format!("https://api.telegram.org/bot{bot_token}/getMyCommands");
    let response = get_http_client()
        .get(url)
        .send()
        .await
        .map_err(|_| anyhow!("Telegram getMyCommands HTTP request failed"))?;
    let status = response.status();
    let payload: TelegramApiResponse<Vec<BotCommand>> = response
        .json()
        .await
        .map_err(|_| anyhow!("Telegram getMyCommands response decode failed"))?;

    if !status.is_success() || !payload.ok {
        let error_code = payload.error_code.unwrap_or(status.as_u16());
        let description = payload
            .description
            .unwrap_or_else(|| "unknown Telegram API error".to_string());
        return Err(anyhow!(
            "Telegram getMyCommands failed (status {}): {}",
            error_code,
            description
        ));
    }

    Ok(payload.result.unwrap_or_default())
}

fn public_bot_commands() -> Vec<BotCommand> {
    vec![
        BotCommand::new("start", "介绍AI小喵和可用指令"),
        BotCommand::new("help", "查看帮助与指令说明"),
        BotCommand::new(
            "tldr",
            "汇总最近 N 条消息（默认 100 条，可用 /tldr 50 指定数量）",
        ),
        BotCommand::new(
            "factcheck",
            "回复一条文字/图片/视频/音频消息进行事实核查，支持消息内的 Telegraph/Twitter/YouTube 链接",
        ),
        BotCommand::new(
            "q",
            "提问或分析媒体，弹出模型选择（默认 Gemini，自动隐藏不支持当前媒体的模型）",
        ),
        BotCommand::new(
            "qq",
            "Quick Question（快问快答），小喵会用Gemini的低思考级别尽量快捷地回答你的问题",
        ),
        BotCommand::new(
            "qc",
            "询问本群聊里的历史内容，可检索当前聊天记录并在需要时联网搜索",
        ),
        BotCommand::new("s", "搜索本群聊相关消息，返回命中的消息摘要和直达链接"),
        BotCommand::new(
            "img",
            "用 Gemini（或已配置的 Vertex）生成/编辑图片，可直接描述或回复图片/贴纸",
        ),
        BotCommand::new("image", "与 /img 相同，但附带分辨率与长宽比按钮"),
        BotCommand::new("vid", "用 Veo 生成视频"),
        BotCommand::new("profileme", "基于你在本群的聊天记录生成个人简介"),
        BotCommand::new("paintme", "基于你在本群的聊天记录生成艺术形象"),
        BotCommand::new("portraitme", "基于你在本群的聊天记录生成肖像"),
        BotCommand::new("mysong", "基于你在本群的聊天记录生成你的主题歌"),
        BotCommand::new("support", "投喂AI小喵"),
    ]
}

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
    if CONFIG.publish_bot_commands {
        let commands = public_bot_commands();
        info!(
            "Publishing {} bot commands to Telegram because PUBLISH_BOT_COMMANDS=true; \
             this replaces the default-scope command list managed by BotFather",
            commands.len()
        );
        if let Err(err) = publish_bot_commands(&CONFIG.bot_token, &commands).await {
            warn!("Failed to publish bot command descriptions: {err:#}");
        }
    } else {
        info!(
            "Skipping Telegram command publishing; leave PUBLISH_BOT_COMMANDS=false \
             when BotFather manages the command list"
        );
    }

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
        Command::Mysong(arg) => {
            let bot = bot.clone();
            let state = state.clone();
            let message = message.clone();
            let arg = optional_arg(arg);
            tokio::spawn(async move {
                if let Err(err) = commands::mysong_handler(bot, state, message, arg).await {
                    error!("mysong handler failed: {err}");
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
