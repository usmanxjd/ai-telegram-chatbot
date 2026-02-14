from __future__ import annotations

import logging

from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

from support_bot.config import get_settings
from support_bot.knowledge_base import KnowledgeBaseManager
from support_bot.support_chain import SupportChainFactory

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Hi! I'm your AI customer support assistant. Ask me anything about your product or service."
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Send your support question in plain text. Use /start to begin or /help to show this message."
    )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.message.text.strip()
    if not query:
        await update.message.reply_text("Please send a non-empty support question.")
        return

    chain = context.application.bot_data["support_chain"]

    try:
        result = chain.invoke(query)
        response = str(result).strip()
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to process user query")
        response = (
            "Sorry, I couldn't process that request right now. "
            "Please try again in a moment or contact a human support agent."
        )

    await update.message.reply_text(response)


def build_application() -> Application:
    settings = get_settings()

    vectorstore = None
    if settings.enable_rag:
        kb_manager = KnowledgeBaseManager(settings.knowledge_base_dir, settings.vectorstore_dir)
        vectorstore = kb_manager.build_or_load()

    support_chain = SupportChainFactory(
        hf_api_token=settings.huggingfacehub_api_token,
        repo_id=settings.hf_llm_repo_id,
        retrieval_top_k=settings.retrieval_top_k,
    ).create(vectorstore=vectorstore)

    app = Application.builder().token(settings.telegram_bot_token).build()
    app.bot_data["support_chain"] = support_chain

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    return app


def main() -> None:
    app = build_application()
    logger.info("Starting Telegram support bot polling...")
    app.run_polling(close_loop=False)


if __name__ == "__main__":
    main()
