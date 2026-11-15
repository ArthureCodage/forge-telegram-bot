import asyncio
import logging
import sys
from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.types import BotCommand
from bot.config import settings
from bot.handlers import main_router
from bot.keyboards.inline import get_post_generation_keyboard
from bot.middlewares.auth import AuthMiddleware
from bot.services.forge import forge_service
from bot.services.ollama import ollama_service
from bot.services.queue import gpu_queue

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("forge-tg-bot")


async def set_bot_commands(bot: Bot):
    commands = [
        BotCommand(command="settings", description="🎛️ Studio & Réglages"),
        BotCommand(command="models", description="🧠 Checkpoints SD"),
        BotCommand(command="vae", description="🔮 Sélecteur de VAE"),
        BotCommand(command="sampler", description="🎯 Méthode d'Échantillonnage"),
        BotCommand(command="scheduler", description="📈 Type de Schedule"),
        BotCommand(command="steps", description="🔢 Nombre de Steps"),
        BotCommand(command="cfg", description="🎚️ CFG Scale"),
        BotCommand(command="size", description="📐 Grandeur / Dimensions"),
        BotCommand(command="batch", description="📦 Batch Count & Size"),
        BotCommand(command="loras", description="🧬 Gestionnaire LoRA"),
        BotCommand(command="presets", description="⚡ Presets d'Architecture"),
        BotCommand(command="vram", description="📊 Moniteur GPU 4090"),
    ]
    try:
        await bot.set_my_commands(commands)
    except Exception as e:
        logger.warning(f"Failed to set bot commands: {e}")


async def check_backends():
    logger.info("Checking backend services connectivity...")
    try:
        models = await forge_service.get_models()
        if models:
            logger.info(f"✅ Forge Neo API connected! Found {len(models)} checkpoints.")
        else:
            logger.warning("⚠️ Forge Neo API returned no models.")
    except Exception as e:
        logger.error(f"❌ Forge Neo API unreachable at {settings.FORGE_API_URL}: {e}")

    try:
        vaes = await forge_service.get_vaes()
        logger.info(f"✅ Available VAEs detected: {vaes}")
    except Exception as e:
        logger.warning(f"Failed detecting VAEs: {e}")

    try:
        ollama_models = await ollama_service.list_models()
        if ollama_models:
            logger.info(f"✅ Ollama API connected! Found {len(ollama_models)} models: {ollama_models}")
        else:
            logger.warning("⚠️ Ollama API connected but returned no models.")
    except Exception as e:
        logger.error(f"❌ Ollama API unreachable at {settings.OLLAMA_API_URL}: {e}")


async def main():
    logger.info("Initializing Forge Telegram Bot...")
    bot = Bot(
        token=settings.TELEGRAM_BOT_TOKEN,
        default=DefaultBotProperties(parse_mode=ParseMode.HTML)
    )
    dp = Dispatcher()

    dp.message.middleware(AuthMiddleware())
    dp.callback_query.middleware(AuthMiddleware())

    dp.include_router(main_router)

    gpu_queue.start(bot, get_post_generation_keyboard)

    await check_backends()
    await set_bot_commands(bot)

    logger.info(f"Bot started in Long Polling mode! Whitelist: {settings.allowed_user_ids_list}")
    try:
        await dp.start_polling(bot, drop_pending_updates=True)
    finally:
        await bot.session.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        logger.info("Bot stopped gracefully.")
