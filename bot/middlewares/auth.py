import logging
from typing import Any, Awaitable, Callable, Dict
from aiogram import BaseMiddleware
from aiogram.types import TelegramObject, Message, CallbackQuery
from bot.config import settings

logger = logging.getLogger(__name__)


class AuthMiddleware(BaseMiddleware):
    async def __call__(
        self,
        handler: Callable[[TelegramObject, Dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: Dict[str, Any]
    ) -> Any:
        user = getattr(event, "from_user", None)
        if not user:
            return None

        allowed = settings.allowed_user_ids_list
        if user.id not in allowed:
            logger.warning(
                f"Unauthorized access blocked: user_id={user.id}, username=@{user.username}, full_name='{user.full_name}'"
            )
            # Silently drop or reject
            if isinstance(event, Message):
                await event.answer("⛔ Accès refusé. Cet identifiant n'est pas autorisé.")
            elif isinstance(event, CallbackQuery):
                await event.answer("⛔ Accès non autorisé.", show_alert=True)
            return None

        return await handler(event, data)
