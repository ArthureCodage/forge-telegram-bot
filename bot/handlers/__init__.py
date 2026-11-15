from aiogram import Router
from .commands import router as commands_router
from .generation import router as generation_router
from .callbacks import router as callbacks_router

main_router = Router()
main_router.include_router(commands_router)
main_router.include_router(callbacks_router)
main_router.include_router(generation_router)

__all__ = ["main_router"]
