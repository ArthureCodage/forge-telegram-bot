import logging
from html import escape
from aiogram import Router
from aiogram.filters import Command
from aiogram.types import Message
from bot.keyboards.inline import (
    get_batch_keyboard,
    get_cfg_keyboard,
    get_dimensions_keyboard,
    get_loras_keyboard,
    get_models_keyboard,
    get_samplers_keyboard,
    get_schedulers_keyboard,
    get_settings_keyboard,
    get_steps_keyboard,
    get_vaes_keyboard,
    get_vram_keyboard,
)
from bot.services.forge import forge_service
from bot.services.queue import gpu_queue
from bot.services.state import state_manager
from bot.services.vram_guard import vram_guard

logger = logging.getLogger(__name__)
router = Router()


@router.message(Command("start", "settings"))
async def cmd_settings(message: Message):
    """Opens the main Control Center dashboard."""
    user_id = message.from_user.id
    user_state = state_manager.get_user_state(user_id)

    cur_model = await forge_service.get_current_model()
    cur_vae = await forge_service.get_current_vae()
    if cur_model:
        user_state["model"] = cur_model
    if cur_vae:
        user_state["vae"] = cur_vae

    text = (
        "🎛️ <b>Studio de Contrôle &amp; Génération</b>\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "Configurez l'ensemble de vos paramètres de rendu par simple clic tactile :"
    )
    await message.answer(
        text,
        reply_markup=get_settings_keyboard(user_state, active_model_name=cur_model, active_vae_name=cur_vae)
    )


@router.message(Command("models"))
async def cmd_models(message: Message):
    """Displays paginated list of checkpoints."""
    msg = await message.answer("🔄 <i>Interrogation des modèles Forge Neo...</i>")
    models = await forge_service.get_models()
    if not models:
        await msg.edit_text("❌ Impossible de récupérer la liste des modèles depuis Forge Neo.")
        return
    current = await forge_service.get_current_model()
    text = (
        f"🧠 <b>Gestionnaire de Checkpoints</b>\n"
        f"Modèle actif : <code>{escape(current or 'Inconnu')}</code>\n\n"
        f"Cliquez sur un modèle pour le charger dans la VRAM :"
    )
    await msg.edit_text(text, reply_markup=get_models_keyboard(models, current, page=0))


@router.message(Command("vae"))
async def cmd_vae(message: Message):
    """Displays list of VAEs."""
    vaes = await forge_service.get_vaes()
    cur_vae = await forge_service.get_current_vae()
    text = (
        f"🔮 <b>Gestionnaire de VAE</b>\n"
        f"VAE actif : <code>{escape(str(cur_vae))}</code>\n\n"
        f"Sélectionnez le VAE souhaité :"
    )
    await message.answer(text, reply_markup=get_vaes_keyboard(vaes, cur_vae, page=0))


@router.message(Command("sampler"))
async def cmd_sampler(message: Message):
    """Displays list of samplers."""
    user_id = message.from_user.id
    user_state = state_manager.get_user_state(user_id)
    samplers = await forge_service.get_samplers()
    cur_sampler = user_state.get("sampler", "Euler a")
    text = f"🎯 <b>Méthode d'Échantillonnage (Sampling Method) :</b>\nActuel : <code>{escape(str(cur_sampler))}</code>"
    await message.answer(text, reply_markup=get_samplers_keyboard(samplers, cur_sampler, page=0))


@router.message(Command("scheduler"))
async def cmd_scheduler(message: Message):
    """Displays list of schedulers."""
    user_id = message.from_user.id
    user_state = state_manager.get_user_state(user_id)
    schedulers = await forge_service.get_schedulers()
    cur_sched = user_state.get("scheduler", "Karras")
    text = f"📈 <b>Type de Schedule (Scheduler) :</b>\nActuel : <code>{escape(str(cur_sched))}</code>"
    await message.answer(text, reply_markup=get_schedulers_keyboard(schedulers, cur_sched))


@router.message(Command("steps"))
async def cmd_steps(message: Message):
    """Steps adjustment."""
    user_id = message.from_user.id
    user_state = state_manager.get_user_state(user_id)
    cur_steps = user_state.get("steps", 28)
    text = f"🔢 <b>Nombre de Steps d'échantillonnage :</b>\nActuel : <b>{cur_steps}</b>"
    await message.answer(text, reply_markup=get_steps_keyboard(cur_steps))


@router.message(Command("cfg"))
async def cmd_cfg(message: Message):
    """CFG scale adjustment."""
    user_id = message.from_user.id
    user_state = state_manager.get_user_state(user_id)
    cur_cfg = user_state.get("cfg_scale", 6.0)
    text = f"🎚️ <b>CFG Scale (Guidance Prompt) :</b>\nActuel : <b>{cur_cfg:.1f}</b>"
    await message.answer(text, reply_markup=get_cfg_keyboard(cur_cfg))


@router.message(Command("size", "dimensions", "resolution"))
async def cmd_dimensions(message: Message):
    """Dimensions adjustment."""
    user_id = message.from_user.id
    user_state = state_manager.get_user_state(user_id)
    w = user_state.get("width", 1024)
    h = user_state.get("height", 1024)
    text = f"📐 <b>Grandeur &amp; Résolution d'image :</b>\nActuel : <b>{w}x{h}</b>"
    await message.answer(text, reply_markup=get_dimensions_keyboard(w, h))


@router.message(Command("batch"))
async def cmd_batch(message: Message):
    """Batch count and size adjustment."""
    user_id = message.from_user.id
    user_state = state_manager.get_user_state(user_id)
    b_count = user_state.get("batch_count", 1)
    b_size = user_state.get("batch_size", 1)
    text = "📦 <b>Gestion des Lots (Batch Count &amp; Batch Size) :</b>"
    await message.answer(text, reply_markup=get_batch_keyboard(b_count, b_size))


@router.message(Command("loras"))
async def cmd_loras(message: Message):
    """LoRA manager."""
    user_id = message.from_user.id
    user_state = state_manager.get_user_state(user_id)
    active_loras = user_state.get("active_loras", {})
    msg = await message.answer("🔄 <i>Chargement des LoRAs disponibles...</i>")
    loras = await forge_service.get_loras()
    if not loras:
        await msg.edit_text("❌ Aucun LoRA détecté dans Forge Neo.")
        return
    text = "🧬 <b>Gestionnaire de LoRAs :</b>\nCliquez pour activer/désactiver et ajuster les poids :"
    await msg.edit_text(text, reply_markup=get_loras_keyboard(loras, active_loras, page=0))


@router.message(Command("presets"))
async def cmd_presets(message: Message):
    """Architecture presets."""
    user_id = message.from_user.id
    user_state = state_manager.get_user_state(user_id)
    cur_model = await forge_service.get_current_model()
    cur_vae = await forge_service.get_current_vae()
    text = (
        "🎛️ <b>Presets Rapides d'Architecture :</b>\n\n"
        "⚡ <b>Flux / Krea :</b> Euler + Simple, 24 steps, CFG 1.0\n"
        "🎨 <b>SDXL / Illustrious :</b> Euler a + Karras, 28 steps, CFG 6.0\n"
        "🚀 <b>Turbo / Lightning :</b> Euler + Normal, 8 steps, CFG 1.8\n\n"
        "Sélectionnez un preset pour appliquer automatiquement les paramètres optimaux :"
    )
    await message.answer(
        text,
        reply_markup=get_settings_keyboard(user_state, active_model_name=cur_model, active_vae_name=cur_vae)
    )


@router.message(Command("vram", "free", "status"))
async def cmd_vram_status(message: Message):
    """Displays hardware monitor."""
    msg = await message.answer("🔍 <i>Lecture des sondes matérielles NVIDIA...</i>")
    gpu_info = await vram_guard.get_vram_info()
    processes = await vram_guard.get_gpu_processes()
    text = vram_guard.format_vram_status(gpu_info, processes)
    await msg.edit_text(text, reply_markup=get_vram_keyboard())


@router.message(Command("cancel", "stop"))
async def cmd_cancel(message: Message):
    """Stops the running generation, empties the queue and leaves any pending text input."""
    user_id = message.from_user.id
    had_input = state_manager.get_user_state(user_id).get("waiting_for") is not None
    if had_input:
        state_manager.update_user_state(user_id, waiting_for=None)

    dropped, running = await gpu_queue.cancel_all()

    lines = []
    if running:
        lines.append("🛑 Génération en cours interrompue.")
    if dropped:
        lines.append(f"🗑️ {dropped} tâche(s) retirée(s) de la file d'attente.")
    if had_input:
        lines.append("✏️ Saisie en attente annulée.")
    await message.answer("\n".join(lines) or "ℹ️ Rien à annuler.")
