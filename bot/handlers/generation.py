import logging
from html import escape
from aiogram import Bot, F, Router
from aiogram.types import Message
from bot.config import PENDING_DIR
from bot.keyboards.inline import get_img2img_strength_keyboard, get_settings_keyboard
from bot.services.payloads import build_payload, fit_dimensions, parse_dimensions
from bot.services.queue import GenerationJob, gpu_queue
from bot.services.state import PRESETS, state_manager

logger = logging.getLogger(__name__)
router = Router()

DEFAULT_IMG2IMG_PROMPT = "high quality, detailed masterpiece"


@router.message(F.text & ~F.text.startswith("/"))
async def handle_text_prompt(message: Message, bot: Bot):
    """Handles prompt text messages and interactive user input states."""
    user_id = message.from_user.id
    user_state = state_manager.get_user_state(user_id)
    text = message.text.strip()
    waiting_for = user_state.get("waiting_for")

    # 1. Custom Negative Prompt
    if waiting_for == "negative_prompt":
        state_manager.update_user_state(user_id, negative_prompt=text, waiting_for=None)
        await message.answer(
            f"✅ <b>Prompt négatif mis à jour !</b>\n\n<code>{escape(text)}</code>",
            reply_markup=get_settings_keyboard(user_state)
        )
        return

    # 2. Custom Steps
    if waiting_for == "custom_steps":
        if text.isdigit() and 1 <= int(text) <= 150:
            val = int(text)
            state_manager.update_user_state(user_id, steps=val, waiting_for=None)
            await message.answer(
                f"✅ <b>Steps réglé à {val}</b>",
                reply_markup=get_settings_keyboard(user_state)
            )
        else:
            await message.answer(
                "❌ Veuillez entrer un nombre entier valide entre 1 et 150 (ex: 28), ou /cancel."
            )
        return

    # 3. Custom CFG Scale
    if waiting_for == "custom_cfg":
        try:
            val = float(text.replace(",", "."))
        except ValueError:
            await message.answer("❌ Format invalide. Entrez un nombre décimal (ex: 6.5), ou /cancel.")
            return
        if 0.5 <= val <= 30.0:
            val = round(val, 1)
            state_manager.update_user_state(user_id, cfg_scale=val, waiting_for=None)
            await message.answer(
                f"✅ <b>CFG Scale réglé à {val}</b>",
                reply_markup=get_settings_keyboard(user_state)
            )
        else:
            await message.answer("❌ Veuillez entrer un nombre entre 0.5 et 30.0 (ex: 6.0), ou /cancel.")
        return

    # 4. Custom Dimensions (Grandeur)
    if waiting_for == "custom_dimensions":
        dims = parse_dimensions(text)
        if dims:
            w, h = dims
            state_manager.update_user_state(user_id, width=w, height=h, waiting_for=None)
            await message.answer(
                f"✅ <b>Dimensions d'image réglées à {w}x{h}</b>",
                reply_markup=get_settings_keyboard(user_state)
            )
        else:
            await message.answer(
                "❌ Format invalide. Entrez vos dimensions sous la forme "
                "<code>896x1152</code> ou <code>1024 1024</code>, ou /cancel."
            )
        return

    # 5. Pending img2img custom prompt
    pending_img2img = user_state.get("pending_img2img")
    if pending_img2img and waiting_for == "img2img_prompt":
        pending_img2img["prompt"] = text
        state_manager.update_user_state(user_id, pending_img2img=pending_img2img, waiting_for=None)
        await message.answer(
            f"🎯 <b>Prompt img2img défini :</b> <i>{escape(text)}</i>\n\n"
            f"Choisissez maintenant l'intensité de transformation (Denoising Strength) :",
            reply_markup=get_img2img_strength_keyboard()
        )
        return

    # ─── STANDARD GENERATION FLOW ───────────────────────────────────────────
    preset_key = user_state.get("preset", "sdxl")
    preset = PRESETS.get(preset_key, PRESETS["sdxl"])
    negative_prompt = user_state.get("negative_prompt", "")
    batch_count = user_state.get("batch_count", 1)
    batch_size = user_state.get("batch_size", 1)

    status_msg = await message.answer("⏳ <i>Préparation de la requête...</i>")

    # Magic Prompt (Ollama) runs inside the GPU queue, right before Forge.
    enhance = None
    if user_state.get("magic_prompt", False):
        enhance = {
            "mode": preset.get("mode", "tag_based"),
            "model": user_state.get("ollama_model", ""),
        }

    job = GenerationJob(
        job_type="txt2img",
        user_id=user_id,
        chat_id=message.chat.id,
        prompt=text,
        negative_prompt=negative_prompt,
        payload=build_payload(
            user_state, negative_prompt, batch_size=batch_size, batch_count=batch_count
        ),
        status_message=status_msg,
        loras=dict(user_state.get("active_loras", {})),
        enhance=enhance,
        extra_meta={"preset": preset_key, "raw_prompt": text},
    )

    pos = await gpu_queue.enqueue(job)
    total_imgs = batch_count * batch_size
    batch_text = f" ({total_imgs} images)" if total_imgs > 1 else ""
    await status_msg.edit_text(
        f"⏳ <b>En file d'attente (Position #{pos}){batch_text}...</b>\n"
        f"<i>/cancel pour annuler</i>"
    )


@router.message(F.photo)
async def handle_photo_upload(message: Message, bot: Bot):
    """Handles incoming photos for mobile img2img workflows."""
    user_id = message.from_user.id
    user_state = state_manager.get_user_state(user_id)

    # The photo is kept on disk; only its path goes into state.json.
    photo = message.photo[-1]
    image_path = PENDING_DIR / f"{user_id}.jpg"
    await bot.download(photo, destination=image_path)

    width, height = fit_dimensions(
        photo.width, photo.height,
        user_state.get("width", 1024), user_state.get("height", 1024),
    )
    caption = (message.caption or "").strip()

    state_manager.update_user_state(
        user_id,
        pending_img2img={
            "image_path": str(image_path),
            "prompt": caption or DEFAULT_IMG2IMG_PROMPT,
            "width": width,
            "height": height,
        },
        waiting_for=None if caption else "img2img_prompt"
    )

    if caption:
        text = (
            f"🖼️ <b>Photo reçue pour img2img !</b> ({width}x{height})\n\n"
            f"📝 <b>Prompt :</b> <i>{escape(caption)}</i>\n\n"
            f"Choisissez l'intensité de transformation (Denoising Strength) :"
        )
    else:
        text = (
            f"🖼️ <b>Photo reçue pour img2img !</b> ({width}x{height})\n\n"
            f"Envoyez votre <b>prompt textuel</b> en réponse, ou cliquez directement sur "
            f"une intensité ci-dessous pour transformer l'image :"
        )

    await message.answer(text, reply_markup=get_img2img_strength_keyboard())
