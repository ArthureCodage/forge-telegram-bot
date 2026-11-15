import asyncio
import base64
import io
import logging
import re
from aiogram import Bot, F, Router
from aiogram.types import Message
from bot.keyboards.inline import get_img2img_strength_keyboard, get_settings_keyboard
from bot.services.forge import forge_service
from bot.services.ollama import ollama_service
from bot.services.queue import GenerationJob, gpu_queue
from bot.services.state import PRESETS, state_manager
from bot.services.vram_guard import vram_guard

logger = logging.getLogger(__name__)
router = Router()


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
            f"✅ <b>Prompt négatif mis à jour !</b>\n\n<code>{text}</code>",
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
            return
        else:
            await message.answer("❌ Veuillez entrer un nombre entier valide entre 1 et 150 (ex: 28).")
            return

    # 3. Custom CFG Scale
    if waiting_for == "custom_cfg":
        try:
            val = float(text.replace(",", "."))
            if 0.5 <= val <= 30.0:
                val = round(val, 1)
                state_manager.update_user_state(user_id, cfg_scale=val, waiting_for=None)
                await message.answer(
                    f"✅ <b>CFG Scale réglé à {val}</b>",
                    reply_markup=get_settings_keyboard(user_state)
                )
                return
            else:
                await message.answer("❌ Veuillez entrer un nombre entre 0.5 et 30.0 (ex: 6.0).")
                return
        except ValueError:
            await message.answer("❌ Format invalide. Entrez un nombre décimal (ex: 6.5).")
            return

    # 4. Custom Dimensions (Grandeur)
    if waiting_for == "custom_dimensions":
        match = re.search(r"(\d{3,4})[\s*xX,;:]+(\d{3,4})", text)
        if match:
            w = int(match.group(1))
            h = int(match.group(2))
            # Round to multiple of 8
            w = max(256, min(2048, (w // 8) * 8))
            h = max(256, min(2048, (h // 8) * 8))
            state_manager.update_user_state(user_id, width=w, height=h, waiting_for=None)
            await message.answer(
                f"✅ <b>Dimensions d'image réglées à {w}x{h}</b>",
                reply_markup=get_settings_keyboard(user_state)
            )
            return
        else:
            await message.answer("❌ Format invalide. Entrez vos dimensions sous la forme <code>896x1152</code> ou <code>1024 1024</code>.")
            return

    # 5. Pending img2img custom prompt
    pending_img2img = user_state.get("pending_img2img")
    if pending_img2img and waiting_for == "img2img_prompt":
        pending_img2img["prompt"] = text
        state_manager.update_user_state(user_id, pending_img2img=pending_img2img, waiting_for=None)
        await message.answer(
            f"🎯 <b>Prompt img2img défini :</b> <i>{text}</i>\n\n"
            f"Choisissez maintenant l'intensité de transformation (Denoising Strength) :",
            reply_markup=get_img2img_strength_keyboard()
        )
        return

    # ─── STANDARD GENERATION FLOW ───────────────────────────────────────────
    raw_prompt = text
    preset_key = user_state.get("preset", "sdxl")
    preset = PRESETS.get(preset_key, PRESETS["sdxl"])
    magic_on = user_state.get("magic_prompt", False)

    # Active Generation Parameters
    width = user_state.get("width", 1024)
    height = user_state.get("height", 1024)
    sampler = user_state.get("sampler", preset["sampler"])
    scheduler = user_state.get("scheduler", preset["scheduler"])
    steps = user_state.get("steps", preset["steps"])
    cfg_scale = user_state.get("cfg_scale", preset["cfg_scale"])
    batch_count = user_state.get("batch_count", 1)
    batch_size = user_state.get("batch_size", 1)

    negative_prompt = user_state.get("negative_prompt", "")
    active_loras = user_state.get("active_loras", {})

    status_msg = await message.answer("⏳ <i>Préparation de la requête...</i>")

    # Step 1: Magic Prompt Enhancement (Ollama) if enabled
    final_prompt = raw_prompt
    if magic_on:
        model_name = user_state.get("ollama_model", "huihui_ai/qwen3-abliterated:14b")
        short_model = model_name.split("/")[-1]
        try:
            await status_msg.edit_text(f"✨ <i>Optimisation du prompt via Ollama ({short_model})...</i>")
            enhanced = await ollama_service.enhance_prompt(
                prompt=raw_prompt,
                mode=preset.get("mode", "tag_based"),
                model=model_name
            )
            final_prompt = enhanced
            await vram_guard.unload_ollama()
        except Exception as e:
            logger.error(f"Error in magic prompt: {e}")
            final_prompt = raw_prompt

    # Step 2: Append LoRAs
    forge_prompt = forge_service.build_prompt_with_loras(final_prompt, active_loras)

    # Step 3: Build Forge payload
    payload = {
        "prompt": forge_prompt,
        "negative_prompt": negative_prompt,
        "steps": steps,
        "sampler_name": sampler,
        "scheduler": scheduler,
        "cfg_scale": cfg_scale,
        "width": width,
        "height": height,
        "batch_size": batch_size,
        "n_iter": batch_count,
        "seed": -1,
        "save_images": True,
    }

    # Step 4: Enqueue
    job = GenerationJob(
        job_type="txt2img",
        user_id=user_id,
        chat_id=message.chat.id,
        prompt=final_prompt,
        negative_prompt=negative_prompt,
        payload=payload,
        status_message=status_msg,
        future=asyncio.get_running_loop().create_future(),
        extra_meta={
            "preset": preset_key,
            "raw_prompt": raw_prompt,
        }
    )

    pos = await gpu_queue.enqueue(job)
    total_imgs = batch_count * batch_size
    batch_text = f" ({total_imgs} image{'s' if total_imgs > 1 else ''})" if total_imgs > 1 else ""
    await status_msg.edit_text(
        f"⏳ <b>En file d'attente (Position #{pos}){batch_text}...</b>\n"
        f"Rendu GPU RTX 4090 en cours de démarrage."
    )


@router.message(F.photo)
async def handle_photo_upload(message: Message, bot: Bot):
    """Handles incoming photos for mobile img2img workflows."""
    user_id = message.from_user.id

    photo = message.photo[-1]
    file_info = await bot.get_file(photo.file_id)
    file_bytes_io = io.BytesIO()
    await bot.download_file(file_info.file_path, destination=file_bytes_io)
    img_b64 = base64.b64encode(file_bytes_io.getvalue()).decode("utf-8")

    caption = (message.caption or "").strip()

    pending_data = {
        "image_b64": img_b64,
        "prompt": caption or "high quality, detailed masterpiece",
        "has_custom_prompt": bool(caption),
    }

    state_manager.update_user_state(
        user_id,
        pending_img2img=pending_data,
        waiting_for=None if caption else "img2img_prompt"
    )

    if caption:
        text = (
            f"🖼️ <b>Photo reçue pour img2img !</b>\n\n"
            f"📝 <b>Prompt :</b> <i>{caption}</i>\n\n"
            f"Choisissez l'intensité de transformation (Denoising Strength) :"
        )
    else:
        text = (
            f"🖼️ <b>Photo reçue pour img2img !</b>\n\n"
            f"Envoyez votre <b>prompt textuel</b> en réponse, ou cliquez directement sur une intensité ci-dessous pour transformer l'image :"
        )

    await message.answer(text, reply_markup=get_img2img_strength_keyboard())
