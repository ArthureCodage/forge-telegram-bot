import asyncio
import base64
import logging
from html import escape
from pathlib import Path
from typing import Any, Dict, Optional
from aiogram import Bot, F, Router
from aiogram.types import CallbackQuery, FSInputFile
from bot.keyboards.inline import (
    get_batch_keyboard,
    get_cfg_keyboard,
    get_dimensions_keyboard,
    get_loras_keyboard,
    get_models_keyboard,
    get_ollama_models_keyboard,
    get_samplers_keyboard,
    get_schedulers_keyboard,
    get_settings_keyboard,
    get_steps_keyboard,
    get_vaes_keyboard,
    get_vram_keyboard,
)
from bot.services import history
from bot.services.forge import forge_service
from bot.services.ollama import ollama_service
from bot.services.payloads import build_payload
from bot.services.queue import UPSCALE_FACTOR, GenerationJob, gpu_queue
from bot.services.state import DEFAULT_USER_STATE, PRESETS, state_manager
from bot.services.vram_guard import vram_guard

logger = logging.getLogger(__name__)
router = Router()


async def refresh_settings_view(query: CallbackQuery):
    user_id = query.from_user.id
    user_state = state_manager.get_user_state(user_id)
    cur_model = await forge_service.get_current_model()
    cur_vae = await forge_service.get_current_vae()
    await query.message.edit_text(
        "🎛️ <b>Studio de Contrôle &amp; Génération</b>\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "Configurez l'ensemble de vos paramètres de rendu par simple clic tactile :",
        reply_markup=get_settings_keyboard(user_state, active_model_name=cur_model, active_vae_name=cur_vae)
    )


@router.callback_query(F.data == "back_settings")
async def cb_back_settings(query: CallbackQuery):
    await refresh_settings_view(query)
    await query.answer()


# ─── MODELS CALLBACKS ───────────────────────────────────────────────────────

@router.callback_query(F.data.startswith("menu_models:") | F.data.startswith("models_page:"))
async def cb_menu_models(query: CallbackQuery):
    page = int(query.data.split(":")[1])
    models = await forge_service.get_models()
    current = await forge_service.get_current_model()
    await query.message.edit_text(
        f"🧠 <b>Sélection du Modèle / Checkpoint</b>\nActuel : <code>{escape(str(current))}</code>",
        reply_markup=get_models_keyboard(models, current, page=page)
    )
    await query.answer()


@router.callback_query(F.data.startswith("set_model:"))
async def cb_set_model(query: CallbackQuery):
    idx = int(query.data.split(":")[1])
    models = await forge_service.get_models()
    if 0 <= idx < len(models):
        target = models[idx]
        title = target.get("title") or target.get("model_name")
        await query.answer("⏳ Chargement du modèle dans la VRAM...", show_alert=False)
        msg = await query.message.answer(f"⏳ <i>Chargement de <b>{escape(str(title))}</b> dans la VRAM...</i>")
        success = await forge_service.set_model(title)
        if success:
            await msg.edit_text(f"✅ Modèle chargé avec succès : <b>{escape(str(title))}</b>")
            state_manager.update_user_state(query.from_user.id, model=title)
            await refresh_settings_view(query)
        else:
            await msg.edit_text("❌ Échec lors du changement de modèle dans Forge Neo.")
    else:
        await query.answer("Index modèle invalide.", show_alert=True)


# ─── VAE CALLBACKS ──────────────────────────────────────────────────────────

@router.callback_query(F.data.startswith("menu_vaes:") | F.data.startswith("vaes_page:"))
async def cb_menu_vaes(query: CallbackQuery):
    page = int(query.data.split(":")[1])
    vaes = await forge_service.get_vaes()
    cur_vae = await forge_service.get_current_vae()
    await query.message.edit_text(
        f"🔮 <b>Sélection du VAE</b>\nActuel : <code>{escape(str(cur_vae))}</code>",
        reply_markup=get_vaes_keyboard(vaes, cur_vae, page=page)
    )
    await query.answer()


@router.callback_query(F.data.startswith("set_vae:"))
async def cb_set_vae(query: CallbackQuery):
    idx = int(query.data.split(":")[1])
    vaes = await forge_service.get_vaes()
    if 0 <= idx < len(vaes):
        target = vaes[idx]
        success = await forge_service.set_vae(target)
        if success:
            state_manager.update_user_state(query.from_user.id, vae=target)
            await query.answer(f"VAE appliqué : {target}")
            await refresh_settings_view(query)
        else:
            await query.answer("Erreur lors de l'application du VAE", show_alert=True)
    else:
        await query.answer("VAE introuvable.", show_alert=True)


@router.callback_query(F.data == "refresh_vaes")
async def cb_refresh_vaes(query: CallbackQuery):
    await forge_service.refresh_vae()
    vaes = await forge_service.get_vaes()
    cur_vae = await forge_service.get_current_vae()
    await query.message.edit_reply_markup(reply_markup=get_vaes_keyboard(vaes, cur_vae, page=0))
    await query.answer("Liste des VAEs rafraîchie !")


# ─── SAMPLERS CALLBACKS ─────────────────────────────────────────────────────

@router.callback_query(F.data.startswith("menu_samplers:") | F.data.startswith("samplers_page:"))
async def cb_menu_samplers(query: CallbackQuery):
    page = int(query.data.split(":")[1])
    user_state = state_manager.get_user_state(query.from_user.id)
    cur = user_state.get("sampler", "Euler a")
    samplers = await forge_service.get_samplers()
    await query.message.edit_text(
        f"🎯 <b>Sélection de la Méthode d'Échantillonnage</b>\nActuel : <code>{escape(str(cur))}</code>",
        reply_markup=get_samplers_keyboard(samplers, cur, page=page)
    )
    await query.answer()


@router.callback_query(F.data.startswith("set_sampler:"))
async def cb_set_sampler(query: CallbackQuery):
    idx = int(query.data.split(":")[1])
    samplers = await forge_service.get_samplers()
    if 0 <= idx < len(samplers):
        chosen = samplers[idx]
        state_manager.update_user_state(query.from_user.id, sampler=chosen)
        await query.answer(f"Sampler : {chosen}")
        await refresh_settings_view(query)
    else:
        await query.answer("Sampler introuvable.", show_alert=True)


# ─── SCHEDULERS CALLBACKS ───────────────────────────────────────────────────

@router.callback_query(F.data == "menu_schedulers")
async def cb_menu_schedulers(query: CallbackQuery):
    user_state = state_manager.get_user_state(query.from_user.id)
    cur = user_state.get("scheduler", "Karras")
    schedulers = await forge_service.get_schedulers()
    await query.message.edit_text(
        f"📈 <b>Sélection du Type de Schedule</b>\nActuel : <code>{escape(str(cur))}</code>",
        reply_markup=get_schedulers_keyboard(schedulers, cur)
    )
    await query.answer()


@router.callback_query(F.data.startswith("set_sched:"))
async def cb_set_sched(query: CallbackQuery):
    chosen = query.data.split(":", 1)[1]
    state_manager.update_user_state(query.from_user.id, scheduler=chosen)
    await query.answer(f"Scheduler : {chosen}")
    await refresh_settings_view(query)


# ─── STEPS CALLBACKS ────────────────────────────────────────────────────────

@router.callback_query(F.data == "menu_steps")
async def cb_menu_steps(query: CallbackQuery):
    user_state = state_manager.get_user_state(query.from_user.id)
    cur = user_state.get("steps", 28)
    await query.message.edit_text(
        f"🔢 <b>Nombre de Steps</b>\nActuel : <b>{cur}</b>\nAjustez via les boutons ci-dessous :",
        reply_markup=get_steps_keyboard(cur)
    )
    await query.answer()


@router.callback_query(F.data.startswith("set_steps_val:"))
async def cb_set_steps_val(query: CallbackQuery):
    val = int(query.data.split(":")[1])
    state_manager.update_user_state(query.from_user.id, steps=val)
    await query.message.edit_reply_markup(reply_markup=get_steps_keyboard(val))
    await query.answer(f"Steps réglé à {val}")


@router.callback_query(F.data.startswith("step_adj:"))
async def cb_step_adj(query: CallbackQuery):
    delta = int(query.data.split(":")[1])
    user_state = state_manager.get_user_state(query.from_user.id)
    cur = user_state.get("steps", 28)
    new_val = max(1, min(150, cur + delta))
    state_manager.update_user_state(query.from_user.id, steps=new_val)
    await query.message.edit_reply_markup(reply_markup=get_steps_keyboard(new_val))
    await query.answer(f"Steps : {new_val}")


@router.callback_query(F.data == "enter_custom_steps")
async def cb_enter_custom_steps(query: CallbackQuery):
    state_manager.update_user_state(query.from_user.id, waiting_for="custom_steps")
    await query.message.answer("✏️ <b>Entrez le nombre de steps désiré (ex: 35) :</b>")
    await query.answer()


# ─── CFG SCALE CALLBACKS ────────────────────────────────────────────────────

@router.callback_query(F.data == "menu_cfg")
async def cb_menu_cfg(query: CallbackQuery):
    user_state = state_manager.get_user_state(query.from_user.id)
    cur = user_state.get("cfg_scale", 6.0)
    await query.message.edit_text(
        f"🎚️ <b>CFG Scale (Prompt Guidance)</b>\nActuel : <b>{cur:.1f}</b>\nAjustez via les boutons ci-dessous :",
        reply_markup=get_cfg_keyboard(cur)
    )
    await query.answer()


@router.callback_query(F.data.startswith("set_cfg_val:"))
async def cb_set_cfg_val(query: CallbackQuery):
    val = float(query.data.split(":")[1])
    state_manager.update_user_state(query.from_user.id, cfg_scale=val)
    await query.message.edit_reply_markup(reply_markup=get_cfg_keyboard(val))
    await query.answer(f"CFG réglé à {val:.1f}")


@router.callback_query(F.data.startswith("cfg_adj:"))
async def cb_cfg_adj(query: CallbackQuery):
    delta = float(query.data.split(":")[1])
    user_state = state_manager.get_user_state(query.from_user.id)
    cur = user_state.get("cfg_scale", 6.0)
    new_val = max(0.5, min(30.0, round(cur + delta, 1)))
    state_manager.update_user_state(query.from_user.id, cfg_scale=new_val)
    await query.message.edit_reply_markup(reply_markup=get_cfg_keyboard(new_val))
    await query.answer(f"CFG : {new_val:.1f}")


@router.callback_query(F.data == "enter_custom_cfg")
async def cb_enter_custom_cfg(query: CallbackQuery):
    state_manager.update_user_state(query.from_user.id, waiting_for="custom_cfg")
    await query.message.answer("✏️ <b>Entrez la valeur CFG désirée (ex: 5.5) :</b>")
    await query.answer()


# ─── DIMENSIONS (GRANDEUR) CALLBACKS ────────────────────────────────────────

@router.callback_query(F.data == "menu_dimensions")
async def cb_menu_dimensions(query: CallbackQuery):
    user_state = state_manager.get_user_state(query.from_user.id)
    w = user_state.get("width", 1024)
    h = user_state.get("height", 1024)
    await query.message.edit_text(
        f"📐 <b>Grandeur / Dimensions d'image</b>\nActuel : <b>{w}x{h}</b>",
        reply_markup=get_dimensions_keyboard(w, h)
    )
    await query.answer()


@router.callback_query(F.data.startswith("set_dim:"))
async def cb_set_dim(query: CallbackQuery):
    parts = query.data.split(":")
    w, h = int(parts[1]), int(parts[2])
    state_manager.update_user_state(query.from_user.id, width=w, height=h)
    await query.answer(f"Dimensions : {w}x{h}")
    await refresh_settings_view(query)


@router.callback_query(F.data == "enter_custom_dim")
async def cb_enter_custom_dim(query: CallbackQuery):
    state_manager.update_user_state(query.from_user.id, waiting_for="custom_dimensions")
    await query.message.answer("✏️ <b>Entrez vos dimensions au format LargeurxHauteur (ex: 896x1152 ou 1024 1024) :</b>")
    await query.answer()


# ─── BATCH COUNT & SIZE CALLBACKS ───────────────────────────────────────────

@router.callback_query(F.data == "menu_batch")
async def cb_menu_batch(query: CallbackQuery):
    user_state = state_manager.get_user_state(query.from_user.id)
    c = user_state.get("batch_count", 1)
    s = user_state.get("batch_size", 1)
    await query.message.edit_text(
        "📦 <b>Configuration des Lots (Batch Count &amp; Size)</b>\n"
        "• <b>Batch Count :</b> nombre de passages successifs\n"
        "• <b>Batch Size :</b> nombre d'images générées en parallèle par passage",
        reply_markup=get_batch_keyboard(c, s)
    )
    await query.answer()


@router.callback_query(F.data.startswith("set_bcount:"))
async def cb_set_bcount(query: CallbackQuery):
    c = int(query.data.split(":")[1])
    user_state = state_manager.get_user_state(query.from_user.id)
    s = user_state.get("batch_size", 1)
    state_manager.update_user_state(query.from_user.id, batch_count=c)
    await query.message.edit_reply_markup(reply_markup=get_batch_keyboard(c, s))
    await query.answer(f"Batch Count : {c}")


@router.callback_query(F.data.startswith("set_bsize:"))
async def cb_set_bsize(query: CallbackQuery):
    s = int(query.data.split(":")[1])
    user_state = state_manager.get_user_state(query.from_user.id)
    c = user_state.get("batch_count", 1)
    state_manager.update_user_state(query.from_user.id, batch_size=s)
    await query.message.edit_reply_markup(reply_markup=get_batch_keyboard(c, s))
    await query.answer(f"Batch Size : {s}")


# ─── PRESET CALLBACKS ───────────────────────────────────────────────────────

@router.callback_query(F.data.startswith("set_preset:"))
async def cb_set_preset(query: CallbackQuery):
    preset_key = query.data.split(":")[1]
    state_manager.apply_preset(query.from_user.id, preset_key)
    p_name = PRESETS.get(preset_key, {}).get("name", preset_key)
    await query.answer(f"Preset appliqué : {p_name}")
    await refresh_settings_view(query)


# ─── MAGIC PROMPT & LLM CALLBACKS ───────────────────────────────────────────

@router.callback_query(F.data == "toggle_magic")
async def cb_toggle_magic(query: CallbackQuery):
    user_id = query.from_user.id
    user_state = state_manager.get_user_state(user_id)
    new_val = not user_state.get("magic_prompt", False)
    state_manager.update_user_state(user_id, magic_prompt=new_val)
    status_str = "ACTIVÉ ✨" if new_val else "Désactivé"
    await query.answer(f"Magic Prompt {status_str}")
    await refresh_settings_view(query)


@router.callback_query(F.data == "menu_ollama_models")
async def cb_menu_ollama_models(query: CallbackQuery):
    models = await ollama_service.list_models()
    if not models:
        await query.answer("Aucun modèle Ollama disponible.", show_alert=True)
        return
    user_state = state_manager.get_user_state(query.from_user.id)
    cur = user_state.get("ollama_model", "")
    await query.message.edit_text(
        "🤖 <b>Modèle LLM pour l'optimisation des prompts :</b>",
        reply_markup=get_ollama_models_keyboard(models, cur)
    )
    await query.answer()


@router.callback_query(F.data.startswith("set_ollama:"))
async def cb_set_ollama(query: CallbackQuery):
    idx = int(query.data.split(":")[1])
    models = await ollama_service.list_models()
    if 0 <= idx < len(models):
        chosen = models[idx]
        state_manager.update_user_state(query.from_user.id, ollama_model=chosen)
        await query.answer(f"Modèle Ollama : {chosen.split('/')[-1]}")
        await refresh_settings_view(query)
    else:
        await query.answer("Modèle introuvable.", show_alert=True)


# ─── NEGATIVE PROMPT CALLBACKS ──────────────────────────────────────────────

@router.callback_query(F.data == "view_neg")
async def cb_view_neg(query: CallbackQuery):
    user_state = state_manager.get_user_state(query.from_user.id)
    neg = user_state.get("negative_prompt", "")
    text = (
        f"🚫 <b>Prompt Négatif Actuel :</b>\n\n"
        f"<code>{escape(neg)}</code>\n\n"
        f"Pour le modifier, cliquez sur 'Éditer' et écrivez votre nouveau prompt négatif."
    )
    from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup
    kb = InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(text="✏️ Éditer", callback_data="edit_neg"),
            InlineKeyboardButton(text="🔄 Réinitialiser", callback_data="reset_neg"),
        ],
        [InlineKeyboardButton(text="🔙 Retour Réglages", callback_data="back_settings")]
    ])
    await query.message.edit_text(text, reply_markup=kb)
    await query.answer()


@router.callback_query(F.data == "edit_neg")
async def cb_edit_neg(query: CallbackQuery):
    state_manager.update_user_state(query.from_user.id, waiting_for="negative_prompt")
    await query.message.answer("✏️ <b>Envoyez votre nouveau prompt négatif en message :</b>")
    await query.answer("En attente de votre texte...")


@router.callback_query(F.data == "reset_neg")
async def cb_reset_neg(query: CallbackQuery):
    default_neg = DEFAULT_USER_STATE["negative_prompt"]
    state_manager.update_user_state(query.from_user.id, negative_prompt=default_neg)
    await query.answer("Prompt négatif réinitialisé !")
    await refresh_settings_view(query)


# ─── LORAS CALLBACKS ────────────────────────────────────────────────────────

@router.callback_query(F.data.startswith("loras_page:"))
async def cb_loras_page(query: CallbackQuery):
    page = int(query.data.split(":")[1])
    user_state = state_manager.get_user_state(query.from_user.id)
    loras = await forge_service.get_loras()
    await query.message.edit_reply_markup(
        reply_markup=get_loras_keyboard(loras, user_state.get("active_loras", {}), page=page)
    )
    await query.answer()


@router.callback_query(F.data.startswith("toggle_lora:"))
async def cb_toggle_lora(query: CallbackQuery):
    idx = int(query.data.split(":")[1])
    user_id = query.from_user.id
    user_state = state_manager.get_user_state(user_id)
    active_loras = dict(user_state.get("active_loras", {}))
    loras = await forge_service.get_loras()
    if 0 <= idx < len(loras):
        name = loras[idx].get("name") or loras[idx].get("alias")
        if name in active_loras:
            del active_loras[name]
            await query.answer(f"LoRA '{name}' désactivé")
        else:
            active_loras[name] = 0.8
            await query.answer(f"LoRA '{name}' activé (poids 0.8)")
        state_manager.update_user_state(user_id, active_loras=active_loras)
        page = idx // 4
        await query.message.edit_reply_markup(
            reply_markup=get_loras_keyboard(loras, active_loras, page=page)
        )
    else:
        await query.answer("LoRA non trouvé.", show_alert=True)


@router.callback_query(F.data.startswith("lora_w:"))
async def cb_lora_w(query: CallbackQuery):
    parts = query.data.split(":")
    idx = int(parts[1])
    weight = float(parts[2])
    user_id = query.from_user.id
    user_state = state_manager.get_user_state(user_id)
    active_loras = dict(user_state.get("active_loras", {}))
    loras = await forge_service.get_loras()
    if 0 <= idx < len(loras):
        name = loras[idx].get("name") or loras[idx].get("alias")
        active_loras[name] = weight
        state_manager.update_user_state(user_id, active_loras=active_loras)
        page = idx // 4
        await query.message.edit_reply_markup(
            reply_markup=get_loras_keyboard(loras, active_loras, page=page)
        )
        await query.answer(f"Poids de '{name}' : {weight}")


@router.callback_query(F.data == "refresh_loras")
async def cb_refresh_loras(query: CallbackQuery):
    await forge_service.refresh_loras()
    loras = await forge_service.get_loras()
    user_state = state_manager.get_user_state(query.from_user.id)
    await query.message.edit_reply_markup(
        reply_markup=get_loras_keyboard(loras, user_state.get("active_loras", {}), page=0)
    )
    await query.answer("LoRAs rafraîchis !")


@router.callback_query(F.data == "clear_loras")
async def cb_clear_loras(query: CallbackQuery):
    state_manager.update_user_state(query.from_user.id, active_loras={})
    loras = await forge_service.get_loras()
    await query.message.edit_reply_markup(
        reply_markup=get_loras_keyboard(loras, {}, page=0)
    )
    await query.answer("Tous les LoRAs désactivés.")


# ─── VRAM MONITOR CALLBACKS ─────────────────────────────────────────────────

@router.callback_query(F.data == "vram_refresh")
async def cb_vram_refresh(query: CallbackQuery):
    gpu_info = await vram_guard.get_vram_info()
    processes = await vram_guard.get_gpu_processes()
    text = vram_guard.format_vram_status(gpu_info, processes)
    try:
        await query.message.edit_text(text, reply_markup=get_vram_keyboard())
    except Exception:
        pass
    await query.answer("VRAM actualisée")


@router.callback_query(F.data == "vram_free")
async def cb_vram_free(query: CallbackQuery):
    res = await vram_guard.unload_ollama()
    gpu_info = await vram_guard.get_vram_info()
    processes = await vram_guard.get_gpu_processes()
    text = vram_guard.format_vram_status(gpu_info, processes)
    if res.get("unloaded"):
        text += f"\n\n✅ <b>Modèles Ollama déchargés :</b> {escape(', '.join(res['unloaded']))}"
    elif res.get("message"):
        text += f"\n\nℹ️ {res['message']}"
    try:
        await query.message.edit_text(text, reply_markup=get_vram_keyboard())
    except Exception:
        pass
    await query.answer("VRAM libérée !")


# ─── POST-GENERATION ACTIONS ────────────────────────────────────────────────

def _resolve_generation(query: CallbackQuery) -> Optional[Dict[str, Any]]:
    """Finds the metadata of the image the clicked button belongs to."""
    gen_id = query.data.split(":", 1)[1]
    meta = history.load_meta(gen_id)
    if meta:
        return meta
    # Images made before per-image metadata existed: only the last one is known.
    last_gen = state_manager.get_user_state(query.from_user.id).get("last_generation")
    if last_gen and last_gen.get("gen_id") == gen_id:
        return last_gen
    return None


def _loras_for(meta: Dict[str, Any], user_state: Dict[str, Any]) -> Dict[str, float]:
    """LoRAs used for the original image; current ones for images saved without that info."""
    if "loras" in meta:
        return dict(meta["loras"])
    return dict(user_state.get("active_loras", {}))


async def _enqueue(query: CallbackQuery, job: GenerationJob, label: str) -> None:
    pos = await gpu_queue.enqueue(job)
    await job.status_message.edit_text(
        f"⏳ <b>{label} en file d'attente (#{pos})...</b>\n<i>/cancel pour annuler</i>"
    )
    await query.answer()


@router.callback_query(F.data.startswith("reroll:"))
async def cb_reroll(query: CallbackQuery):
    meta = _resolve_generation(query)
    if not meta:
        await query.answer("Métadonnées introuvables pour cette image.", show_alert=True)
        return

    user_state = state_manager.get_user_state(query.from_user.id)
    neg_prompt = meta.get("negative_prompt", "")
    status_msg = await query.message.answer("🔄 <i>Reroll en cours...</i>")

    job = GenerationJob(
        job_type="txt2img",
        user_id=query.from_user.id,
        chat_id=query.message.chat.id,
        prompt=meta["prompt"],
        negative_prompt=neg_prompt,
        payload=build_payload(
            meta, neg_prompt,
            batch_size=user_state.get("batch_size", 1),
            batch_count=user_state.get("batch_count", 1),
        ),
        status_message=status_msg,
        loras=_loras_for(meta, user_state),
        extra_meta={"preset": meta.get("preset", "sdxl")},
    )
    await _enqueue(query, job, "Reroll")


@router.callback_query(F.data.startswith("remix:"))
async def cb_remix(query: CallbackQuery):
    meta = _resolve_generation(query)
    if not meta:
        await query.answer("Métadonnées introuvables pour cette image.", show_alert=True)
        return

    user_state = state_manager.get_user_state(query.from_user.id)
    neg_prompt = meta.get("negative_prompt", "")
    preset_key = meta.get("preset", "sdxl")
    preset = PRESETS.get(preset_key, PRESETS["sdxl"])
    status_msg = await query.message.answer("✨ <i>Création d'une variation via Ollama...</i>")

    job = GenerationJob(
        job_type="txt2img",
        user_id=query.from_user.id,
        chat_id=query.message.chat.id,
        prompt=meta.get("prompt", ""),
        negative_prompt=neg_prompt,
        payload=build_payload(
            meta, neg_prompt,
            batch_size=user_state.get("batch_size", 1),
            batch_count=user_state.get("batch_count", 1),
        ),
        status_message=status_msg,
        loras=_loras_for(meta, user_state),
        # The Ollama rewrite happens in the GPU queue, in the style of the image's preset.
        enhance={
            "mode": preset.get("mode", "tag_based"),
            "model": user_state.get("ollama_model", ""),
        },
        extra_meta={"preset": preset_key},
    )
    await _enqueue(query, job, "Remix")


@router.callback_query(F.data.startswith("upscale:"))
async def cb_upscale(query: CallbackQuery):
    meta = _resolve_generation(query)
    if not meta:
        await query.answer("Métadonnées introuvables pour cette image.", show_alert=True)
        return

    user_state = state_manager.get_user_state(query.from_user.id)
    neg_prompt = meta.get("negative_prompt", "")
    upscaler = user_state.get("hr_upscaler", "R-ESRGAN 4x+")
    common = dict(
        user_id=query.from_user.id,
        chat_id=query.message.chat.id,
        prompt=meta["prompt"],
        negative_prompt=neg_prompt,
        loras=_loras_for(meta, user_state),
    )

    if meta.get("job_type", "txt2img") == "txt2img" and not meta.get("upscaled"):
        # Plain txt2img: replay the same seed with Hires.fix.
        status_msg = await query.message.answer("🔍 <i>Hires.fix 1.5x en préparation...</i>")
        job = GenerationJob(
            job_type="txt2img",
            payload=build_payload(
                meta, neg_prompt,
                seed=meta.get("seed", -1),
                enable_hr=True,
                hr_scale=UPSCALE_FACTOR,
                hr_upscaler=upscaler,
                denoising_strength=0.4,
            ),
            status_message=status_msg,
            extra_meta={"preset": meta.get("preset", "sdxl"), "upscaled": True},
            **common,
        )
    else:
        # img2img result or already upscaled image: it cannot be replayed from a seed,
        # so enlarge the saved PNG itself.
        source_png = history.png_path(meta.get("gen_id", ""))
        if not source_png:
            await query.answer("Fichier original introuvable sur le disque.", show_alert=True)
            return
        status_msg = await query.message.answer("🔍 <i>Agrandissement 1.5x en préparation...</i>")
        job = GenerationJob(
            job_type="upscale",
            payload={"hr_upscaler": upscaler},
            status_message=status_msg,
            source_png=source_png,
            source_meta=meta,
            extra_meta={"upscaled": True},
            **common,
        )
    await _enqueue(query, job, "Upscale")


@router.callback_query(F.data.startswith("raw_png:"))
async def cb_raw_png(query: CallbackQuery, bot: Bot):
    gen_id = query.data.split(":", 1)[1]
    filepath = history.png_path(gen_id)
    if not filepath:
        await query.answer("Fichier original introuvable sur le disque.", show_alert=True)
        return

    await query.answer("Envoi du fichier PNG non compressé...")
    await bot.send_document(
        chat_id=query.message.chat.id,
        document=FSInputFile(str(filepath), filename=filepath.name),
        caption="📥 <b>PNG Brut avec métadonnées intactes</b>"
    )


# ─── IMG2IMG CALLBACKS ──────────────────────────────────────────────────────

def _discard_pending_image(pending: Optional[Dict[str, Any]]) -> None:
    if pending and pending.get("image_path"):
        Path(pending["image_path"]).unlink(missing_ok=True)


async def _pending_image_b64(pending: Optional[Dict[str, Any]]) -> Optional[str]:
    if not pending:
        return None
    if pending.get("image_b64"):  # state saved by an older version of the bot
        return pending["image_b64"]
    path = Path(pending.get("image_path") or "")
    if not path.is_file():
        return None
    data = await asyncio.to_thread(path.read_bytes)
    return base64.b64encode(data).decode("ascii")


@router.callback_query(F.data.startswith("img2img_str:"))
async def cb_img2img_strength(query: CallbackQuery):
    strength = float(query.data.split(":")[1])
    user_id = query.from_user.id
    user_state = state_manager.get_user_state(user_id)
    pending = user_state.get("pending_img2img")
    image_b64 = await _pending_image_b64(pending)
    if not image_b64:
        await query.answer("Aucune image en attente.", show_alert=True)
        return

    status_msg = await query.message.answer("🎨 <i>Lancement de la transformation img2img...</i>")
    negative_prompt = user_state.get("negative_prompt", "")
    # Size computed from the photo's aspect ratio when it was received.
    source = dict(user_state)
    source["width"] = pending.get("width", user_state.get("width", 1024))
    source["height"] = pending.get("height", user_state.get("height", 1024))

    job = GenerationJob(
        job_type="img2img",
        user_id=user_id,
        chat_id=query.message.chat.id,
        prompt=pending.get("prompt") or "masterpiece, best quality",
        negative_prompt=negative_prompt,
        payload=build_payload(
            source, negative_prompt,
            batch_size=user_state.get("batch_size", 1),
            batch_count=user_state.get("batch_count", 1),
            init_images=[image_b64],
            denoising_strength=strength,
        ),
        status_message=status_msg,
        loras=dict(user_state.get("active_loras", {})),
        extra_meta={"preset": user_state.get("preset", "sdxl")},
    )
    _discard_pending_image(pending)
    state_manager.update_user_state(user_id, pending_img2img=None, waiting_for=None)
    await _enqueue(query, job, "img2img")


@router.callback_query(F.data == "cancel_img2img")
async def cb_cancel_img2img(query: CallbackQuery):
    user_id = query.from_user.id
    _discard_pending_image(state_manager.get_user_state(user_id).get("pending_img2img"))
    state_manager.update_user_state(user_id, pending_img2img=None, waiting_for=None)
    await query.message.edit_text("❌ <b>Session img2img annulée.</b>")
    await query.answer("Annulé")


@router.callback_query(F.data == "noop")
async def cb_noop(query: CallbackQuery):
    await query.answer()
