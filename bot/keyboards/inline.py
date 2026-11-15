import math
from typing import Any, Dict, List, Optional
from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup
from bot.services.state import ASPECT_RATIOS, PRESETS


def get_settings_keyboard(
    user_state: Dict[str, Any],
    active_model_name: Optional[str] = None,
    active_vae_name: Optional[str] = None
) -> InlineKeyboardMarkup:
    """Builds the comprehensive Control Center dashboard."""
    # Model display
    model_disp = active_model_name or user_state.get("model") or "Chargement..."
    model_disp = model_disp.split(".")[0]
    if len(model_disp) > 16:
        model_disp = model_disp[:14] + ".."

    # VAE display
    vae_disp = active_vae_name or user_state.get("vae", "Automatic")
    vae_disp = vae_disp.split(".")[0]
    if len(vae_disp) > 14:
        vae_disp = vae_disp[:12] + ".."

    # Sampler & Scheduler
    sampler = user_state.get("sampler", "Euler a")
    if len(sampler) > 14:
        sampler = sampler[:12] + ".."
    scheduler = user_state.get("scheduler", "Karras")
    if len(scheduler) > 14:
        scheduler = scheduler[:12] + ".."

    # Steps & CFG
    steps = user_state.get("steps", 28)
    cfg = user_state.get("cfg_scale", 6.0)

    # Resolution & Batch
    w = user_state.get("width", 1024)
    h = user_state.get("height", 1024)
    b_count = user_state.get("batch_count", 1)
    b_size = user_state.get("batch_size", 1)
    total_imgs = b_count * b_size

    # Preset
    current_preset = user_state.get("preset", "sdxl")

    # Magic Prompt & Ollama
    magic_on = user_state.get("magic_prompt", False)
    ollama_model = user_state.get("ollama_model", "qwen")
    short_llm = ollama_model.split("/")[-1].split(":")[0]

    keyboard = [
        # Row 1: Model & VAE
        [
            InlineKeyboardButton(text=f"🧠 Modèle: {model_disp}", callback_data="menu_models:0"),
            InlineKeyboardButton(text=f"🔮 VAE: {vae_disp}", callback_data="menu_vaes:0"),
        ],
        # Row 2: Sampler & Scheduler
        [
            InlineKeyboardButton(text=f"🎯 Sampler: {sampler}", callback_data="menu_samplers:0"),
            InlineKeyboardButton(text=f"📈 Sched: {scheduler}", callback_data="menu_schedulers"),
        ],
        # Row 3: Steps & CFG
        [
            InlineKeyboardButton(text=f"🔢 Steps: {steps}", callback_data="menu_steps"),
            InlineKeyboardButton(text=f"🎚️ CFG: {cfg:.1f}", callback_data="menu_cfg"),
        ],
        # Row 4: Dimensions & Batch
        [
            InlineKeyboardButton(text=f"📐 Taille: {w}x{h}", callback_data="menu_dimensions"),
            InlineKeyboardButton(text=f"📦 Batch: {b_count}x{b_size} ({total_imgs} img)", callback_data="menu_batch"),
        ],
        # Row 5: Architecture Presets Switcher
        [
            InlineKeyboardButton(
                text=f"{'✅ ' if current_preset == 'flux' else ''}⚡ Flux/Krea",
                callback_data="set_preset:flux"
            ),
            InlineKeyboardButton(
                text=f"{'✅ ' if current_preset == 'sdxl' else ''}🎨 SDXL",
                callback_data="set_preset:sdxl"
            ),
            InlineKeyboardButton(
                text=f"{'✅ ' if current_preset == 'turbo' else ''}🚀 Turbo",
                callback_data="set_preset:turbo"
            ),
        ],
        # Row 6: Magic Prompt & LLM
        [
            InlineKeyboardButton(
                text=f"✨ Magic Prompt: {'🟢 ON' if magic_on else '⚪ OFF'}",
                callback_data="toggle_magic"
            ),
            InlineKeyboardButton(
                text=f"🤖 LLM: {short_llm}",
                callback_data="menu_ollama_models"
            ),
        ],
        # Row 7: LoRAs, Negative, VRAM
        [
            InlineKeyboardButton(text="🧬 LoRAs", callback_data="loras_page:0"),
            InlineKeyboardButton(text="🚫 Négatif", callback_data="view_neg"),
            InlineKeyboardButton(text="📊 VRAM GPU", callback_data="vram_refresh"),
        ],
    ]
    return InlineKeyboardMarkup(inline_keyboard=keyboard)


# ─── MODELS KEYBOARD ────────────────────────────────────────────────────────

def get_models_keyboard(
    models: List[Dict[str, Any]],
    current_model: Optional[str],
    page: int = 0,
    per_page: int = 5
) -> InlineKeyboardMarkup:
    total_pages = max(1, math.ceil(len(models) / per_page))
    page = max(0, min(page, total_pages - 1))
    start = page * per_page
    end = start + per_page
    page_models = models[start:end]

    rows = []
    for idx, m in enumerate(page_models):
        global_idx = start + idx
        title = m.get("model_name") or m.get("title", f"Model {global_idx}")
        disp = title.split(".")[0]
        if len(disp) > 28:
            disp = disp[:25] + "..."

        is_current = current_model and (current_model in m.get("title", "") or current_model in title)
        prefix = "✅ " if is_current else "📦 "
        rows.append([
            InlineKeyboardButton(text=f"{prefix}{disp}", callback_data=f"set_model:{global_idx}")
        ])

    nav = []
    if page > 0:
        nav.append(InlineKeyboardButton(text="⬅️ Précédent", callback_data=f"models_page:{page-1}"))
    nav.append(InlineKeyboardButton(text=f"📄 {page+1}/{total_pages}", callback_data="noop"))
    if page < total_pages - 1:
        nav.append(InlineKeyboardButton(text="Suivant ➡️", callback_data=f"models_page:{page+1}"))

    rows.append(nav)
    rows.append([
        InlineKeyboardButton(text="🔄 Rafraîchir", callback_data="models_page:0"),
        InlineKeyboardButton(text="🔙 Retour Réglages", callback_data="back_settings")
    ])
    return InlineKeyboardMarkup(inline_keyboard=rows)


# ─── VAE KEYBOARD ───────────────────────────────────────────────────────────

def get_vaes_keyboard(
    vaes: List[str],
    current_vae: str,
    page: int = 0,
    per_page: int = 5
) -> InlineKeyboardMarkup:
    total_pages = max(1, math.ceil(len(vaes) / per_page))
    page = max(0, min(page, total_pages - 1))
    start = page * per_page
    end = start + per_page
    page_vaes = vaes[start:end]

    rows = []
    for idx, v in enumerate(page_vaes):
        global_idx = start + idx
        disp = v.split(".")[0]
        if len(disp) > 26:
            disp = disp[:24] + ".."
        is_cur = (current_vae == v) or (current_vae in v)
        prefix = "✅ " if is_cur else "🔮 "
        rows.append([
            InlineKeyboardButton(text=f"{prefix}{disp}", callback_data=f"set_vae:{global_idx}")
        ])

    nav = []
    if page > 0:
        nav.append(InlineKeyboardButton(text="⬅️ Précédent", callback_data=f"vaes_page:{page-1}"))
    nav.append(InlineKeyboardButton(text=f"📄 {page+1}/{total_pages}", callback_data="noop"))
    if page < total_pages - 1:
        nav.append(InlineKeyboardButton(text="Suivant ➡️", callback_data=f"vaes_page:{page+1}"))

    rows.append(nav)
    rows.append([
        InlineKeyboardButton(text="🔄 Rafraîchir VAE", callback_data="refresh_vaes"),
        InlineKeyboardButton(text="🔙 Retour Réglages", callback_data="back_settings")
    ])
    return InlineKeyboardMarkup(inline_keyboard=rows)


# ─── SAMPLERS KEYBOARD ──────────────────────────────────────────────────────

def get_samplers_keyboard(
    samplers: List[str],
    current_sampler: str,
    page: int = 0,
    per_page: int = 6
) -> InlineKeyboardMarkup:
    total_pages = max(1, math.ceil(len(samplers) / per_page))
    page = max(0, min(page, total_pages - 1))
    start = page * per_page
    end = start + per_page
    page_samplers = samplers[start:end]

    rows = []
    for idx, s in enumerate(page_samplers):
        global_idx = start + idx
        is_cur = s.lower() == current_sampler.lower()
        prefix = "✅ " if is_cur else "🎯 "
        rows.append([
            InlineKeyboardButton(text=f"{prefix}{s}", callback_data=f"set_sampler:{global_idx}")
        ])

    nav = []
    if page > 0:
        nav.append(InlineKeyboardButton(text="⬅️ Précédent", callback_data=f"samplers_page:{page-1}"))
    nav.append(InlineKeyboardButton(text=f"📄 {page+1}/{total_pages}", callback_data="noop"))
    if page < total_pages - 1:
        nav.append(InlineKeyboardButton(text="Suivant ➡️", callback_data=f"samplers_page:{page+1}"))

    rows.append(nav)
    rows.append([InlineKeyboardButton(text="🔙 Retour Réglages", callback_data="back_settings")])
    return InlineKeyboardMarkup(inline_keyboard=rows)


# ─── SCHEDULERS KEYBOARD ────────────────────────────────────────────────────

def get_schedulers_keyboard(
    schedulers: List[Dict[str, str]],
    current_scheduler: str
) -> InlineKeyboardMarkup:
    rows = []
    current_row = []
    for s in schedulers:
        name = s["name"]
        label = s["label"]
        is_cur = (name.lower() == current_scheduler.lower()) or (label.lower() == current_scheduler.lower())
        prefix = "✅ " if is_cur else ""
        btn = InlineKeyboardButton(text=f"{prefix}{label}", callback_data=f"set_sched:{name}")
        current_row.append(btn)
        if len(current_row) == 2:
            rows.append(current_row)
            current_row = []
    if current_row:
        rows.append(current_row)

    rows.append([InlineKeyboardButton(text="🔙 Retour Réglages", callback_data="back_settings")])
    return InlineKeyboardMarkup(inline_keyboard=rows)


# ─── STEPS KEYBOARD ─────────────────────────────────────────────────────────

def get_steps_keyboard(current_steps: int) -> InlineKeyboardMarkup:
    presets = [8, 15, 20, 24, 28, 35, 50]
    p_row = []
    for p in presets:
        prefix = "✅" if current_steps == p else ""
        p_row.append(InlineKeyboardButton(text=f"{prefix}{p}", callback_data=f"set_steps_val:{p}"))

    stepper = [
        InlineKeyboardButton(text="➖ 5", callback_data="step_adj:-5"),
        InlineKeyboardButton(text="➖ 1", callback_data="step_adj:-1"),
        InlineKeyboardButton(text=f"🔢 {current_steps}", callback_data="noop"),
        InlineKeyboardButton(text="➕ 1", callback_data="step_adj:1"),
        InlineKeyboardButton(text="➕ 5", callback_data="step_adj:5"),
    ]

    keyboard = [
        p_row[:4],
        p_row[4:],
        stepper,
        [InlineKeyboardButton(text="✏️ Entrer valeur manuelle", callback_data="enter_custom_steps")],
        [InlineKeyboardButton(text="🔙 Retour Réglages", callback_data="back_settings")],
    ]
    return InlineKeyboardMarkup(inline_keyboard=keyboard)


# ─── CFG SCALE KEYBOARD ─────────────────────────────────────────────────────

def get_cfg_keyboard(current_cfg: float) -> InlineKeyboardMarkup:
    presets = [1.0, 1.8, 3.5, 5.0, 6.0, 7.0, 8.0]
    p_row = []
    for p in presets:
        prefix = "✅" if abs(current_cfg - p) < 0.05 else ""
        p_row.append(InlineKeyboardButton(text=f"{prefix}{p}", callback_data=f"set_cfg_val:{p}"))

    stepper = [
        InlineKeyboardButton(text="➖ 1.0", callback_data="cfg_adj:-1.0"),
        InlineKeyboardButton(text="➖ 0.5", callback_data="cfg_adj:-0.5"),
        InlineKeyboardButton(text=f"🎚️ {current_cfg:.1f}", callback_data="noop"),
        InlineKeyboardButton(text="➕ 0.5", callback_data="cfg_adj:0.5"),
        InlineKeyboardButton(text="➕ 1.0", callback_data="cfg_adj:1.0"),
    ]

    keyboard = [
        p_row[:4],
        p_row[4:],
        stepper,
        [InlineKeyboardButton(text="✏️ Entrer valeur manuelle", callback_data="enter_custom_cfg")],
        [InlineKeyboardButton(text="🔙 Retour Réglages", callback_data="back_settings")],
    ]
    return InlineKeyboardMarkup(inline_keyboard=keyboard)


# ─── DIMENSIONS (GRANDEUR) KEYBOARD ─────────────────────────────────────────

def get_dimensions_keyboard(current_w: int, current_h: int) -> InlineKeyboardMarkup:
    choices = [
        ("1:1 (1024x1024)", 1024, 1024),
        ("1:1 (512x512)", 512, 512),
        ("4:5 Portrait (896x1152)", 896, 1152),
        ("9:16 Mobile (832x1216)", 832, 1216),
        ("3:4 Classic (768x1024)", 768, 1024),
        ("16:9 Cinema (1216x832)", 1216, 832),
        ("4:3 Landscape (1024x768)", 1024, 768),
        ("21:9 UltraWide (1344x576)", 1344, 576),
    ]

    rows = []
    for label, w, h in choices:
        is_cur = (current_w == w and current_h == h)
        prefix = "✅ " if is_cur else "📐 "
        rows.append([
            InlineKeyboardButton(text=f"{prefix}{label}", callback_data=f"set_dim:{w}:{h}")
        ])

    rows.append([InlineKeyboardButton(text="✏️ Entrer Dimensions personnalisées", callback_data="enter_custom_dim")])
    rows.append([InlineKeyboardButton(text="🔙 Retour Réglages", callback_data="back_settings")])
    return InlineKeyboardMarkup(inline_keyboard=rows)


# ─── BATCH KEYBOARD ─────────────────────────────────────────────────────────

def get_batch_keyboard(batch_count: int, batch_size: int) -> InlineKeyboardMarkup:
    # Row 1: Batch Count (n_iter)
    c_btns = []
    for c in [1, 2, 3, 4]:
        prefix = "✅" if batch_count == c else ""
        c_btns.append(InlineKeyboardButton(text=f"{prefix}{c}", callback_data=f"set_bcount:{c}"))

    # Row 2: Batch Size
    s_btns = []
    for s in [1, 2, 4]:
        prefix = "✅" if batch_size == s else ""
        s_btns.append(InlineKeyboardButton(text=f"{prefix}{s}", callback_data=f"set_bsize:{s}"))

    total = batch_count * batch_size
    keyboard = [
        [InlineKeyboardButton(text="📦 Batch Count (Répétitions séquentielles) :", callback_data="noop")],
        c_btns,
        [InlineKeyboardButton(text="🖼️ Batch Size (Générations simultanées) :", callback_data="noop")],
        s_btns,
        [InlineKeyboardButton(text=f"📊 Total : {total} image(s) par envoi", callback_data="noop")],
        [InlineKeyboardButton(text="🔙 Retour Réglages", callback_data="back_settings")],
    ]
    return InlineKeyboardMarkup(inline_keyboard=keyboard)


# ─── LORAS KEYBOARD ─────────────────────────────────────────────────────────

def get_loras_keyboard(
    loras: List[Dict[str, Any]],
    active_loras: Dict[str, float],
    page: int = 0,
    per_page: int = 4
) -> InlineKeyboardMarkup:
    total_pages = max(1, math.ceil(len(loras) / per_page))
    page = max(0, min(page, total_pages - 1))
    start = page * per_page
    end = start + per_page
    page_loras = loras[start:end]

    rows = []
    for idx, lora in enumerate(page_loras):
        global_idx = start + idx
        name = lora.get("name") or lora.get("alias", f"LoRA {global_idx}")
        current_weight = active_loras.get(name)

        is_active = current_weight is not None and current_weight > 0
        disp = name if len(name) <= 20 else name[:17] + "..."

        status_label = f"✅ {disp} ({current_weight:.1f})" if is_active else f"⚪ {disp}"
        toggle_btn = InlineKeyboardButton(text=status_label, callback_data=f"toggle_lora:{global_idx}")

        weight_btns = []
        for w in [0.6, 0.8, 1.0]:
            is_w_active = is_active and abs(current_weight - w) < 0.05
            w_prefix = "🔘" if is_w_active else ""
            weight_btns.append(
                InlineKeyboardButton(text=f"{w_prefix}{w}", callback_data=f"lora_w:{global_idx}:{w}")
            )

        rows.append([toggle_btn])
        rows.append(weight_btns)

    nav = []
    if page > 0:
        nav.append(InlineKeyboardButton(text="⬅️ Précédent", callback_data=f"loras_page:{page-1}"))
    nav.append(InlineKeyboardButton(text=f"📄 {page+1}/{total_pages}", callback_data="noop"))
    if page < total_pages - 1:
        nav.append(InlineKeyboardButton(text="Suivant ➡️", callback_data=f"loras_page:{page+1}"))

    rows.append(nav)
    rows.append([
        InlineKeyboardButton(text="🔄 Rafraîchir", callback_data="refresh_loras"),
        InlineKeyboardButton(text="❌ Tout désactiver", callback_data="clear_loras")
    ])
    rows.append([InlineKeyboardButton(text="🔙 Retour Réglages", callback_data="back_settings")])
    return InlineKeyboardMarkup(inline_keyboard=rows)


# ─── POST-GENERATION KEYBOARD ───────────────────────────────────────────────

def get_post_generation_keyboard(gen_id: str) -> InlineKeyboardMarkup:
    keyboard = [
        [
            InlineKeyboardButton(text="🔄 Reroll", callback_data=f"reroll:{gen_id}"),
            InlineKeyboardButton(text="✨ Remix (Ollama)", callback_data=f"remix:{gen_id}"),
        ],
        [
            InlineKeyboardButton(text="🔍 Hires.fix / Upscale", callback_data=f"upscale:{gen_id}"),
            InlineKeyboardButton(text="📥 PNG Brut (Metadata)", callback_data=f"raw_png:{gen_id}"),
        ]
    ]
    return InlineKeyboardMarkup(inline_keyboard=keyboard)


def get_vram_keyboard() -> InlineKeyboardMarkup:
    keyboard = [
        [
            InlineKeyboardButton(text="🧹 Libérer VRAM Ollama", callback_data="vram_free"),
            InlineKeyboardButton(text="🔄 Actualiser", callback_data="vram_refresh"),
        ],
        [InlineKeyboardButton(text="🔙 Retour Réglages", callback_data="back_settings")]
    ]
    return InlineKeyboardMarkup(inline_keyboard=keyboard)


def get_img2img_strength_keyboard() -> InlineKeyboardMarkup:
    keyboard = [
        [
            InlineKeyboardButton(text="🌱 0.35 Subtil", callback_data="img2img_str:0.35"),
            InlineKeyboardButton(text="⚖️ 0.55 Équilibré", callback_data="img2img_str:0.55"),
            InlineKeyboardButton(text="🎨 0.75 Créatif", callback_data="img2img_str:0.75"),
        ],
        [InlineKeyboardButton(text="❌ Annuler", callback_data="cancel_img2img")]
    ]
    return InlineKeyboardMarkup(inline_keyboard=keyboard)


def get_ollama_models_keyboard(models: List[str], current: str) -> InlineKeyboardMarkup:
    rows = []
    for idx, m in enumerate(models):
        prefix = "✅ " if m == current else "🤖 "
        short = m.split("/")[-1]
        rows.append([
            InlineKeyboardButton(text=f"{prefix}{short}", callback_data=f"set_ollama:{idx}")
        ])
    rows.append([InlineKeyboardButton(text="🔙 Retour Réglages", callback_data="back_settings")])
    return InlineKeyboardMarkup(inline_keyboard=rows)
