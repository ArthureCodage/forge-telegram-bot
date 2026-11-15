import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from bot.config import STATE_FILE, DATA_DIR, OUTPUTS_DIR

logger = logging.getLogger(__name__)

ASPECT_RATIOS: Dict[str, Tuple[int, int]] = {
    "1:1 (1024²)": (1024, 1024),
    "1:1 (512²)": (512, 512),
    "1:1 (768²)": (768, 768),
    "4:5 Portrait": (896, 1152),
    "9:16 Mobile": (832, 1216),
    "3:4 Portrait": (768, 1024),
    "16:9 Cinema": (1216, 832),
    "4:3 Landscape": (1024, 768),
    "21:9 UltraWide": (1344, 576),
}

PRESETS: Dict[str, Dict[str, Any]] = {
    "flux": {
        "id": "flux",
        "name": "⚡ Flux / Krea",
        "sampler": "Euler",
        "scheduler": "Simple",
        "steps": 24,
        "cfg_scale": 1.0,
        "mode": "natural_language",
    },
    "sdxl": {
        "id": "sdxl",
        "name": "🎨 SDXL / Illustrious",
        "sampler": "Euler a",
        "scheduler": "Karras",
        "steps": 28,
        "cfg_scale": 6.0,
        "mode": "tag_based",
    },
    "turbo": {
        "id": "turbo",
        "name": "🚀 Turbo / Lightning",
        "sampler": "Euler",
        "scheduler": "Normal",
        "steps": 8,
        "cfg_scale": 1.8,
        "mode": "tag_based",
    },
}

DEFAULT_USER_STATE: Dict[str, Any] = {
    "model": None,
    "vae": "Automatic",
    "sampler": "Euler a",
    "scheduler": "Karras",
    "steps": 28,
    "cfg_scale": 6.0,
    "width": 1024,
    "height": 1024,
    "aspect_ratio": "1:1 (1024²)",
    "batch_count": 1,
    "batch_size": 1,
    "preset": "sdxl",
    "magic_prompt": False,
    "ollama_model": "huihui_ai/qwen3-abliterated:14b",
    "negative_prompt": "ugly, deformed, bad anatomy, bad hands, blurry, low quality, watermark, signature",
    "active_loras": {},
    "hr_upscaler": "R-ESRGAN 4x+",
    "waiting_for": None,
    "last_generation": None,
    "pending_img2img": None,
}


class StateManager:
    def __init__(self, filepath: Path = STATE_FILE):
        self.filepath = filepath
        self._data: Dict[str, Any] = {}
        self._ensure_dirs()
        self.load()

    def _ensure_dirs(self):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    def load(self):
        if self.filepath.exists():
            try:
                with open(self.filepath, "r", encoding="utf-8") as f:
                    self._data = json.load(f)
            except Exception as e:
                logger.error(f"Failed to load state.json: {e}")
                self._data = {}
        else:
            self._data = {}

    def save(self):
        try:
            tmp = self.filepath.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self._data, f, indent=2, ensure_ascii=False)
            tmp.replace(self.filepath)
        except Exception as e:
            logger.error(f"Failed to save state.json: {e}")

    def get_user_state(self, user_id: int) -> Dict[str, Any]:
        uid = str(user_id)
        if uid not in self._data:
            self._data[uid] = dict(DEFAULT_USER_STATE)
            self.save()
        else:
            updated = False
            for k, v in DEFAULT_USER_STATE.items():
                if k not in self._data[uid]:
                    self._data[uid][k] = v
                    updated = True
            if updated:
                self.save()
        return self._data[uid]

    def update_user_state(self, user_id: int, **kwargs):
        state = self.get_user_state(user_id)
        state.update(kwargs)
        self.save()

    def apply_preset(self, user_id: int, preset_key: str):
        if preset_key in PRESETS:
            p = PRESETS[preset_key]
            self.update_user_state(
                user_id,
                preset=preset_key,
                sampler=p["sampler"],
                scheduler=p["scheduler"],
                steps=p["steps"],
                cfg_scale=p["cfg_scale"],
            )


state_manager = StateManager()
