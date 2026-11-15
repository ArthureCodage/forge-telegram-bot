"""Pure helpers that build Forge payloads. No network, no Telegram: easy to test."""
import math
import re
import struct
from typing import Any, Dict, Mapping, Optional, Tuple

MIN_SIDE = 256
MAX_SIDE = 2048


def build_prompt_with_loras(prompt: str, active_loras: Optional[Mapping[str, float]]) -> str:
    """Appends <lora:name:weight> tags for every LoRA with a positive weight."""
    if not active_loras:
        return prompt
    tags = [
        f"<lora:{name}:{weight:.2f}>"
        for name, weight in active_loras.items()
        if weight and weight > 0
    ]
    if not tags:
        return prompt
    return f"{prompt.strip()} {' '.join(tags)}"


def build_payload(
    source: Mapping[str, Any],
    negative_prompt: str,
    *,
    batch_size: int = 1,
    batch_count: int = 1,
    seed: int = -1,
    **extra: Any,
) -> Dict[str, Any]:
    """
    Builds a txt2img/img2img payload from a settings mapping (user state or the
    saved metadata of a previous image: both use the same key names).
    The prompt is added by the queue worker, after the optional Ollama pass.
    """
    payload: Dict[str, Any] = {
        "negative_prompt": negative_prompt,
        "steps": source.get("steps", 28),
        "sampler_name": source.get("sampler", "Euler a"),
        "scheduler": source.get("scheduler", "Karras"),
        "cfg_scale": source.get("cfg_scale", 6.0),
        "width": source.get("width", 1024),
        "height": source.get("height", 1024),
        "batch_size": batch_size,
        "n_iter": batch_count,
        "seed": seed,
        "save_images": True,
        # Without this, Forge returns a grid as first image and seeds get misaligned.
        "do_not_save_grid": True,
    }
    payload.update(extra)
    return payload


def image_count(payload: Mapping[str, Any]) -> int:
    return max(1, int(payload.get("batch_size", 1))) * max(1, int(payload.get("n_iter", 1)))


def compute_timeout(payload: Mapping[str, Any], per_image_seconds: int) -> int:
    """Timeout that scales with the number of images; Hires.fix doubles the budget."""
    factor = image_count(payload) * (2 if payload.get("enable_hr") else 1)
    return per_image_seconds * factor


def _clamp_side(value: float) -> int:
    return max(MIN_SIDE, min(MAX_SIDE, (int(value) // 8) * 8))


def parse_dimensions(text: str) -> Optional[Tuple[int, int]]:
    """Parses '896x1152', '1024 1024', '832*1216'... Returns sides clamped to multiples of 8."""
    match = re.search(r"(\d{3,4})[\s*xX,;:]+(\d{3,4})", text)
    if not match:
        return None
    return _clamp_side(int(match.group(1))), _clamp_side(int(match.group(2)))


def fit_dimensions(src_w: int, src_h: int, target_w: int, target_h: int) -> Tuple[int, int]:
    """
    Keeps the aspect ratio of a source image while matching the pixel budget of
    the target size, so img2img does not stretch the photo.
    """
    if src_w <= 0 or src_h <= 0:
        return target_w, target_h
    area = target_w * target_h
    ratio = src_w / src_h
    return _clamp_side(round(math.sqrt(area * ratio))), _clamp_side(round(math.sqrt(area / ratio)))


def png_size(data: bytes) -> Optional[Tuple[int, int]]:
    """Reads width/height from a PNG header, or None if the data is not a PNG."""
    if len(data) < 24 or data[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    width, height = struct.unpack(">II", data[16:24])
    return width, height
