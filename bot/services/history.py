"""Per-image metadata stored next to each PNG, so action buttons target the right image."""
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, Optional
from bot.config import OUTPUTS_DIR

logger = logging.getLogger(__name__)

_GEN_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,80}$")


def is_valid_gen_id(gen_id: str) -> bool:
    """gen_id comes from callback data: refuse anything that could escape the outputs folder."""
    return bool(_GEN_ID_RE.match(gen_id or ""))


def save_meta(meta: Dict[str, Any], outputs_dir: Optional[Path] = None) -> None:
    gen_id = meta.get("gen_id", "")
    if not is_valid_gen_id(gen_id):
        logger.error(f"Refusing to save metadata with invalid gen_id: {gen_id!r}")
        return
    try:
        path = (outputs_dir or OUTPUTS_DIR) / f"{gen_id}.json"
        path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception as e:
        logger.error(f"Failed to save metadata for {gen_id}: {e}")


def load_meta(gen_id: str, outputs_dir: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    if not is_valid_gen_id(gen_id):
        return None
    path = (outputs_dir or OUTPUTS_DIR) / f"{gen_id}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.error(f"Failed to read metadata for {gen_id}: {e}")
        return None


def png_path(gen_id: str, outputs_dir: Optional[Path] = None) -> Optional[Path]:
    if not is_valid_gen_id(gen_id):
        return None
    path = (outputs_dir or OUTPUTS_DIR) / f"{gen_id}.png"
    return path if path.exists() else None
