import base64
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import aiohttp
from bot.config import settings

logger = logging.getLogger(__name__)

DEFAULT_VAE_PATH = Path("/home/ubuntu/ai/sd-webui-forge-neo/models/VAE")


class ForgeService:
    def __init__(
        self,
        base_url: str = settings.FORGE_API_URL,
        user: str = settings.FORGE_AUTH_USER,
        password: str = settings.FORGE_AUTH_PASS,
    ):
        self.base_url = base_url.rstrip("/")
        self.auth = aiohttp.BasicAuth(user, password) if user and password else None

    def _get_auth(self) -> Optional[aiohttp.BasicAuth]:
        return self.auth

    # ─── MODELS / CHECKPOINTS ───────────────────────────────────────────────

    async def get_models(self) -> List[Dict[str, Any]]:
        url = f"{self.base_url}/sdapi/v1/sd-models"
        try:
            async with aiohttp.ClientSession(auth=self._get_auth()) as session:
                async with session.get(url, timeout=15) as resp:
                    if resp.status == 200:
                        return await resp.json()
        except Exception as e:
            logger.error(f"Failed to fetch sd-models: {e}")
        return []

    async def get_current_model(self) -> Optional[str]:
        url = f"{self.base_url}/sdapi/v1/options"
        try:
            async with aiohttp.ClientSession(auth=self._get_auth()) as session:
                async with session.get(url, timeout=15) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return data.get("sd_model_checkpoint")
        except Exception as e:
            logger.error(f"Failed to fetch current model: {e}")
        return None

    async def set_model(self, checkpoint_title: str) -> bool:
        url = f"{self.base_url}/sdapi/v1/options"
        payload = {"sd_model_checkpoint": checkpoint_title}
        try:
            timeout = aiohttp.ClientTimeout(total=90)
            async with aiohttp.ClientSession(auth=self._get_auth(), timeout=timeout) as session:
                async with session.post(url, json=payload) as resp:
                    return resp.status == 200
        except Exception as e:
            logger.error(f"Error changing model: {e}")
        return False

    # ─── VAE MANAGEMENT ─────────────────────────────────────────────────────

    async def get_vaes(self) -> List[str]:
        """Discovers available VAEs from Forge and filesystem."""
        vaes = ["Automatic", "None"]
        # Look in VAE directory
        if DEFAULT_VAE_PATH.exists():
            valid_exts = {".safetensors", ".pt", ".pth", ".ckpt", ".bin", ".sft", ".gguf"}
            for p in sorted(DEFAULT_VAE_PATH.glob("*")):
                if p.is_file() and p.suffix.lower() in valid_exts:
                    if not p.name.endswith(".preview.jpg") and not p.name.endswith(".civitai.info"):
                        if p.name not in vaes:
                            vaes.append(p.name)
        return vaes

    async def get_current_vae(self) -> str:
        url = f"{self.base_url}/sdapi/v1/options"
        try:
            async with aiohttp.ClientSession(auth=self._get_auth()) as session:
                async with session.get(url, timeout=10) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return data.get("sd_vae", "Automatic")
        except Exception:
            pass
        return "Automatic"

    async def set_vae(self, vae_name: str) -> bool:
        url = f"{self.base_url}/sdapi/v1/options"
        payload = {"sd_vae": vae_name}
        try:
            timeout = aiohttp.ClientTimeout(total=30)
            async with aiohttp.ClientSession(auth=self._get_auth(), timeout=timeout) as session:
                async with session.post(url, json=payload) as resp:
                    return resp.status == 200
        except Exception as e:
            logger.error(f"Error changing VAE: {e}")
        return False

    async def refresh_vae(self) -> bool:
        url = f"{self.base_url}/sdapi/v1/refresh-vae"
        try:
            async with aiohttp.ClientSession(auth=self._get_auth()) as session:
                async with session.post(url, timeout=15) as resp:
                    return resp.status == 200
        except Exception:
            return False

    # ─── SAMPLERS & SCHEDULERS ──────────────────────────────────────────────

    async def get_samplers(self) -> List[str]:
        url = f"{self.base_url}/sdapi/v1/samplers"
        try:
            async with aiohttp.ClientSession(auth=self._get_auth()) as session:
                async with session.get(url, timeout=10) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        names = [s.get("name") for s in data if s.get("name")]
                        if names:
                            return names
        except Exception as e:
            logger.warning(f"Failed to fetch samplers: {e}")
        return [
            "Euler a", "Euler", "DPM++ 2M", "DPM++ 2M SDE", "DPM++ SDE",
            "DPM++ 3M SDE", "LCM", "Heun", "DDIM", "Restart", "UniPC"
        ]

    async def get_schedulers(self) -> List[Dict[str, str]]:
        url = f"{self.base_url}/sdapi/v1/schedulers"
        try:
            async with aiohttp.ClientSession(auth=self._get_auth()) as session:
                async with session.get(url, timeout=10) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        scheds = []
                        for s in data:
                            name = s.get("name")
                            label = s.get("label", name)
                            if name:
                                scheds.append({"name": name, "label": label})
                        if scheds:
                            return scheds
        except Exception as e:
            logger.warning(f"Failed to fetch schedulers: {e}")
        return [
            {"name": "automatic", "label": "Automatic"},
            {"name": "karras", "label": "Karras"},
            {"name": "exponential", "label": "Exponential"},
            {"name": "simple", "label": "Simple"},
            {"name": "normal", "label": "Normal"},
            {"name": "sgm_uniform", "label": "SGM Uniform"},
            {"name": "align_your_steps", "label": "Align Your Steps"},
            {"name": "turbo", "label": "Turbo"},
            {"name": "ddim", "label": "DDIM"},
        ]

    # ─── LORAS ──────────────────────────────────────────────────────────────

    async def get_loras(self) -> List[Dict[str, Any]]:
        url = f"{self.base_url}/sdapi/v1/loras"
        try:
            async with aiohttp.ClientSession(auth=self._get_auth()) as session:
                async with session.get(url, timeout=15) as resp:
                    if resp.status == 200:
                        return await resp.json()
        except Exception as e:
            logger.error(f"Failed to fetch loras: {e}")
        return []

    async def refresh_loras(self) -> bool:
        url = f"{self.base_url}/sdapi/v1/refresh-loras"
        try:
            async with aiohttp.ClientSession(auth=self._get_auth()) as session:
                async with session.post(url, timeout=30) as resp:
                    return resp.status == 200
        except Exception as e:
            logger.error(f"Error refreshing loras: {e}")
            return False

    # ─── PROGRESS & GENERATION ──────────────────────────────────────────────

    async def get_progress(self) -> Dict[str, Any]:
        url = f"{self.base_url}/sdapi/v1/progress"
        try:
            async with aiohttp.ClientSession(auth=self._get_auth()) as session:
                async with session.get(url, timeout=5) as resp:
                    if resp.status == 200:
                        return await resp.json()
        except Exception:
            pass
        return {"progress": 0.0, "eta_relative": 0.0}

    @staticmethod
    def build_prompt_with_loras(prompt: str, active_loras: Dict[str, float]) -> str:
        if not active_loras:
            return prompt
        lora_tags = []
        for name, weight in active_loras.items():
            if weight and weight > 0:
                lora_tags.append(f"<lora:{name}:{weight:.2f}>")
        if lora_tags:
            return f"{prompt.strip()} {' '.join(lora_tags)}"
        return prompt

    async def txt2img(self, payload: Dict[str, Any]) -> Tuple[List[bytes], Dict[str, Any]]:
        """
        Executes txt2img generation. Returns list of PNG bytes (for batch) and info dict.
        """
        url = f"{self.base_url}/sdapi/v1/txt2img"
        timeout = aiohttp.ClientTimeout(total=settings.FORGE_TIMEOUT_SECONDS)
        async with aiohttp.ClientSession(auth=self._get_auth(), timeout=timeout) as session:
            async with session.post(url, json=payload) as resp:
                if resp.status != 200:
                    err = await resp.text()
                    raise RuntimeError(f"Forge txt2img failed ({resp.status}): {err[:200]}")

                data = await resp.json()
                images_b64 = data.get("images", [])
                if not images_b64:
                    raise RuntimeError("Forge txt2img n'a retourné aucune image.")

                result_images: List[bytes] = []
                for b64 in images_b64:
                    if "," in b64:
                        b64 = b64.split(",", 1)[1]
                    result_images.append(base64.b64decode(b64))

                info_raw = data.get("info", "{}")
                try:
                    info_dict = json.loads(info_raw) if isinstance(info_raw, str) else info_raw
                except Exception:
                    info_dict = {"info": str(info_raw)}

                return result_images, info_dict

    async def img2img(self, payload: Dict[str, Any]) -> Tuple[List[bytes], Dict[str, Any]]:
        """
        Executes img2img generation. Returns list of PNG bytes and info dict.
        """
        url = f"{self.base_url}/sdapi/v1/img2img"
        timeout = aiohttp.ClientTimeout(total=settings.FORGE_TIMEOUT_SECONDS)
        async with aiohttp.ClientSession(auth=self._get_auth(), timeout=timeout) as session:
            async with session.post(url, json=payload) as resp:
                if resp.status != 200:
                    err = await resp.text()
                    raise RuntimeError(f"Forge img2img failed ({resp.status}): {err[:200]}")

                data = await resp.json()
                images_b64 = data.get("images", [])
                if not images_b64:
                    raise RuntimeError("Forge img2img n'a retourné aucune image.")

                result_images: List[bytes] = []
                for b64 in images_b64:
                    if "," in b64:
                        b64 = b64.split(",", 1)[1]
                    result_images.append(base64.b64decode(b64))

                info_raw = data.get("info", "{}")
                try:
                    info_dict = json.loads(info_raw) if isinstance(info_raw, str) else info_raw
                except Exception:
                    info_dict = {"info": str(info_raw)}

                return result_images, info_dict


forge_service = ForgeService()
