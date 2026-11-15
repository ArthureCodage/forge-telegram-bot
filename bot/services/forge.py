import base64
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import aiohttp
from bot.config import settings

logger = logging.getLogger(__name__)

VAE_EXTENSIONS = {".safetensors", ".pt", ".pth", ".ckpt", ".bin", ".sft", ".gguf"}

FALLBACK_SAMPLERS = [
    "Euler a", "Euler", "DPM++ 2M", "DPM++ 2M SDE", "DPM++ SDE",
    "DPM++ 3M SDE", "LCM", "Heun", "DDIM", "Restart", "UniPC"
]

FALLBACK_SCHEDULERS = [
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

GENERATION_ENDPOINTS = {
    "txt2img": "/sdapi/v1/txt2img",
    "img2img": "/sdapi/v1/img2img",
}


class ForgeError(RuntimeError):
    """Forge answered with an error, or with an unusable response."""


def _decode_image(b64: str) -> bytes:
    if "," in b64:
        b64 = b64.split(",", 1)[1]
    return base64.b64decode(b64)


class ForgeService:
    def __init__(
        self,
        base_url: str = settings.FORGE_API_URL,
        user: str = settings.FORGE_AUTH_USER,
        password: str = settings.FORGE_AUTH_PASS,
        vae_dir: str = settings.FORGE_VAE_DIR,
    ):
        self.base_url = base_url.rstrip("/")
        self.auth = aiohttp.BasicAuth(user, password) if user and password else None
        self.vae_dir = Path(vae_dir) if vae_dir else None
        self._session: Optional[aiohttp.ClientSession] = None

    # ─── HTTP PLUMBING ──────────────────────────────────────────────────────

    def _get_session(self) -> aiohttp.ClientSession:
        """One shared session, created lazily inside the running event loop."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(auth=self.auth)
        return self._session

    async def close(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()

    async def _request(self, method: str, path: str, *, timeout: float, payload: Any = None) -> Any:
        """Sends a request and returns the decoded JSON body (None if the body is empty)."""
        url = f"{self.base_url}{path}"
        async with self._get_session().request(
            method, url, json=payload, timeout=aiohttp.ClientTimeout(total=timeout)
        ) as resp:
            body = await resp.read()
            if resp.status != 200:
                detail = body.decode("utf-8", errors="replace")[:200]
                raise ForgeError(f"Forge {path} a répondu {resp.status} : {detail}")
            if not body:
                return None
            try:
                return json.loads(body)
            except ValueError:
                return None

    async def _get_or(self, path: str, default: Any, *, timeout: float = 15) -> Any:
        """GET that never raises: logs and returns `default` when Forge is unreachable."""
        try:
            data = await self._request("GET", path, timeout=timeout)
            return default if data is None else data
        except Exception as e:
            logger.warning(f"Forge GET {path} failed: {e}")
            return default

    async def _post_ok(self, path: str, payload: Any = None, *, timeout: float = 30) -> bool:
        """POST that never raises: returns True on HTTP 200."""
        try:
            await self._request("POST", path, timeout=timeout, payload=payload)
            return True
        except Exception as e:
            logger.error(f"Forge POST {path} failed: {e}")
            return False

    # ─── MODELS / CHECKPOINTS ───────────────────────────────────────────────

    async def get_models(self) -> List[Dict[str, Any]]:
        return await self._get_or("/sdapi/v1/sd-models", [])

    async def get_options(self) -> Dict[str, Any]:
        return await self._get_or("/sdapi/v1/options", {})

    async def get_current_model(self) -> Optional[str]:
        return (await self.get_options()).get("sd_model_checkpoint")

    async def set_model(self, checkpoint_title: str) -> bool:
        # Loading a large checkpoint (Flux) from disk can take several minutes.
        return await self._post_ok(
            "/sdapi/v1/options", {"sd_model_checkpoint": checkpoint_title}, timeout=300
        )

    # ─── VAE MANAGEMENT ─────────────────────────────────────────────────────

    async def get_vaes(self) -> List[str]:
        """Lists VAEs from the configured folder, or from the Forge API if the folder is missing."""
        vaes = ["Automatic", "None"]
        if self.vae_dir is not None and self.vae_dir.exists():
            for p in sorted(self.vae_dir.glob("*")):
                if p.is_file() and p.suffix.lower() in VAE_EXTENSIONS and p.name not in vaes:
                    vaes.append(p.name)
            return vaes

        for item in await self._get_or("/sdapi/v1/sd-vae", []):
            name = item.get("model_name") if isinstance(item, dict) else None
            if name and name not in vaes:
                vaes.append(name)
        return vaes

    async def get_current_vae(self) -> str:
        return (await self.get_options()).get("sd_vae") or "Automatic"

    async def set_vae(self, vae_name: str) -> bool:
        return await self._post_ok("/sdapi/v1/options", {"sd_vae": vae_name}, timeout=30)

    async def refresh_vae(self) -> bool:
        return await self._post_ok("/sdapi/v1/refresh-vae", timeout=15)

    # ─── SAMPLERS & SCHEDULERS ──────────────────────────────────────────────

    async def get_samplers(self) -> List[str]:
        data = await self._get_or("/sdapi/v1/samplers", [], timeout=10)
        names = [s.get("name") for s in data if isinstance(s, dict) and s.get("name")]
        return names or list(FALLBACK_SAMPLERS)

    async def get_schedulers(self) -> List[Dict[str, str]]:
        data = await self._get_or("/sdapi/v1/schedulers", [], timeout=10)
        scheds = [
            {"name": s["name"], "label": s.get("label") or s["name"]}
            for s in data
            if isinstance(s, dict) and s.get("name")
        ]
        return scheds or [dict(s) for s in FALLBACK_SCHEDULERS]

    async def get_upscalers(self) -> List[str]:
        """Upscaler names installed in this Forge (empty list if the API cannot be reached)."""
        data = await self._get_or("/sdapi/v1/upscalers", [], timeout=10)
        return [u.get("name") for u in data if isinstance(u, dict) and u.get("name")]

    # ─── LORAS ──────────────────────────────────────────────────────────────

    async def get_loras(self) -> List[Dict[str, Any]]:
        return await self._get_or("/sdapi/v1/loras", [])

    async def refresh_loras(self) -> bool:
        return await self._post_ok("/sdapi/v1/refresh-loras", timeout=30)

    # ─── PROGRESS, INTERRUPT & GENERATION ───────────────────────────────────

    async def get_progress(self) -> Dict[str, Any]:
        try:
            data = await self._request("GET", "/sdapi/v1/progress", timeout=5)
            if isinstance(data, dict):
                return data
        except Exception:
            pass
        return {"progress": 0.0, "eta_relative": 0.0}

    async def interrupt(self) -> bool:
        """Asks Forge to stop the generation in progress."""
        return await self._post_ok("/sdapi/v1/interrupt", timeout=10)

    async def generate(
        self, job_type: str, payload: Dict[str, Any], *, timeout: float
    ) -> Tuple[List[bytes], Dict[str, Any]]:
        """Runs txt2img or img2img. Returns the PNG bytes of each image and the info dict."""
        path = GENERATION_ENDPOINTS[job_type]
        data = await self._request("POST", path, timeout=timeout, payload=payload) or {}

        images_b64 = data.get("images") or []
        if not images_b64:
            raise ForgeError(f"Forge {job_type} n'a retourné aucune image.")

        info_raw = data.get("info", "{}")
        try:
            info = json.loads(info_raw) if isinstance(info_raw, str) else dict(info_raw)
        except Exception:
            info = {"info": str(info_raw)}

        # If Forge still prepended a grid, skip it so images line up with all_seeds.
        first = int(info.get("index_of_first_image") or 0)
        if 0 < first < len(images_b64):
            images_b64 = images_b64[first:]

        return [_decode_image(b64) for b64 in images_b64], info

    async def upscale_image(
        self, image: bytes, *, scale: float, upscaler: str, timeout: float
    ) -> bytes:
        """Upscales an existing image with the 'Extras' endpoint (no re-generation)."""
        payload = {
            "image": base64.b64encode(image).decode("ascii"),
            "upscaling_resize": scale,
            "upscaler_1": upscaler,
        }
        data = await self._request(
            "POST", "/sdapi/v1/extra-single-image", timeout=timeout, payload=payload
        ) or {}
        if not data.get("image"):
            raise ForgeError("Forge n'a retourné aucune image agrandie.")
        return _decode_image(data["image"])


forge_service = ForgeService()
