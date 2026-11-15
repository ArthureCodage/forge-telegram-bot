import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from aiogram import Bot
from aiogram.types import BufferedInputFile, Message
from bot.config import OUTPUTS_DIR, settings
from bot.services import history
from bot.services.forge import forge_service
from bot.services.ollama import ollama_service
from bot.services.payloads import build_prompt_with_loras, compute_timeout, png_size
from bot.services.state import state_manager
from bot.services.vram_guard import vram_guard

logger = logging.getLogger(__name__)

UPSCALE_FACTOR = 1.5
MAX_CAPTION_PROMPT = 500


@dataclass
class GenerationJob:
    job_type: str  # "txt2img", "img2img" or "upscale" (Extras upscale of an existing PNG)
    user_id: int
    chat_id: int
    prompt: str  # prompt without LoRA tags; they are appended by the worker
    negative_prompt: str
    payload: Dict[str, Any]
    status_message: Message
    loras: Dict[str, float] = field(default_factory=dict)
    # {"mode": ..., "model": ...}: rewrite the prompt with Ollama inside the GPU queue
    enhance: Optional[Dict[str, str]] = None
    # Extra fields merged into the saved metadata of each image
    extra_meta: Dict[str, Any] = field(default_factory=dict)
    # For "upscale" jobs: the PNG to enlarge and its metadata
    source_png: Optional[Path] = None
    source_meta: Optional[Dict[str, Any]] = None
    cancelled: bool = False


def build_caption(meta: Dict[str, Any], index: int, total: int) -> str:
    """HTML caption for a generated image. Every dynamic value is escaped."""
    prompt = meta.get("prompt", "")
    if len(prompt) > MAX_CAPTION_PROMPT:
        prompt = prompt[: MAX_CAPTION_PROMPT - 3] + "..."

    batch_info = f" [Image {index + 1}/{total}]" if total > 1 else ""
    width = meta.get("final_width", meta.get("width"))
    height = meta.get("final_height", meta.get("height"))

    return (
        f"✨ <b>Image générée en {meta.get('gen_time', 0)}s{batch_info}</b>\n\n"
        f"📝 <b>Prompt :</b> <i>{escape(prompt)}</i>\n"
        f"🎲 <b>Seed :</b> <code>{escape(str(meta.get('seed', '?')))}</code>\n"
        f"🧠 <b>Modèle :</b> <code>{escape(str(meta.get('model', '?')))}</code>\n"
        f"🎯 <b>Sampler :</b> {escape(str(meta.get('sampler', '?')))} "
        f"({escape(str(meta.get('scheduler', '?')))}) | Steps: {meta.get('steps', '?')} "
        f"| CFG: {meta.get('cfg_scale', '?')}\n"
        f"📐 <b>Format :</b> {width}x{height}"
    )


class GPUQueue:
    def __init__(self):
        self._queue: asyncio.Queue[GenerationJob] = asyncio.Queue()
        self._worker_task: Optional[asyncio.Task] = None
        self._bot: Optional[Bot] = None
        self._post_keyboard_builder: Optional[Callable[[str], Any]] = None
        self._current: Optional[GenerationJob] = None

    def start(self, bot: Bot, post_keyboard_builder: Callable[[str], Any]):
        self._bot = bot
        self._post_keyboard_builder = post_keyboard_builder
        if self._worker_task is None or self._worker_task.done():
            self._worker_task = asyncio.create_task(self._worker_loop())
            logger.info("GPU Queue worker loop started.")

    async def enqueue(self, job: GenerationJob) -> int:
        """Adds a job and returns its position, counting the job currently running."""
        position = self._queue.qsize() + 1 + (1 if self._current is not None else 0)
        await self._queue.put(job)
        return position

    def queue_size(self) -> int:
        return self._queue.qsize()

    async def cancel_all(self) -> Tuple[int, bool]:
        """
        Drops every waiting job and interrupts the running one.
        Returns (number of waiting jobs dropped, whether a job was running).
        """
        dropped = 0
        while True:
            try:
                job = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            self._queue.task_done()
            dropped += 1
            await self._edit_status(job, "🛑 <b>Génération annulée.</b>")

        running = self._current is not None
        if running:
            self._current.cancelled = True
            await forge_service.interrupt()
        return dropped, running

    @staticmethod
    async def _edit_status(job: GenerationJob, text: str) -> None:
        try:
            await job.status_message.edit_text(text)
        except Exception:
            pass

    async def _worker_loop(self):
        while True:
            job = await self._queue.get()
            self._current = job
            try:
                await self._process_job(job)
            except Exception as e:
                logger.error(f"Error processing job for user {job.user_id}: {e}", exc_info=True)
                await self._edit_status(
                    job,
                    f"❌ <b>Erreur pendant la génération :</b>\n<code>{escape(str(e)[:300])}</code>",
                )
            finally:
                self._current = None
                self._queue.task_done()

    async def _progress_tracker(self, status_msg: Message, stop_event: asyncio.Event):
        last_pct = -1.0
        last_edit_time = 0.0

        while not stop_event.is_set():
            try:
                prog_data = await forge_service.get_progress()
                progress = float(prog_data.get("progress", 0.0))
                eta = float(prog_data.get("eta_relative", 0.0))
                pct = int(progress * 100)

                now = time.time()
                if (abs(pct - last_pct) >= 10 or (now - last_edit_time >= 2.0)) and pct > 0:
                    filled = int(round(pct / 10))
                    bar = "█" * filled + "░" * (10 - filled)
                    eta_str = f"{int(eta)}s" if eta > 0 else "..."

                    text = (
                        f"🎨 <b>Génération en cours...</b>\n\n[{bar}] {pct}% | ETA: {eta_str}\n"
                        f"<i>/cancel pour annuler</i>"
                    )
                    try:
                        await status_msg.edit_text(text)
                        last_pct = pct
                        last_edit_time = now
                    except Exception:
                        pass
            except Exception as e:
                logger.debug(f"Progress polling exception: {e}")

            await asyncio.sleep(1.5)

    async def _enhance_prompt(self, job: GenerationJob) -> str:
        """Ollama pass, run inside the queue so it never competes with Forge for VRAM."""
        model = job.enhance.get("model") or settings.OLLAMA_DEFAULT_MODEL
        short_model = escape(model.split("/")[-1])
        await self._edit_status(job, f"✨ <i>Optimisation du prompt via Ollama ({short_model})...</i>")
        try:
            return await ollama_service.enhance_prompt(
                prompt=job.prompt, mode=job.enhance.get("mode", "tag_based"), model=model
            )
        finally:
            await vram_guard.unload_ollama()

    async def _run_forge(self, job: GenerationJob, prompt: str) -> Tuple[List[bytes], Dict[str, Any]]:
        if job.job_type == "upscale":
            source = await asyncio.to_thread(job.source_png.read_bytes)
            image = await forge_service.upscale_image(
                source,
                scale=UPSCALE_FACTOR,
                upscaler=job.payload.get("hr_upscaler", "R-ESRGAN 4x+"),
                timeout=settings.FORGE_TIMEOUT_SECONDS,
            )
            return [image], {}

        payload = dict(job.payload)
        payload["prompt"] = build_prompt_with_loras(prompt, job.loras)
        timeout = compute_timeout(payload, settings.FORGE_TIMEOUT_SECONDS)
        try:
            return await forge_service.generate(job.job_type, payload, timeout=timeout)
        except asyncio.TimeoutError:
            # Without this, Forge keeps the GPU busy on a job nobody is waiting for.
            await forge_service.interrupt()
            raise RuntimeError(f"Délai dépassé ({timeout}s). Génération interrompue côté Forge.")

    async def _process_job(self, job: GenerationJob):
        start_time = time.time()

        prompt = job.prompt
        if job.enhance:
            prompt = await self._enhance_prompt(job)
        if job.cancelled:
            await self._edit_status(job, "🛑 <b>Génération annulée.</b>")
            return

        await self._edit_status(job, "⚡ <b>Initialisation de la génération...</b>")

        stop_event = asyncio.Event()
        progress_task = asyncio.create_task(self._progress_tracker(job.status_message, stop_event))
        try:
            images_bytes, info_dict = await self._run_forge(job, prompt)
        finally:
            stop_event.set()
            await progress_task

        if job.cancelled:
            # Forge returns a partial image after an interrupt: do not deliver it.
            await self._edit_status(job, "🛑 <b>Génération annulée.</b>")
            return

        gen_time = round(time.time() - start_time, 1)
        base = job.source_meta or {}
        payload = job.payload

        all_seeds = info_dict.get("all_seeds") or [
            info_dict.get("seed", base.get("seed", payload.get("seed", -1)))
        ]
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        total_images = len(images_bytes)
        last_meta = None

        for idx, img_data in enumerate(images_bytes):
            seed = all_seeds[idx] if idx < len(all_seeds) else all_seeds[0]
            gen_id = f"{timestamp_str}_{seed}_{idx}" if total_images > 1 else f"{timestamp_str}_{seed}"
            if not history.is_valid_gen_id(gen_id):
                gen_id = f"{timestamp_str}_{idx}"
            filepath = OUTPUTS_DIR / f"{gen_id}.png"
            await asyncio.to_thread(filepath.write_bytes, img_data)

            gen_meta = {
                # width/height are the generation size (reused by Reroll/Remix);
                # final_width/final_height are the real size of the PNG.
                "job_type": base.get("job_type", job.job_type),
                "prompt": prompt,
                "negative_prompt": job.negative_prompt,
                "seed": seed,
                "model": info_dict.get("sd_model_name") or base.get("model", "Forge Neo"),
                "width": payload.get("width", base.get("width", 1024)),
                "height": payload.get("height", base.get("height", 1024)),
                "preset": base.get("preset", "sdxl"),
                "sampler": payload.get("sampler_name", base.get("sampler", "Euler")),
                "scheduler": payload.get("scheduler", base.get("scheduler", "Automatic")),
                "steps": payload.get("steps", base.get("steps", 20)),
                "cfg_scale": payload.get("cfg_scale", base.get("cfg_scale", 6.0)),
                "loras": dict(job.loras),
            }
            gen_meta.update(job.extra_meta)
            gen_meta.update({
                "gen_id": gen_id,
                "filepath": str(filepath),
                "gen_time": gen_time,
                "created_at": datetime.now().isoformat(),
            })
            size = png_size(img_data)
            if size:
                gen_meta["final_width"], gen_meta["final_height"] = size

            await asyncio.to_thread(history.save_meta, gen_meta)
            last_meta = gen_meta

            reply_markup = self._post_keyboard_builder(gen_id) if self._post_keyboard_builder else None
            await self._bot.send_photo(
                chat_id=job.chat_id,
                photo=BufferedInputFile(img_data, filename=f"gen_{gen_id}.png"),
                caption=build_caption(gen_meta, idx, total_images),
                reply_markup=reply_markup,
            )

        if last_meta:
            state_manager.update_user_state(job.user_id, last_generation=last_meta)

        try:
            await job.status_message.delete()
        except Exception:
            pass


gpu_queue = GPUQueue()
