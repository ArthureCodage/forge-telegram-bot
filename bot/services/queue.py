import asyncio
import io
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from aiogram import Bot
from aiogram.types import BufferedInputFile, FSInputFile, Message
from bot.config import OUTPUTS_DIR
from bot.services.forge import forge_service
from bot.services.state import state_manager

logger = logging.getLogger(__name__)


@dataclass
class GenerationJob:
    job_type: str  # "txt2img" or "img2img"
    user_id: int
    chat_id: int
    prompt: str
    negative_prompt: str
    payload: Dict[str, Any]
    status_message: Message
    future: asyncio.Future
    extra_meta: Dict[str, Any]


class GPUQueue:
    def __init__(self):
        self._queue: asyncio.Queue[GenerationJob] = asyncio.Queue()
        self._worker_task: Optional[asyncio.Task] = None
        self._bot: Optional[Bot] = None
        self._post_keyboard_builder: Optional[Callable[[str], Any]] = None

    def start(self, bot: Bot, post_keyboard_builder: Callable[[str], Any]):
        self._bot = bot
        self._post_keyboard_builder = post_keyboard_builder
        if self._worker_task is None or self._worker_task.done():
            self._worker_task = asyncio.create_task(self._worker_loop())
            logger.info("GPU Queue worker loop started.")

    async def enqueue(self, job: GenerationJob) -> int:
        position = self._queue.qsize() + 1
        await self._queue.put(job)
        return position

    def queue_size(self) -> int:
        return self._queue.qsize()

    async def _worker_loop(self):
        while True:
            job = await self._queue.get()
            try:
                await self._process_job(job)
            except Exception as e:
                logger.error(f"Error processing job for user {job.user_id}: {e}", exc_info=True)
                if not job.future.done():
                    job.future.set_exception(e)
                try:
                    await job.status_message.edit_text(f"❌ <b>Erreur pendant la génération :</b>\n<code>{str(e)[:300]}</code>")
                except Exception:
                    pass
            finally:
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

                    text = f"🎨 <b>Génération en cours...</b>\n\n[{bar}] {pct}% | ETA: {eta_str}"
                    try:
                        await status_msg.edit_text(text)
                        last_pct = pct
                        last_edit_time = now
                    except Exception:
                        pass
            except Exception as e:
                logger.debug(f"Progress polling exception: {e}")

            await asyncio.sleep(1.5)

    async def _process_job(self, job: GenerationJob):
        start_time = time.time()
        try:
            await job.status_message.edit_text("⚡ <b>Initialisation de la génération...</b>")
        except Exception:
            pass

        stop_event = asyncio.Event()
        progress_task = asyncio.create_task(self._progress_tracker(job.status_message, stop_event))

        images_bytes: List[bytes] = []
        info_dict: Dict[str, Any] = {}

        try:
            if job.job_type == "txt2img":
                images_bytes, info_dict = await forge_service.txt2img(job.payload)
            else:
                images_bytes, info_dict = await forge_service.img2img(job.payload)
        finally:
            stop_event.set()
            await progress_task

        gen_time = round(time.time() - start_time, 1)

        model_name = info_dict.get("sd_model_name", "Forge Neo")
        width = job.payload.get("width", 1024)
        height = job.payload.get("height", 1024)
        sampler = job.payload.get("sampler_name", "Euler")
        scheduler = job.payload.get("scheduler", "Automatic")
        steps = job.payload.get("steps", 20)
        cfg_scale = job.payload.get("cfg_scale", 6.0)

        all_seeds = info_dict.get("all_seeds", [])
        if not all_seeds:
            all_seeds = [info_dict.get("seed", job.payload.get("seed", -1))]

        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        total_images = len(images_bytes)

        last_meta = None

        for idx, img_data in enumerate(images_bytes):
            seed = all_seeds[idx] if idx < len(all_seeds) else all_seeds[0]
            gen_id = f"{timestamp_str}_{seed}_{idx}" if total_images > 1 else f"{timestamp_str}_{seed}"
            filepath = OUTPUTS_DIR / f"{gen_id}.png"

            with open(filepath, "wb") as f:
                f.write(img_data)

            gen_meta = {
                "gen_id": gen_id,
                "filepath": str(filepath),
                "job_type": job.job_type,
                "prompt": job.prompt,
                "negative_prompt": job.negative_prompt,
                "seed": seed,
                "model": model_name,
                "width": width,
                "height": height,
                "preset": job.extra_meta.get("preset", "sdxl"),
                "sampler": sampler,
                "scheduler": scheduler,
                "steps": steps,
                "cfg_scale": cfg_scale,
                "gen_time": gen_time,
                "created_at": datetime.now().isoformat(),
            }
            last_meta = gen_meta

            # Caption
            display_prompt = job.prompt
            if len(display_prompt) > 600:
                display_prompt = display_prompt[:597] + "..."

            batch_info = f" [Image {idx+1}/{total_images}]" if total_images > 1 else ""

            caption = (
                f"✨ <b>Image générée en {gen_time}s{batch_info}</b>\n\n"
                f"📝 <b>Prompt :</b> <i>{display_prompt}</i>\n"
                f"🎲 <b>Seed :</b> <code>{seed}</code>\n"
                f"🧠 <b>Modèle :</b> <code>{model_name}</code>\n"
                f"🎯 <b>Sampler :</b> {sampler} ({scheduler}) | Steps: {steps} | CFG: {cfg_scale}\n"
                f"📐 <b>Format :</b> {width}x{height}"
            )

            reply_markup = self._post_keyboard_builder(gen_id) if self._post_keyboard_builder else None
            input_photo = BufferedInputFile(img_data, filename=f"gen_{gen_id}.png")

            await self._bot.send_photo(
                chat_id=job.chat_id,
                photo=input_photo,
                caption=caption,
                reply_markup=reply_markup,
            )

        if last_meta:
            state_manager.update_user_state(job.user_id, last_generation=last_meta)

        try:
            await job.status_message.delete()
        except Exception:
            pass

        if not job.future.done():
            job.future.set_result(last_meta)


gpu_queue = GPUQueue()
