"""Runs the real queue worker against fake Forge / Ollama / Telegram objects."""
import asyncio
import json
import struct

import pytest

from bot.services import history
from bot.services import queue as queue_module
from bot.services.queue import GenerationJob, GPUQueue, build_caption
from bot.services.state import StateManager

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\rIHDR" + struct.pack(">II", 1024, 1024) + b"rest"


class FakeMessage:
    def __init__(self):
        self.texts = []
        self.deleted = False

    async def edit_text(self, text, **kwargs):
        self.texts.append(text)

    async def delete(self):
        self.deleted = True


class FakeBot:
    def __init__(self):
        self.photos = []
        self.albums = []
        self.messages = []

    async def send_photo(self, **kwargs):
        self.photos.append(kwargs)

    async def send_media_group(self, **kwargs):
        self.albums.append(kwargs)

    async def send_message(self, **kwargs):
        self.messages.append(kwargs)


class FakeForge:
    def __init__(self, images=1):
        self.images = images
        self.payloads = []
        self.interrupted = 0
        self.release = asyncio.Event()
        self.release.set()
        self.started = asyncio.Event()

    async def get_progress(self):
        return {"progress": 0.0, "eta_relative": 0.0}

    async def interrupt(self):
        self.interrupted += 1
        self.release.set()
        return True

    async def generate(self, job_type, payload, *, timeout):
        self.payloads.append((job_type, payload, timeout))
        self.started.set()
        await self.release.wait()
        seeds = [100 + i for i in range(self.images)]
        return [PNG] * self.images, {"all_seeds": seeds, "sd_model_name": "model<x>"}


@pytest.fixture
def env(tmp_path, monkeypatch):
    forge = FakeForge()
    events = []

    class FakeOllama:
        async def enhance_prompt(self, prompt, mode, model):
            events.append(("enhance", prompt, mode, model))
            return f"enhanced {prompt}"

    class FakeVram:
        async def unload_ollama(self):
            events.append(("unload",))
            return {}

    monkeypatch.setattr(queue_module, "forge_service", forge)
    monkeypatch.setattr(queue_module, "ollama_service", FakeOllama())
    monkeypatch.setattr(queue_module, "vram_guard", FakeVram())
    monkeypatch.setattr(queue_module, "OUTPUTS_DIR", tmp_path)
    monkeypatch.setattr(history, "OUTPUTS_DIR", tmp_path)
    monkeypatch.setattr(queue_module, "state_manager", StateManager(tmp_path / "state.json"))
    return forge, events, tmp_path


def make_job(**overrides):
    params = dict(
        job_type="txt2img",
        user_id=1,
        chat_id=10,
        prompt="un chat <lora:x:1> & co",
        negative_prompt="blurry",
        payload={"width": 1024, "height": 1024, "steps": 28, "sampler_name": "Euler a",
                 "scheduler": "Karras", "cfg_scale": 6.0, "batch_size": 1, "n_iter": 1, "seed": -1},
        status_message=FakeMessage(),
    )
    params.update(overrides)
    return GenerationJob(**params)


async def run_jobs(jobs):
    bot = FakeBot()
    gpu_queue = GPUQueue()
    gpu_queue.start(bot, lambda gen_id: f"kb:{gen_id}", lambda gen_ids: list(gen_ids))
    for job in jobs:
        await gpu_queue.enqueue(job)
    await asyncio.wait_for(gpu_queue._queue.join(), timeout=10)
    gpu_queue._worker_task.cancel()
    return bot


def test_job_sends_image_with_escaped_caption_and_saves_metadata(env):
    forge, _, outputs = env
    job = make_job(loras={"detail": 0.8})
    bot = asyncio.run(run_jobs([job]))

    # LoRA tags are added to what Forge receives, not to the stored prompt.
    _, payload, timeout = forge.payloads[0]
    assert payload["prompt"] == "un chat <lora:x:1> & co <lora:detail:0.80>"
    assert timeout == 300

    assert len(bot.photos) == 1
    caption = bot.photos[0]["caption"]
    assert "un chat &lt;lora:x:1&gt; &amp; co" in caption
    assert "model&lt;x&gt;" in caption
    assert "<lora" not in caption

    metas = [json.loads(p.read_text(encoding="utf-8")) for p in outputs.glob("*.json") if p.name != "state.json"]
    assert len(metas) == 1
    meta = metas[0]
    assert meta["seed"] == 100
    assert meta["prompt"] == "un chat <lora:x:1> & co"
    assert meta["loras"] == {"detail": 0.8}
    assert (meta["final_width"], meta["final_height"]) == (1024, 1024)
    assert (outputs / f"{meta['gen_id']}.png").read_bytes() == PNG
    assert bot.photos[0]["reply_markup"] == f"kb:{meta['gen_id']}"
    assert job.status_message.deleted


def test_batch_is_sent_as_one_album_with_a_single_caption(env):
    forge, _, outputs = env
    forge.images = 3
    bot = asyncio.run(run_jobs([make_job()]))

    # One album, not one long message per image.
    assert bot.photos == []
    assert len(bot.albums) == 1
    media = bot.albums[0]["media"]
    assert len(media) == 3
    assert [m.caption is not None for m in media] == [True, False, False]
    assert "3 images" in media[0].caption
    assert "100, 101, 102" in media[0].caption
    assert "un chat &lt;lora:x:1&gt; &amp; co" in media[0].caption
    assert len(media[0].caption) < 1024

    # Buttons cannot sit on an album: one message carries the actions of every image.
    assert len(bot.messages) == 1
    gen_ids = bot.messages[0]["reply_markup"]
    assert len(gen_ids) == 3
    assert all((outputs / f"{g}.png").exists() for g in gen_ids)

    seeds = sorted(
        json.loads(p.read_text(encoding="utf-8"))["seed"]
        for p in outputs.glob("*.json") if p.name != "state.json"
    )
    assert seeds == [100, 101, 102]


def test_ollama_runs_inside_the_queue_before_forge(env):
    forge, events, _ = env
    job = make_job(prompt="a cat", enhance={"mode": "natural_language", "model": "llm"})
    asyncio.run(run_jobs([job]))

    assert events == [("enhance", "a cat", "natural_language", "llm"), ("unload",)]
    assert forge.payloads[0][1]["prompt"] == "enhanced a cat"


def test_error_is_reported_and_next_job_still_runs(env):
    forge, _, _ = env
    original = forge.generate
    calls = {"n": 0}

    async def flaky(job_type, payload, *, timeout):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("boom <b>")
        return await original(job_type, payload, timeout=timeout)

    forge.generate = flaky
    first, second = make_job(), make_job()
    bot = asyncio.run(run_jobs([first, second]))

    assert "boom &lt;b&gt;" in first.status_message.texts[-1]
    assert len(bot.photos) == 1


def test_timeout_interrupts_forge(env):
    forge, _, _ = env

    async def too_slow(job_type, payload, *, timeout):
        raise asyncio.TimeoutError()

    forge.generate = too_slow
    job = make_job()
    asyncio.run(run_jobs([job]))

    assert forge.interrupted == 1
    assert "Délai dépassé" in job.status_message.texts[-1]


def test_cancel_interrupts_running_job_and_drops_waiting_ones(env):
    forge, _, _ = env
    forge.release.clear()  # the first job blocks inside Forge until interrupted
    running, waiting = make_job(), make_job()

    async def scenario():
        bot = FakeBot()
        gpu_queue = GPUQueue()
        gpu_queue.start(bot, lambda gen_id: None)
        assert await gpu_queue.enqueue(running) == 1
        await asyncio.wait_for(forge.started.wait(), timeout=5)
        assert await gpu_queue.enqueue(waiting) == 2  # counts the running job

        result = await gpu_queue.cancel_all()
        await asyncio.wait_for(gpu_queue._queue.join(), timeout=10)
        gpu_queue._worker_task.cancel()
        return result, bot

    (dropped, was_running), bot = asyncio.run(scenario())

    assert (dropped, was_running) == (1, True)
    assert forge.interrupted == 1
    assert bot.photos == []  # the partial image is not delivered
    assert "annulée" in running.status_message.texts[-1]
    assert "annulée" in waiting.status_message.texts[-1]


def test_batch_over_ten_images_is_split_into_albums(env):
    forge, _, _ = env
    forge.images = 11
    bot = asyncio.run(run_jobs([make_job()]))

    assert [len(a["media"]) for a in bot.albums] == [10]
    assert len(bot.photos) == 1  # the 11th image cannot form an album on its own
    assert len(bot.messages[0]["reply_markup"]) == 11


def test_batch_keyboard_has_one_row_of_actions_per_image():
    from bot.keyboards.inline import get_batch_actions_keyboard

    rows = get_batch_actions_keyboard(["20261115_120000_1_0", "20261115_120000_2_1"]).inline_keyboard
    assert [b.callback_data for b in rows[1]] == [
        "reroll:20261115_120000_2_1", "remix:20261115_120000_2_1",
        "upscale:20261115_120000_2_1", "raw_png:20261115_120000_2_1",
    ]
    assert all(len(b.callback_data.encode()) <= 64 for row in rows for b in row)


def test_caption_truncates_long_prompts():
    caption = build_caption({"prompt": "x" * 2000, "seed": 1, "width": 512, "height": 512}, 0, 1)
    assert len(caption) < 1024
