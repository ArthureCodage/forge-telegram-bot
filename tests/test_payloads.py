import struct

from bot.services.payloads import (
    build_payload,
    build_prompt_with_loras,
    compute_timeout,
    fit_dimensions,
    parse_dimensions,
    png_size,
)


def test_loras_are_appended_and_zero_weights_skipped():
    prompt = build_prompt_with_loras(" a cat ", {"detail": 0.8, "off": 0, "style": 1})
    assert prompt == "a cat <lora:detail:0.80> <lora:style:1.00>"


def test_prompt_unchanged_without_loras():
    assert build_prompt_with_loras("a cat", {}) == "a cat"
    assert build_prompt_with_loras("a cat", {"off": 0}) == "a cat"


def test_payload_reads_user_state_keys():
    state = {"steps": 8, "sampler": "Euler", "scheduler": "Simple", "cfg_scale": 1.0,
             "width": 832, "height": 1216}
    payload = build_payload(state, "blurry", batch_size=2, batch_count=3)
    assert payload["steps"] == 8
    assert payload["sampler_name"] == "Euler"
    assert payload["scheduler"] == "Simple"
    assert (payload["width"], payload["height"]) == (832, 1216)
    assert (payload["batch_size"], payload["n_iter"]) == (2, 3)
    assert payload["negative_prompt"] == "blurry"
    assert payload["seed"] == -1
    assert payload["do_not_save_grid"] is True
    assert "prompt" not in payload  # added by the queue worker


def test_payload_extra_fields_override():
    payload = build_payload({}, "", seed=42, enable_hr=True, hr_scale=1.5)
    assert payload["seed"] == 42
    assert payload["enable_hr"] is True
    assert payload["hr_scale"] == 1.5


def test_timeout_scales_with_images_and_hires():
    assert compute_timeout({"batch_size": 1, "n_iter": 1}, 300) == 300
    assert compute_timeout({"batch_size": 4, "n_iter": 2}, 300) == 2400
    assert compute_timeout({"enable_hr": True}, 300) == 600


def test_parse_dimensions():
    assert parse_dimensions("896x1152") == (896, 1152)
    assert parse_dimensions("1024 1024") == (1024, 1024)
    assert parse_dimensions("1000*601") == (1000, 600)  # rounded down to a multiple of 8
    assert parse_dimensions("9999x100") == (2048, 256)  # clamped
    assert parse_dimensions("large") is None


def test_fit_dimensions_keeps_ratio_and_pixel_budget():
    w, h = fit_dimensions(3000, 4000, 1024, 1024)
    assert w % 8 == 0 and h % 8 == 0
    assert abs(w / h - 0.75) < 0.02
    assert abs(w * h - 1024 * 1024) / (1024 * 1024) < 0.05
    assert fit_dimensions(0, 0, 832, 1216) == (832, 1216)


def test_png_size():
    header = b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\rIHDR" + struct.pack(">II", 1536, 1024)
    assert png_size(header) == (1536, 1024)
    assert png_size(b"\xff\xd8\xff not a png, long enough to pass the length test") is None


def test_pick_upscaler_falls_back_to_an_installed_one():
    from bot.services.payloads import pick_upscaler

    installed = ["None", "Lanczos", "Nearest", "ESRGAN"]
    assert pick_upscaler("ESRGAN", installed) == "ESRGAN"
    assert pick_upscaler("R-ESRGAN 4x+", installed) == "ESRGAN"
    assert pick_upscaler("R-ESRGAN 4x+", ["None", "Lanczos", "Nearest"]) == "Lanczos"
    assert pick_upscaler("R-ESRGAN 4x+", []) == "R-ESRGAN 4x+"
