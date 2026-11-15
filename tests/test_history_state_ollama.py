from bot.services import history
from bot.services.ollama import clean_llm_output
from bot.services.state import DEFAULT_USER_STATE, StateManager


def test_gen_id_validation_blocks_path_tricks():
    assert history.is_valid_gen_id("20261115_120000_12345_2")
    assert not history.is_valid_gen_id("../state")
    assert not history.is_valid_gen_id("a/b")
    assert not history.is_valid_gen_id("*")
    assert not history.is_valid_gen_id("")


def test_meta_roundtrip_and_png_lookup(tmp_path):
    meta = {"gen_id": "20261115_120000_7", "prompt": "un chat <lora:x:1> & co", "seed": 7}
    history.save_meta(meta, tmp_path)
    assert history.load_meta("20261115_120000_7", tmp_path) == meta
    assert history.load_meta("unknown", tmp_path) is None
    assert history.load_meta("../etc", tmp_path) is None

    assert history.png_path("20261115_120000_7", tmp_path) is None
    (tmp_path / "20261115_120000_7.png").write_bytes(b"x")
    assert history.png_path("20261115_120000_7", tmp_path) == tmp_path / "20261115_120000_7.png"


def test_invalid_gen_id_is_not_written(tmp_path):
    history.save_meta({"gen_id": "../evil"}, tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_clean_llm_output():
    assert clean_llm_output("<think>hmm</think>\n1girl, solo") == "1girl, solo"
    assert clean_llm_output("reasoning...</think> a cat") == "a cat"
    assert clean_llm_output("```text\na cat\n```") == "a cat"
    assert clean_llm_output('"a cat"') == "a cat"
    assert clean_llm_output("<think>never finished") == ""


def test_state_defaults_persist_and_presets(tmp_path):
    path = tmp_path / "state.json"
    manager = StateManager(path)
    assert manager.get_user_state(1)["steps"] == DEFAULT_USER_STATE["steps"]

    manager.update_user_state(1, steps=40)
    manager.apply_preset(1, "flux")
    manager.apply_preset(1, "does-not-exist")

    reloaded = StateManager(path).get_user_state(1)
    assert reloaded["preset"] == "flux"
    assert reloaded["cfg_scale"] == 1.0
    assert reloaded["steps"] == 24
