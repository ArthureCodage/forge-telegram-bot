"""
Script to fix the 'TypeError: NoneType object is not callable' memory leak bug in Forge Neo
Location: backend/memory_management.py
"""
import sys
from pathlib import Path


def patch_memory_management(forge_path: Path):
    target = forge_path / "backend" / "memory_management.py"
    if not target.exists():
        print(f"Error: {target} does not exist.")
        return False

    original = target.read_text(encoding="utf-8")
    content = original

    old_is_dead = """    def is_dead(self):
        return self.real_model() is not None and self.model is None"""

    new_is_dead = """    def is_dead(self):
        return callable(self.real_model) and self.real_model() is not None and self.model is None"""

    if old_is_dead in content:
        content = content.replace(old_is_dead, new_is_dead)
        print("Patched is_dead()")

    old_cleanup = """def cleanup_models_gc(*, target: list["ModelPatcher"] = []):
    _gc: bool = False
    _del: list[int] = []

    for i in range(len(current_loaded_models)):
        cur = current_loaded_models[i]
        if not cur.is_dead():
            continue
        if any(mdl.model is cur.real_model() for mdl in target):
            _del.append(i)
            break

        logger.info("Potential memory leak detected with model {}...".format(cur.real_model().__class__.__name__))
        _gc = True

    if not _gc and len(_del) == 0:
        return

    for i in reversed(_del):
        m = current_loaded_models.pop(i)
        del m

    gc.collect()
    soft_empty_cache()

    for mdl in current_loaded_models:
        if mdl.is_dead():
            logger.warning("Memory Leak with model {} !".format(mdl.real_model().__class__.__name__))


def cleanup_models():
    to_delete = []
    for i in range(len(current_loaded_models)):
        if current_loaded_models[i].real_model() is None:
            to_delete = [i] + to_delete

    for i in to_delete:
        x = current_loaded_models.pop(i)
        del x"""

    new_cleanup = """def cleanup_models_gc(*, target: list["ModelPatcher"] = []):
    _gc: bool = False
    _del: list[int] = []

    for i in range(len(current_loaded_models)):
        cur = current_loaded_models[i]
        if not cur.is_dead():
            continue
        cur_real = cur.real_model() if callable(cur.real_model) else None
        if cur_real is not None and any(mdl.model is cur_real for mdl in target):
            _del.append(i)
            break

        model_name = cur_real.__class__.__name__ if cur_real is not None else "Unknown"
        logger.info("Potential memory leak detected with model {}...".format(model_name))
        _gc = True

    if not _gc and len(_del) == 0:
        return

    for i in reversed(_del):
        m = current_loaded_models.pop(i)
        del m

    gc.collect()
    soft_empty_cache()

    for mdl in current_loaded_models:
        if mdl.is_dead():
            mdl_real = mdl.real_model() if callable(mdl.real_model) else None
            model_name = mdl_real.__class__.__name__ if mdl_real is not None else "Unknown"
            logger.warning("Memory Leak with model {} !".format(model_name))


def cleanup_models():
    to_delete = []
    for i in range(len(current_loaded_models)):
        m = current_loaded_models[i]
        if not callable(m.real_model) or m.real_model() is None:
            to_delete = [i] + to_delete

    for i in to_delete:
        x = current_loaded_models.pop(i)
        del x"""

    if old_cleanup in content:
        content = content.replace(old_cleanup, new_cleanup)
        print("Patched cleanup functions.")

    if content == original:
        if new_is_dead in content and new_cleanup in content:
            print("Nothing to do: the patch is already applied.")
            return True
        print(
            "Error: the expected code was not found. This Forge Neo version differs "
            "from the one the patch was written for; nothing was changed."
        )
        return False

    backup = target.with_name(target.name + ".bak")
    if not backup.exists():
        backup.write_text(original, encoding="utf-8")
        print(f"Backup written to {backup}")

    target.write_text(content, encoding="utf-8")
    print("Patch successfully applied!")
    return True


if __name__ == "__main__":
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/home/ubuntu/ai/sd-webui-forge-neo")
    sys.exit(0 if patch_memory_management(path) else 1)
