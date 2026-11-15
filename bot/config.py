from pathlib import Path
from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    TELEGRAM_BOT_TOKEN: str
    ALLOWED_USER_IDS: str

    FORGE_API_URL: str = "http://127.0.0.1:7860"
    # No default credentials: set them in .env. Leave both empty if Forge runs without --api-auth.
    FORGE_AUTH_USER: str = ""
    FORGE_AUTH_PASS: str = ""
    # Folder scanned to list VAE files. If it does not exist, the Forge API is used instead.
    FORGE_VAE_DIR: str = "/home/ubuntu/ai/sd-webui-forge-neo/models/VAE"

    OLLAMA_API_URL: str = "http://127.0.0.1:11434"
    OLLAMA_DEFAULT_MODEL: str = "huihui_ai/qwen3-abliterated:14b"

    # Time budget for ONE image; the real timeout scales with batch size and Hires.fix.
    FORGE_TIMEOUT_SECONDS: int = 300
    OLLAMA_TIMEOUT_SECONDS: int = 120

    @property
    def allowed_user_ids_list(self) -> List[int]:
        ids: List[int] = []
        for raw in self.ALLOWED_USER_IDS.split(","):
            raw = raw.strip()
            if raw.isdigit():
                ids.append(int(raw))
        return ids


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUTS_DIR = DATA_DIR / "outputs"
PENDING_DIR = DATA_DIR / "pending"
STATE_FILE = DATA_DIR / "state.json"

DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
PENDING_DIR.mkdir(parents=True, exist_ok=True)

settings = Settings()
