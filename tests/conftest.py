import os
import sys
from pathlib import Path

# Settings are read at import time: provide harmless values before importing the bot.
os.environ.setdefault("TELEGRAM_BOT_TOKEN", "123456:TEST-TOKEN")
os.environ.setdefault("ALLOWED_USER_IDS", "1")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
