<div align="center">

# 🎨 Forge & Ollama Telegram Remote Studio

**A production-ready, asynchronous Telegram Bot acting as a tactile mobile remote control and generation studio for local Stable Diffusion WebUI Forge Neo and Ollama on Linux VPS.**

[![Python Version](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue.svg)](https://www.python.org/)
[![Aiogram Version](https://img.shields.io/badge/aiogram-v3.x-2CA5E0.svg?logo=telegram)](https://github.com/aiogram/aiogram)
[![Hardware](https://img.shields.io/badge/Hardware-NVIDIA%20RTX%204090%20(24GB)-76B900.svg?logo=nvidia)](https://www.nvidia.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)]()

</div>

---

## 🌟 Overview & Key Features

* 🔒 **Strict Whitelist Security (Middleware) :** Only authorized Telegram User IDs defined in `ALLOWED_USER_IDS` can interact with the bot. Unauthorized messages and callbacks are dropped immediately at the outermost middleware level.
* ⚡ **Zero-Port Long Polling :** Fully private. Runs entirely via Telegram Long Polling (`getUpdates`); no public ports or webhooks need to be opened on the VPS.
* 🧠 **VRAM Guard & Sequential Orchestration :** Automatically orchestrates VRAM between Ollama LLMs and Forge Neo. During **Magic Prompt** enhancement, Ollama executes with `keep_alive: 0` so the model is unloaded from VRAM the millisecond token generation finishes, freeing 100% of the 24GB VRAM for Stable Diffusion.
* ✍️ **Architecture-Aware Prompt Enhancement (Ollama) :**
  * **Tag-Based Mode (SDXL / Pony / Illustrious) :** Converts raw descriptions into high-impact Danbooru tags with quality keywords.
  * **Natural Language Mode (Flux.1 / Krea / Chroma) :** Expands ideas into cinematic descriptive prose without tag spam.
  * Automatic stripping of reasoning blocks (`<think>...</think>`) and markdown fences.
* 📊 **Async GPU Job Queue (`asyncio.Queue`) & Live Progress Bar :**
  * Immediate queue position notification.
  * Real-time polling of `/sdapi/v1/progress` with a live visual ASCII progress bar: `[██████░░░░] 60% | ETA: 4s`.
* 📱 **Tactile Touch Control Center (`/settings`) :**
  * **Checkpoints :** Paginated SD model browser with 1-click VRAM loading.
  * **VAEs :** Select between `Automatic`, `None`, and all detected VAE files in `models/VAE`.
  * **Sampling Methods :** Full support for `Euler a`, `Euler`, `DPM++ 2M`, `DPM++ 2M SDE`, `DPM++ SDE`, `LCM`, `Heun`, `DDIM`, etc.
  * **Schedulers :** `Automatic`, `Karras`, `Exponential`, `Simple`, `Normal`, `SGM Uniform`, `Align Your Steps`, etc.
  * **Steps & CFG :** Quick buttons, step increments (`-5`, `-1`, `+1`, `+5`), and manual text input.
  * **Resolution & Aspect Ratio :** 8 aspect ratios (`1:1 1024²`, `4:5 Portrait`, `9:16 Mobile`, `16:9 Cinema`, `21:9 UltraWide`, etc.) or custom dimensions.
  * **Batch Count & Batch Size :** Generate 1 to 16 images per prompt with individual image delivery.
* 📸 **Mobile Img2Img :** Send any photo to the bot to transform it with adjustable denoising strength (`0.35 Subtle`, `0.55 Balanced`, `0.75 Creative`).
* 🔄 **Post-Generation Action Bar :**
  * `🔄 Reroll` : Re-run with new random seed.
  * `✨ Remix (Ollama)` : Generate a fresh variation using Ollama.
  * `🔍 Hires.fix / Upscale` : 1.5x Upscale with `R-ESRGAN 4x+` keeping the original seed.
  * `📥 PNG Brut (Metadata)` : Send uncompressed PNG as document preserving A1111/Forge generation parameters for desktop drag-and-drop.
* 🌡️ **Hardware Monitor (`/vram`) :** Real-time display of RTX 4090 VRAM usage, temperature, power draw (W), and compute processes, with a button to force-evict Ollama from VRAM.

---

## 🏗️ Architecture

```text
forge-telegram-bot/
├── .env                  # Environment variables & secrets (ignored in git)
├── .env.example          # Template configuration
├── requirements.txt      # Dependencies
├── README.md             # Documentation
├── LICENSE               # MIT License
├── run.py                # Main async entry point
├── systemd/
│   └── forge-tg-bot.service  # Systemd daemon configuration
├── scripts/
│   └── fix_forge_memory_leak.py # Utility patch for Forge Neo memory leak
├── data/
│   ├── state.json        # Persistent user settings and state (ignored in git)
│   └── outputs/          # Generated PNG outputs backup (ignored in git)
└── bot/
    ├── config.py         # Pydantic Settings configuration
    ├── middlewares/
    │   └── auth.py       # Strict security whitelist middleware
    ├── services/
    │   ├── vram_guard.py # GPU metrics & Ollama VRAM eviction
    │   ├── ollama.py     # Architecture-aware prompt enhancer
    │   ├── forge.py      # Authenticated Forge Neo REST client
    │   ├── queue.py      # Async GPU job queue & live progress tracker
    │   └── state.py      # User settings persistence
    ├── handlers/
    │   ├── commands.py   # /settings, /models, /vae, /sampler, /steps, /vram...
    │   ├── generation.py # txt2img & mobile img2img handlers
    │   └── callbacks.py  # Inline button callbacks & action bar
    └── keyboards/
        └── inline.py     # Dynamic inline keyboards & pagination
```

---

## 🚀 Installation & Setup

### 1. Prerequisites
- Linux VPS with Python 3.11+
- Stable Diffusion WebUI Forge Neo running with `--api` (default port `7860`)
- Ollama running locally (default port `11434`)
- A Telegram Bot Token from [@BotFather](https://t.me/BotFather)
- Your numeric Telegram User ID from [@userinfobot](https://t.me/userinfobot)

### 2. Clone & Install
```bash
git clone https://github.com/ArthureCodage/forge-telegram-bot.git
cd forge-telegram-bot

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure Environment
Copy `.env.example` to `.env` and fill in your credentials:
```bash
cp .env.example .env
nano .env
```
```ini
# Telegram Configuration
TELEGRAM_BOT_TOKEN=YOUR_BOT_TOKEN_HERE
ALLOWED_USER_IDS=YOUR_TELEGRAM_USER_ID

# Forge Neo API
FORGE_API_URL=http://127.0.0.1:7860
FORGE_AUTH_USER=forge
FORGE_AUTH_PASS=your_forge_password

# Ollama API
OLLAMA_API_URL=http://127.0.0.1:11434
OLLAMA_DEFAULT_MODEL=huihui_ai/qwen3-abliterated:14b

# Timeouts in seconds
FORGE_TIMEOUT_SECONDS=300
OLLAMA_TIMEOUT_SECONDS=120
```

### 4. Run Directly
```bash
python run.py
```

### 5. Setup Systemd Service (Background Daemon)
To keep the bot running persistently in the background:
```bash
sudo cp systemd/forge-tg-bot.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now forge-tg-bot.service

# Check service status & logs
sudo systemctl status forge-tg-bot.service
journalctl -u forge-tg-bot.service -f
```

---

## 📱 Bot Commands Reference

| Command | Description |
|---|---|
| `/settings` | 🎛️ Opens the full tactile Control Center |
| `/models` | 🧠 Browse and load SD Checkpoints into VRAM |
| `/vae` | 🔮 Select active VAE (`Automatic`, `None`, custom) |
| `/sampler` | 🎯 Choose Sampling Method (Euler, DPM++ 2M SDE...) |
| `/scheduler` | 📈 Choose Schedule Type (Karras, Simple, Normal...) |
| `/steps` | 🔢 Adjust Sampling Steps |
| `/cfg` | 🎚️ Adjust CFG Scale (Prompt Guidance) |
| `/size` | 📐 Change Dimensions & Aspect Ratios |
| `/batch` | 📦 Configure Batch Count & Batch Size |
| `/loras` | 🧬 Manage and dose LoRAs (`0.6`, `0.8`, `1.0`) |
| `/presets` | ⚡ Quick architecture switcher (Flux, SDXL, Turbo) |
| `/vram` | 📊 Real-time RTX 4090 VRAM, Wattage & Temp monitor |

---

## 📄 License
This project is licensed under the [MIT License](LICENSE).
