import asyncio
import logging
from html import escape
from typing import Any, Dict, List, Optional
import aiohttp
from bot.config import settings

logger = logging.getLogger(__name__)


class VRAMGuard:
    @staticmethod
    async def get_vram_info() -> Dict[str, Any]:
        """Queries nvidia-smi asynchronously to get GPU metrics."""
        try:
            proc = await asyncio.create_subprocess_exec(
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used,memory.free,temperature.gpu,power.draw,power.limit,utilization.gpu",
                "--format=csv,noheader,nounits",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()
            if proc.returncode != 0:
                logger.error(f"nvidia-smi error: {stderr.decode()}")
                return {"error": "Impossible d'exécuter nvidia-smi"}

            lines = stdout.decode().strip().split("\n")
            if not lines or not lines[0]:
                return {"error": "Aucune information GPU retournée"}

            parts = [p.strip() for p in lines[0].split(",")]
            # name, total, used, free, temp, pwr_draw, pwr_limit, util
            name = parts[0]
            total_mb = float(parts[1])
            used_mb = float(parts[2])
            free_mb = float(parts[3])
            temp_c = int(parts[4])
            power_draw_w = float(parts[5])
            power_limit_w = float(parts[6])
            gpu_util = int(parts[7])

            used_percent = (used_mb / total_mb) * 100 if total_mb > 0 else 0

            return {
                "name": name,
                "total_mb": total_mb,
                "used_mb": used_mb,
                "free_mb": free_mb,
                "used_percent": round(used_percent, 1),
                "temperature": temp_c,
                "power_draw": round(power_draw_w, 1),
                "power_limit": round(power_limit_w, 1),
                "gpu_util": gpu_util,
            }
        except Exception as e:
            logger.error(f"Failed to query GPU info: {e}")
            return {"error": str(e)}

    @staticmethod
    async def get_gpu_processes() -> List[Dict[str, Any]]:
        """Queries processes currently holding VRAM."""
        processes = []
        try:
            proc = await asyncio.create_subprocess_exec(
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,used_memory",
                "--format=csv,noheader,nounits",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await proc.communicate()
            if proc.returncode == 0:
                for line in stdout.decode().strip().split("\n"):
                    if not line:
                        continue
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) >= 3:
                        processes.append({
                            "pid": parts[0],
                            "name": parts[1].split("/")[-1],
                            "memory_mb": float(parts[2]),
                        })
        except Exception as e:
            logger.error(f"Failed to query compute apps: {e}")
        return processes

    @staticmethod
    async def unload_ollama() -> Dict[str, Any]:
        """Forces all active Ollama models to immediately unload from VRAM."""
        results = {"unloaded": [], "errors": []}
        try:
            async with aiohttp.ClientSession() as session:
                # 1. Fetch currently running models
                ps_url = f"{settings.OLLAMA_API_URL}/api/ps"
                async with session.get(ps_url, timeout=10) as resp:
                    if resp.status != 200:
                        return {"error": f"Ollama /api/ps a répondu avec code {resp.status}"}
                    data = await resp.json()
                    models = data.get("models", [])

                if not models:
                    return {"unloaded": [], "message": "Aucun modèle Ollama n'était chargé en VRAM."}

                # 2. For each loaded model, send keep_alive: 0 to force eviction
                gen_url = f"{settings.OLLAMA_API_URL}/api/generate"
                for m in models:
                    name = m.get("name") or m.get("model")
                    if not name:
                        continue
                    payload = {"model": name, "keep_alive": 0}
                    async with session.post(gen_url, json=payload, timeout=15) as gen_resp:
                        if gen_resp.status == 200:
                            results["unloaded"].append(name)
                        else:
                            results["errors"].append(f"{name} (code {gen_resp.status})")

                # Small sleep to let Linux/driver reclaim
                await asyncio.sleep(0.5)
        except Exception as e:
            logger.error(f"Failed to unload Ollama models: {e}")
            results["errors"].append(str(e))

        return results

    @staticmethod
    def format_vram_status(gpu_info: Dict[str, Any], processes: List[Dict[str, Any]]) -> str:
        if "error" in gpu_info:
            return f"❌ Erreur lors de la lecture GPU : {escape(str(gpu_info['error']))}"

        used_gb = gpu_info["used_mb"] / 1024
        total_gb = gpu_info["total_mb"] / 1024
        free_gb = gpu_info["free_mb"] / 1024
        percent = gpu_info["used_percent"]

        # ASCII Bar 10 chars
        filled = int(round(percent / 10))
        bar = "█" * filled + "░" * (10 - filled)

        text = (
            f"📊 <b>Moniteur VRAM &amp; GPU</b>\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"🎮 <b>GPU :</b> {escape(gpu_info['name'])}\n"
            f"📈 <b>VRAM :</b> [{bar}] {percent}%\n"
            f"💾 <b>Mémoire :</b> {used_gb:.2f} Go / {total_gb:.2f} Go (Libre : {free_gb:.2f} Go)\n"
            f"🌡️ <b>Température :</b> {gpu_info['temperature']} °C\n"
            f"⚡ <b>Puissance :</b> {gpu_info['power_draw']} W / {gpu_info['power_limit']} W\n"
            f"🎯 <b>Utilisation GPU :</b> {gpu_info['gpu_util']} %\n"
        )

        if processes:
            text += "\n<b>Processus actifs sur GPU :</b>\n"
            for p in processes:
                mem = p["memory_mb"] / 1024
                text += f" • <code>{escape(p['name'])}</code> (PID {p['pid']}) : {mem:.2f} Go\n"
        else:
            text += "\n<i>Aucun processus compute détecté.</i>\n"

        return text


vram_guard = VRAMGuard()
