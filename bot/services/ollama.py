import logging
import re
from typing import List, Optional
import aiohttp
from bot.config import settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT_TAG_BASED = """You are an expert prompt engineer for Stable Diffusion SDXL and Danbooru-style image generation models.
Task: Convert the user's input (in French or English) into high-impact comma-separated English tags.
Rules:
1. Start with essential quality tags (e.g. masterwork, best quality, ultra-detailed).
2. Describe subject, attire, pose, environment, lighting, and composition using comma-separated Danbooru/e621-style tags.
3. Translate French words to English tags.
4. Output ONLY the comma-separated prompt text.
5. NO thinking tags (<think>), NO conversational filler, NO intro, NO markdown codeblocks, NO quotes."""

SYSTEM_PROMPT_NATURAL = """You are an expert cinematic prompt engineer for Flux.1, Krea, and advanced text-to-image diffusion models.
Task: Expand the user's idea (in French or English) into rich, cinematic English descriptive prose.
Rules:
1. Describe the scene vividly: camera angle, lighting (e.g., golden hour, rim light, volumetric fog), texture, subject expressions, colors, and atmosphere.
2. Write natural English sentences/paragraphs without comma tag spam.
3. Translate French concepts accurately into cinematic English.
4. Output ONLY the descriptive prompt text.
5. NO thinking tags (<think>), NO conversational filler, NO intro, NO markdown codeblocks, NO quotes."""


def clean_llm_output(text: str) -> str:
    """Removes reasoning blocks, markdown fences and surrounding quotes from an LLM answer."""
    cleaned = text.strip()

    # 1. Strip reasoning thoughts (whether starting with <think> or just ending in </think>)
    if "</think>" in cleaned:
        cleaned = cleaned.split("</think>")[-1].strip()
    elif "<think>" in cleaned:
        # Unclosed block: the model ran out of tokens while thinking, nothing usable.
        cleaned = cleaned.split("<think>")[0].strip()

    # 2. Strip markdown code fences if model enclosed in ```
    cleaned = re.sub(r"^```(?:markdown|text)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    # 3. Strip surrounding quotation marks
    return cleaned.strip(' \n\r\t"\'`')


class OllamaService:
    def __init__(self, base_url: str = settings.OLLAMA_API_URL):
        self.base_url = base_url.rstrip("/")

    async def list_models(self) -> List[str]:
        """Returns list of locally installed Ollama models."""
        url = f"{self.base_url}/api/tags"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=10) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        models = data.get("models", [])
                        names = [m.get("name") for m in models if m.get("name")]
                        return names
                    else:
                        logger.error(f"Ollama tags returned status {resp.status}")
                        return []
        except Exception as e:
            logger.error(f"Error fetching Ollama models: {e}")
            return []

    async def enhance_prompt(
        self,
        prompt: str,
        mode: str = "tag_based",
        model: Optional[str] = None
    ) -> str:
        """
        Enhances the user prompt using Ollama with keep_alive=0 to ensure
        immediate VRAM release for Forge Neo.
        """
        active_model = model or settings.OLLAMA_DEFAULT_MODEL
        url = f"{self.base_url}/api/generate"

        system_instruction = (
            SYSTEM_PROMPT_TAG_BASED if mode == "tag_based" else SYSTEM_PROMPT_NATURAL
        )

        payload = {
            "model": active_model,
            "prompt": f"User Idea: {prompt.strip()}\nFinal Image Prompt:",
            "system": system_instruction,
            "stream": False,
            "keep_alive": 0,  # CRITICAL: unload from VRAM immediately after generation
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
                "num_predict": 250,
                "repeat_penalty": 1.15,
            }
        }

        try:
            timeout = aiohttp.ClientTimeout(total=settings.OLLAMA_TIMEOUT_SECONDS)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, json=payload) as resp:
                    if resp.status != 200:
                        err_text = await resp.text()
                        logger.error(f"Ollama enhancement failed ({resp.status}): {err_text}")
                        return prompt

                    data = await resp.json()
                    enhanced = clean_llm_output(data.get("response", ""))

                    # If empty or failed, fallback to original
                    if not enhanced:
                        return prompt

                    logger.info(f"Ollama enhanced prompt: '{prompt}' -> '{enhanced[:100]}...'")
                    return enhanced
        except Exception as e:
            logger.error(f"Ollama enhance error: {e}")
            return prompt


ollama_service = OllamaService()
