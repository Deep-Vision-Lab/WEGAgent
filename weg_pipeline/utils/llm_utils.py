"""
LLM/VLM client wrappers with unified interface and retry logic.
"""
import json
import os
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

from tenacity import retry, stop_after_attempt, wait_exponential


class BaseLLMClient(ABC):
    """Abstract base class for LLM/VLM clients."""

    def __init__(self, model: str, temperature: float = 0.0, max_tokens: int = 16384):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    @abstractmethod
    def complete(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate text completion."""
        pass

    @abstractmethod
    def complete_with_image(
        self,
        prompt: str,
        image_path: Path | str,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Generate completion with image input (for VLM)."""
        pass


class OpenAIClient(BaseLLMClient):
    """OpenAI GPT client (supports both LLM and VLM)."""

    def __init__(self, model: str = "gpt-4o-mini", **kwargs):
        super().__init__(model, **kwargs)
        from openai import OpenAI
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def complete(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content or ""

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def complete_with_image(
        self,
        prompt: str,
        image_path: Path | str,
        system_prompt: Optional[str] = None,
    ) -> str:
        from .image_utils import image_to_data_uri

        image_uri = image_to_data_uri(image_path, max_size=(1024, 1024))

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_uri}},
            ],
        })

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content or ""


class GeminiClient(BaseLLMClient):
    """Google Gemini client (supports both LLM and VLM)."""

    def __init__(self, model: str = "gemini-1.5-flash", **kwargs):
        super().__init__(model, **kwargs)
        import google.generativeai as genai
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        self.genai = genai
        self.gen_model = genai.GenerativeModel(model)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def complete(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"

        response = self.gen_model.generate_content(
            full_prompt,
            generation_config=self.genai.GenerationConfig(
                temperature=self.temperature,
                max_output_tokens=self.max_tokens,
                response_mime_type="application/json",
            ),
        )
        return response.text or ""

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def complete_with_image(
        self,
        prompt: str,
        image_path: Path | str,
        system_prompt: Optional[str] = None,
    ) -> str:
        from PIL import Image

        img = Image.open(image_path)

        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"

        response = self.gen_model.generate_content(
            [full_prompt, img],
            generation_config=self.genai.GenerationConfig(
                temperature=self.temperature,
                max_output_tokens=self.max_tokens,
                response_mime_type="application/json",
            ),
        )
        return response.text or ""


class AnthropicClient(BaseLLMClient):
    """Anthropic Claude client."""

    def __init__(self, model: str = "claude-3-5-sonnet-20241022", **kwargs):
        super().__init__(model, **kwargs)
        from anthropic import Anthropic
        self.client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def complete(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system_prompt or "",
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def complete_with_image(
        self,
        prompt: str,
        image_path: Path | str,
        system_prompt: Optional[str] = None,
    ) -> str:
        from .image_utils import image_to_base64

        b64 = image_to_base64(image_path, max_size=(1024, 1024))

        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system_prompt or "",
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": b64,
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }],
        )
        return response.content[0].text


class OllamaClient(BaseLLMClient):
    """
    Ollama client for local LLM/VLM inference.
    
    Supports both text-only models (llama3, mistral, etc.) and 
    vision models (llava, llama3.2-vision, etc.).
    
    Requirements:
        - Ollama must be running locally (default: http://localhost:11434)
        - Model must be pulled first: `ollama pull <model_name>`
    
    Recommended models:
        LLM: llama3.2:3b, llama3.1:8b, mistral:7b, qwen2.5:7b
        VLM: llama3.2-vision:11b, llava:13b, llava:7b
    """

    # Vision-capable models in Ollama
    VISION_MODELS = {
        "llava", "llava:7b", "llava:13b", "llava:34b",
        "llava-llama3", "llava-phi3",
        "llama3.2-vision", "llama3.2-vision:11b", "llama3.2-vision:90b",
        "minicpm-v", "moondream", "bakllava",
    }

    def __init__(
        self,
        model: str = "llama3.2:3b",
        base_url: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(model, **kwargs)
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self._is_vision_model = self._check_vision_capability(model)

    def _check_vision_capability(self, model: str) -> bool:
        """Check if model supports vision based on known vision models."""
        model_base = model.split(":")[0].lower()
        return any(
            model_base.startswith(vm.split(":")[0]) 
            for vm in self.VISION_MODELS
        )

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def complete(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        import requests

        url = f"{self.base_url}/api/generate"
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }
        
        if system_prompt:
            payload["system"] = system_prompt

        response = requests.post(url, json=payload, timeout=300)
        response.raise_for_status()
        
        result = response.json()
        return result.get("response", "")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def complete_with_image(
        self,
        prompt: str,
        image_path: Path | str,
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Generate completion with image input for vision models.
        
        Raises:
            ValueError: If the model doesn't support vision
        """
        import requests
        from .image_utils import image_to_base64

        if not self._is_vision_model:
            raise ValueError(
                f"Model '{self.model}' does not support vision. "
                f"Use a vision model like: {', '.join(sorted(self.VISION_MODELS)[:5])}..."
            )

        # Get base64 encoded image (without data URI prefix)
        b64_image = image_to_base64(image_path, max_size=(1024, 1024))

        url = f"{self.base_url}/api/generate"
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "images": [b64_image],  # Ollama expects array of base64 images
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }
        
        if system_prompt:
            payload["system"] = system_prompt

        response = requests.post(url, json=payload, timeout=300)
        response.raise_for_status()
        
        result = response.json()
        return result.get("response", "")

    def list_models(self) -> list[dict]:
        """List all available models in the local Ollama instance."""
        import requests
        
        url = f"{self.base_url}/api/tags"
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        return response.json().get("models", [])

    def is_model_available(self) -> bool:
        """Check if the configured model is available locally."""
        try:
            models = self.list_models()
            model_names = [m.get("name", "") for m in models]
            # Check both exact match and base name match
            return any(
                self.model == name or self.model == name.split(":")[0]
                for name in model_names
            )
        except Exception:
            return False


def get_client(
    provider: str = "openai",
    model: Optional[str] = None,
    **kwargs,
) -> BaseLLMClient:
    """
    Factory function to get appropriate LLM client.
    
    Args:
        provider: "openai", "google", "anthropic", or "ollama"
        model: Model name (uses default if not specified)
        **kwargs: Additional arguments for client
    
    Returns:
        Configured LLM client
    
    Examples:
        # Cloud providers
        client = get_client("openai", "gpt-4o-mini")
        client = get_client("google", "gemini-1.5-flash")
        client = get_client("anthropic", "claude-3-5-sonnet-20241022")
        
        # Local Ollama models
        client = get_client("ollama", "llama3.2:3b")  # LLM
        client = get_client("ollama", "llama3.2-vision:11b")  # VLM
    """
    if provider == "openai":
        return OpenAIClient(model=model or "gpt-4o-mini", **kwargs)
    elif provider == "google":
        return GeminiClient(model=model or "gemini-1.5-flash", **kwargs)
    elif provider == "anthropic":
        return AnthropicClient(model=model or "claude-3-5-sonnet-20241022", **kwargs)
    elif provider == "ollama":
        return OllamaClient(model=model or "llama3.2:3b", **kwargs)
    else:
        raise ValueError(f"Unknown provider: {provider}. Supported: openai, google, anthropic, ollama")


# === JSON Extraction Helpers ===

def extract_json_from_response(text: str) -> Any:
    """
    Extract JSON from LLM response, handling markdown code blocks.
    
    Args:
        text: Raw LLM response text
    
    Returns:
        Parsed JSON data
    
    Raises:
        ValueError: If no valid JSON found
    """
    text = text.strip()

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Remove markdown code blocks
    if text.startswith("```"):
        # Remove opening ```json or ```
        text = re.sub(r"^```(?:json)?\s*\n?", "", text)
        # Remove closing ```
        text = re.sub(r"\n?```\s*$", "", text)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

    # Try to find JSON object or array
    json_patterns = [
        r"\{[\s\S]*\}",  # Object
        r"\[[\s\S]*\]",  # Array
    ]

    for pattern in json_patterns:
        match = re.search(pattern, text)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                continue

    # Try to repair truncated JSON (common with long responses)
    repaired = _try_repair_truncated_json(text)
    if repaired is not None:
        return repaired

    raise ValueError(f"Could not extract JSON from response:\n{text[:1000]}...")


def _try_repair_truncated_json(text: str) -> Any | None:
    """
    Attempt to repair truncated JSON by finding the last valid position
    and closing open brackets.
    
    This handles cases where the LLM response was cut off mid-JSON.
    """
    # Find the start of JSON
    array_start = text.find('[')
    obj_start = text.find('{')
    
    if array_start == -1 and obj_start == -1:
        return None
    
    # Determine if it's an array or object
    if array_start != -1 and (obj_start == -1 or array_start < obj_start):
        json_text = text[array_start:]
    else:
        json_text = text[obj_start:]
    
    # Find all positions of closing brackets/braces that are NOT inside strings
    # We'll try truncating at each one (from last to first) until we get valid JSON
    valid_close_positions = []
    in_string = False
    escape_next = False
    
    for i, char in enumerate(json_text):
        if escape_next:
            escape_next = False
            continue
        if char == '\\':
            escape_next = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char in '}]':
            valid_close_positions.append(i)
    
    # Try truncating at each valid close position (from last to first)
    for pos in reversed(valid_close_positions):
        truncated = json_text[:pos + 1]
        
        # Count remaining open brackets/braces
        open_brackets = 0
        open_braces = 0
        in_str = False
        esc = False
        
        for c in truncated:
            if esc:
                esc = False
                continue
            if c == '\\':
                esc = True
                continue
            if c == '"':
                in_str = not in_str
                continue
            if in_str:
                continue
            if c == '[':
                open_brackets += 1
            elif c == ']':
                open_brackets -= 1
            elif c == '{':
                open_braces += 1
            elif c == '}':
                open_braces -= 1
        
        # Close any remaining open brackets
        if open_brackets >= 0 and open_braces >= 0:
            closing = '}' * open_braces + ']' * open_brackets
            repaired = truncated + closing
            
            try:
                result = json.loads(repaired)
                # Verify we got meaningful data (at least one complete item for arrays)
                if isinstance(result, list) and len(result) > 0:
                    return result
                elif isinstance(result, dict) and len(result) > 0:
                    return result
            except json.JSONDecodeError:
                continue
    
    # Alternative approach: try to find the last complete object in an array
    # Look for pattern: }followed by , or ] (end of object in array)
    last_complete_obj = -1
    in_string = False
    escape_next = False
    brace_depth = 0
    bracket_depth = 0
    
    for i, char in enumerate(json_text):
        if escape_next:
            escape_next = False
            continue
        if char == '\\':
            escape_next = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        
        if char == '[':
            bracket_depth += 1
        elif char == ']':
            bracket_depth -= 1
        elif char == '{':
            brace_depth += 1
        elif char == '}':
            brace_depth -= 1
            # If we just closed an object at array level (bracket_depth == 1)
            if bracket_depth == 1 and brace_depth == 0:
                last_complete_obj = i
    
    if last_complete_obj > 0:
        # Truncate after the last complete object and close the array
        truncated = json_text[:last_complete_obj + 1]
        # Remove trailing comma if present
        truncated = truncated.rstrip().rstrip(',')
        repaired = truncated + ']'
        
        try:
            return json.loads(repaired)
        except json.JSONDecodeError:
            pass
    
    return None
