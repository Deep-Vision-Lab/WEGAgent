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


def get_client(
    provider: str = "openai",
    model: Optional[str] = None,
    **kwargs,
) -> BaseLLMClient:
    """
    Factory function to get appropriate LLM client.
    
    Args:
        provider: "openai", "google", or "anthropic"
        model: Model name (uses default if not specified)
        **kwargs: Additional arguments for client
    
    Returns:
        Configured LLM client
    """
    if provider == "openai":
        return OpenAIClient(model=model or "gpt-4o-mini", **kwargs)
    elif provider == "google":
        return GeminiClient(model=model or "gemini-1.5-flash", **kwargs)
    elif provider == "anthropic":
        return AnthropicClient(model=model or "claude-3-5-sonnet-20241022", **kwargs)
    else:
        raise ValueError(f"Unknown provider: {provider}")


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
