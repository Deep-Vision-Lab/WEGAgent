"""
WEGv2 Configuration Settings
"""
import os
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Global application settings loaded from environment variables."""

    # === API Keys ===
    openai_api_key: str = Field(default="", env="OPENAI_API_KEY")
    google_api_key: str = Field(default="", env="GOOGLE_API_KEY")
    anthropic_api_key: str = Field(default="", env="ANTHROPIC_API_KEY")

    # === Ollama Configuration ===
    ollama_base_url: str = Field(default="http://localhost:11434", env="OLLAMA_BASE_URL")
    use_ollama: bool = Field(default=False, env="USE_OLLAMA")

    # === Model Configuration ===
    default_llm_model: str = Field(default="gpt-4o-mini", env="DEFAULT_LLM_MODEL")
    default_vlm_model: str = Field(default="gemini-1.5-flash", env="DEFAULT_VLM_MODEL")
    
    # === Ollama Model Configuration (used when USE_OLLAMA=true) ===
    ollama_llm_model: str = Field(default="llama3.2:3b", env="OLLAMA_LLM_MODEL")
    ollama_vlm_model: str = Field(default="llama3.2-vision:11b", env="OLLAMA_VLM_MODEL")

    # === Pipeline Settings ===
    max_retries: int = Field(default=3, env="MAX_RETRIES")
    rate_limit_rpm: int = Field(default=60, env="RATE_LIMIT_RPM")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO", env="LOG_LEVEL"
    )

    # === Paths ===
    project_root: Path = Path(__file__).parent.parent
    data_dir: Path = project_root / "data"
    preweg_dir: Path = data_dir / "preweg"
    weg_output_dir: Path = data_dir / "weg_output"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

    def ensure_dirs(self) -> None:
        """Create necessary directories if they don't exist."""
        self.preweg_dir.mkdir(parents=True, exist_ok=True)
        self.weg_output_dir.mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()


# === Model Configurations ===

LLM_MODELS = {
    "gpt-4o": {"provider": "openai", "context_window": 128000, "cost_per_1k_input": 0.005},
    "gpt-4o-mini": {"provider": "openai", "context_window": 128000, "cost_per_1k_input": 0.00015},
    "claude-3-5-sonnet": {"provider": "anthropic", "context_window": 200000, "cost_per_1k_input": 0.003},
    # Ollama Local LLM Models (cost = 0, runs locally)
    "llama3.2:3b": {"provider": "ollama", "context_window": 128000, "cost_per_1k_input": 0.0},
    "llama3.1:8b": {"provider": "ollama", "context_window": 128000, "cost_per_1k_input": 0.0},
    "mistral:7b": {"provider": "ollama", "context_window": 32000, "cost_per_1k_input": 0.0},
    "qwen2.5:7b": {"provider": "ollama", "context_window": 128000, "cost_per_1k_input": 0.0},
    "phi3:mini": {"provider": "ollama", "context_window": 128000, "cost_per_1k_input": 0.0},
}

VLM_MODELS = {
    "gemini-1.5-flash": {"provider": "google", "context_window": 1000000, "cost_per_1k_input": 0.000075},
    "gemini-1.5-pro": {"provider": "google", "context_window": 2000000, "cost_per_1k_input": 0.00125},
    "gpt-4o": {"provider": "openai", "context_window": 128000, "cost_per_1k_input": 0.005},
    # Ollama Local VLM Models (cost = 0, runs locally)
    "llama3.2-vision:11b": {"provider": "ollama", "context_window": 128000, "cost_per_1k_input": 0.0},
    "llava:7b": {"provider": "ollama", "context_window": 4096, "cost_per_1k_input": 0.0},
    "llava:13b": {"provider": "ollama", "context_window": 4096, "cost_per_1k_input": 0.0},
    "minicpm-v": {"provider": "ollama", "context_window": 4096, "cost_per_1k_input": 0.0},
}


# === Agent-specific prompts will be loaded from here ===

AGENT_CONFIGS = {
    "action": {
        "model_type": "vlm",  # Uses vision
        "default_model": "gemini-1.5-flash",
        "temperature": 0.1,
        "max_tokens": 4096,
    },
    "tool": {
        "model_type": "llm",
        "default_model": "gpt-4o-mini",
        "temperature": 0.0,
        "max_tokens": 2048,
    },
    "hands": {
        "model_type": "llm",
        "default_model": "gpt-4o-mini",
        "temperature": 0.0,
        "max_tokens": 1024,
    },
    "part_text": {
        "model_type": "llm",
        "default_model": "gpt-4o-mini",
        "temperature": 0.0,
        "max_tokens": 2048,
    },
    "part_vision": {
        "model_type": "vlm",  # Uses vision for bbox
        "default_model": "gemini-1.5-flash",
        "temperature": 0.1,
        "max_tokens": 4096,
    },
}
