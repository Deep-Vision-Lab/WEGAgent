"""
Base Agent class that all extraction agents inherit from.
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

from rich.console import Console

from ..models.preweg import PreWEGGuide
from ..utils.llm_utils import BaseLLMClient, get_client

console = Console()


class BaseAgent(ABC):
    """
    Abstract base class for all extraction agents.
    
    Each agent is responsible for extracting one type of information
    from a pre-WEG guide (actions, tools, hands, parts, etc.)
    """

    # Override in subclass
    AGENT_NAME: str = "base"
    REQUIRES_VISION: bool = False
    DEFAULT_PROVIDER: str = "openai"
    DEFAULT_MODEL: str = "gpt-4o-mini"

    def __init__(
        self,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        verbose: bool = True,
    ):
        """
        Initialize the agent.
        
        Args:
            model: LLM/VLM model to use
            provider: API provider ("openai", "google", "anthropic")
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            verbose: Print progress to console
        """
        self.provider = provider or self.DEFAULT_PROVIDER
        self.model = model or self.DEFAULT_MODEL
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.verbose = verbose

        # Initialize client lazily
        self._client: Optional[BaseLLMClient] = None

    @property
    def client(self) -> BaseLLMClient:
        """Get or create LLM client."""
        if self._client is None:
            self._client = get_client(
                provider=self.provider,
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        return self._client

    def log(self, message: str, style: str = ""):
        """Log message if verbose mode is on."""
        if self.verbose:
            prefix = f"[bold cyan][{self.AGENT_NAME}][/bold cyan]"
            console.print(f"{prefix} {message}", style=style)

    def log_error(self, message: str):
        """Log error message."""
        prefix = f"[bold red][{self.AGENT_NAME} ERROR][/bold red]"
        console.print(f"{prefix} {message}")

    @abstractmethod
    def get_system_prompt(self) -> str:
        """Return the system prompt for this agent."""
        pass

    @abstractmethod
    def build_prompt(self, guide: PreWEGGuide, **kwargs) -> str:
        """
        Build the user prompt for extraction.
        
        Args:
            guide: Pre-WEG guide data
            **kwargs: Additional arguments (step index, etc.)
        
        Returns:
            Formatted prompt string
        """
        pass

    @abstractmethod
    def parse_response(self, response: str, guide: PreWEGGuide, **kwargs) -> Any:
        """
        Parse the LLM response into structured output.
        
        Args:
            response: Raw LLM response text
            guide: Pre-WEG guide data
            **kwargs: Additional arguments
        
        Returns:
            Parsed and validated output
        """
        pass

    def run(self, guide: PreWEGGuide, **kwargs) -> Any:
        """
        Run the agent on a guide.
        
        Args:
            guide: Pre-WEG guide data
            **kwargs: Additional arguments
        
        Returns:
            Extracted information
        """
        self.log(f"Processing guide: {guide.title} ({guide.num_steps} steps)")

        system_prompt = self.get_system_prompt()
        user_prompt = self.build_prompt(guide, **kwargs)

        self.log("Calling LLM...")
        response = self.client.complete(user_prompt, system_prompt=system_prompt)

        self.log("Parsing response...")
        result = self.parse_response(response, guide, **kwargs)

        self.log("Done!", style="green")
        return result

    def run_with_image(
        self,
        guide: PreWEGGuide,
        image_path: Path | str,
        **kwargs,
    ) -> Any:
        """
        Run the agent with an image input (for VLM agents).
        
        Args:
            guide: Pre-WEG guide data
            image_path: Path to image
            **kwargs: Additional arguments
        
        Returns:
            Extracted information
        """
        if not self.REQUIRES_VISION:
            raise ValueError(f"{self.AGENT_NAME} does not support vision input")

        self.log(f"Processing with image: {image_path}")

        system_prompt = self.get_system_prompt()
        user_prompt = self.build_prompt(guide, **kwargs)

        self.log("Calling VLM...")
        response = self.client.complete_with_image(
            user_prompt,
            image_path,
            system_prompt=system_prompt,
        )

        self.log("Parsing response...")
        result = self.parse_response(response, guide, image_path=image_path, **kwargs)

        self.log("Done!", style="green")
        return result
