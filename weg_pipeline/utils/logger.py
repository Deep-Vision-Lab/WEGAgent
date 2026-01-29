"""
Logging system for WEG Pipeline.

Provides detailed logging of agent activities, reviewer feedback,
and inter-agent communication for observability and debugging.
"""
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from dataclasses import dataclass, field, asdict
from enum import Enum


class LogLevel(Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    AGENT_ACTION = "AGENT_ACTION"
    REVIEWER_FEEDBACK = "REVIEWER_FEEDBACK"
    AGENT_RESPONSE = "AGENT_RESPONSE"


@dataclass
class LogEntry:
    """Single log entry."""
    timestamp: str
    level: str
    agent: str
    step_index: Optional[int]
    message: str
    details: dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    def to_string(self) -> str:
        step_str = f"[Step {self.step_index}]" if self.step_index else ""
        details_str = f" | {json.dumps(self.details)}" if self.details else ""
        return f"[{self.timestamp}] [{self.level}] [{self.agent}] {step_str} {self.message}{details_str}"


class PipelineLogger:
    """
    Centralized logger for the WEG pipeline.
    
    Tracks all agent activities, reviewer feedback, and responses
    for complete observability of the pipeline execution.
    """
    
    def __init__(
        self,
        guide_id: int,
        log_dir: Optional[Path] = None,
        console_output: bool = True,
    ):
        self.guide_id = guide_id
        self.log_dir = log_dir or Path("logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.console_output = console_output
        
        # Create log file path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"pipeline_{guide_id}_{timestamp}.log"
        self.json_log_file = self.log_dir / f"pipeline_{guide_id}_{timestamp}.json"
        
        # In-memory log storage
        self.entries: list[LogEntry] = []
        
        # Agent-specific logs
        self.agent_logs: dict[str, list[LogEntry]] = {}
        
        # Reviewer feedback tracking
        self.feedback_history: list[dict] = []
        
        # Step-level tracking
        self.step_logs: dict[int, list[LogEntry]] = {}
        
        # Initialize file
        self._write_header()
    
    def _write_header(self):
        """Write log header."""
        header = f"""
{'='*80}
WEG Pipeline Log
Guide ID: {self.guide_id}
Started: {datetime.now().isoformat()}
{'='*80}
"""
        with open(self.log_file, "w") as f:
            f.write(header)
    
    def _get_timestamp(self) -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    
    def log(
        self,
        level: LogLevel,
        agent: str,
        message: str,
        step_index: Optional[int] = None,
        details: Optional[dict] = None,
    ):
        """Log an entry."""
        entry = LogEntry(
            timestamp=self._get_timestamp(),
            level=level.value,
            agent=agent,
            step_index=step_index,
            message=message,
            details=details or {},
        )
        
        # Store in memory
        self.entries.append(entry)
        
        # Agent-specific storage
        if agent not in self.agent_logs:
            self.agent_logs[agent] = []
        self.agent_logs[agent].append(entry)
        
        # Step-specific storage
        if step_index is not None:
            if step_index not in self.step_logs:
                self.step_logs[step_index] = []
            self.step_logs[step_index].append(entry)
        
        # Write to file
        with open(self.log_file, "a") as f:
            f.write(entry.to_string() + "\n")
        
        # Console output
        if self.console_output:
            self._print_entry(entry)
    
    def _print_entry(self, entry: LogEntry):
        """Print entry to console with colors."""
        from rich.console import Console
        console = Console()
        
        color_map = {
            "DEBUG": "dim",
            "INFO": "blue",
            "WARNING": "yellow",
            "ERROR": "red",
            "AGENT_ACTION": "cyan",
            "REVIEWER_FEEDBACK": "magenta",
            "AGENT_RESPONSE": "green",
        }
        color = color_map.get(entry.level, "white")
        
        step_str = f"[Step {entry.step_index}]" if entry.step_index else ""
        console.print(f"[{color}][{entry.agent}] {step_str} {entry.message}[/{color}]")
    
    # === Convenience methods ===
    
    def agent_start(self, agent: str, step_index: Optional[int] = None):
        """Log agent starting work."""
        self.log(
            LogLevel.AGENT_ACTION,
            agent,
            f"Starting processing",
            step_index=step_index,
        )
    
    def agent_complete(self, agent: str, step_index: Optional[int] = None, result_summary: str = ""):
        """Log agent completing work."""
        self.log(
            LogLevel.AGENT_ACTION,
            agent,
            f"Completed: {result_summary}",
            step_index=step_index,
        )
    
    def agent_error(self, agent: str, error: str, step_index: Optional[int] = None):
        """Log agent error."""
        self.log(
            LogLevel.ERROR,
            agent,
            f"Error: {error}",
            step_index=step_index,
        )
    
    def agent_reasoning(
        self,
        agent: str,
        step_index: int,
        input_text: str,
        reasoning: str,
        output: Any,
    ):
        """Log agent's reasoning process."""
        self.log(
            LogLevel.AGENT_ACTION,
            agent,
            f"Reasoning for step",
            step_index=step_index,
            details={
                "input": input_text[:200] + "..." if len(input_text) > 200 else input_text,
                "reasoning": reasoning,
                "output": str(output)[:500],
            },
        )
    
    def reviewer_feedback(
        self,
        step_index: int,
        target_agent: str,
        issue: str,
        suggestion: str,
        severity: str = "warning",
    ):
        """Log reviewer feedback to an agent."""
        feedback = {
            "step_index": step_index,
            "target_agent": target_agent,
            "issue": issue,
            "suggestion": suggestion,
            "severity": severity,
            "timestamp": self._get_timestamp(),
            "resolved": False,
        }
        self.feedback_history.append(feedback)
        
        self.log(
            LogLevel.REVIEWER_FEEDBACK,
            "Reviewer",
            f"Feedback to {target_agent}: {issue}",
            step_index=step_index,
            details={"suggestion": suggestion, "severity": severity},
        )
    
    def agent_response_to_feedback(
        self,
        agent: str,
        step_index: int,
        original_value: Any,
        corrected_value: Any,
        explanation: str,
    ):
        """Log agent's response to reviewer feedback."""
        self.log(
            LogLevel.AGENT_RESPONSE,
            agent,
            f"Corrected based on feedback: {explanation}",
            step_index=step_index,
            details={
                "original": str(original_value),
                "corrected": str(corrected_value),
            },
        )
        
        # Mark feedback as resolved
        for fb in reversed(self.feedback_history):
            if fb["step_index"] == step_index and fb["target_agent"] == agent and not fb["resolved"]:
                fb["resolved"] = True
                fb["resolution"] = {
                    "corrected_value": str(corrected_value),
                    "explanation": explanation,
                }
                break
    
    def save_json_log(self, path: Optional[str] = None):
        """Save complete log as JSON for analysis.
        
        Args:
            path: Optional path to save the log. If None, uses default json_log_file.
        """
        output_path = path or self.json_log_file
        
        log_data = {
            "guide_id": self.guide_id,
            "timestamp": datetime.now().isoformat(),
            "entries": [e.to_dict() for e in self.entries],
            "agent_logs": {
                agent: [e.to_dict() for e in logs]
                for agent, logs in self.agent_logs.items()
            },
            "feedback_history": self.feedback_history,
            "summary": self.get_summary(),
        }
        
        with open(output_path, "w") as f:
            json.dump(log_data, f, indent=2)
        
        return output_path
    
    def get_summary(self) -> dict:
        """Get summary statistics."""
        return {
            "total_entries": len(self.entries),
            "agents_involved": list(self.agent_logs.keys()),
            "steps_processed": list(self.step_logs.keys()),
            "total_feedback": len(self.feedback_history),
            "resolved_feedback": sum(1 for fb in self.feedback_history if fb.get("resolved")),
            "errors": sum(1 for e in self.entries if e.level == "ERROR"),
            "warnings": sum(1 for e in self.entries if e.level == "WARNING"),
        }
    
    def get_step_history(self, step_index: int) -> list[LogEntry]:
        """Get all logs for a specific step."""
        return self.step_logs.get(step_index, [])
    
    def get_unresolved_feedback(self) -> list[dict]:
        """Get feedback that hasn't been resolved."""
        return [fb for fb in self.feedback_history if not fb.get("resolved")]


# Global logger instance (set per pipeline run)
_current_logger: Optional[PipelineLogger] = None


def get_logger() -> Optional[PipelineLogger]:
    """Get current pipeline logger."""
    return _current_logger


def set_logger(logger: PipelineLogger):
    """Set current pipeline logger."""
    global _current_logger
    _current_logger = logger


def create_logger(guide_id: int, log_dir: Optional[Path] = None) -> PipelineLogger:
    """Create and set a new pipeline logger."""
    logger = PipelineLogger(guide_id, log_dir)
    set_logger(logger)
    return logger
