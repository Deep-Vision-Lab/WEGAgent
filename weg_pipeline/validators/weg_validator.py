"""
WEG Validator - Validates WEG documents for correctness and completeness.
"""
from typing import Any

from pydantic import ValidationError

from ..models.weg import WEGDocument


class ValidationResult:
    """Result of WEG validation."""

    def __init__(self):
        self.is_valid = True
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def add_error(self, message: str):
        self.is_valid = False
        self.errors.append(message)

    def add_warning(self, message: str):
        self.warnings.append(message)

    def __str__(self) -> str:
        if self.is_valid:
            status = "✓ Valid"
        else:
            status = "✗ Invalid"

        lines = [status]
        if self.errors:
            lines.append("Errors:")
            for err in self.errors:
                lines.append(f"  - {err}")
        if self.warnings:
            lines.append("Warnings:")
            for warn in self.warnings:
                lines.append(f"  - {warn}")

        return "\n".join(lines)


def validate_weg(weg: dict[str, Any]) -> ValidationResult:
    """
    Validate a WEG document for correctness and completeness.
    
    Checks:
    1. Schema validation (Pydantic)
    2. Required fields present
    3. Step ordering
    4. Bounding box validity
    5. Consistency checks
    
    Args:
        weg: WEG document as dictionary
    
    Returns:
        ValidationResult with is_valid flag and any errors/warnings
    """
    result = ValidationResult()

    # 1. Schema validation
    try:
        doc = WEGDocument(**weg)
    except ValidationError as e:
        result.add_error(f"Schema validation failed: {e}")
        return result

    # 2. Check header
    if not doc.header.title:
        result.add_warning("Missing guide title in header")

    # 3. Check steps
    if not doc.steps:
        result.add_error("No steps in WEG document")
        return result

    # 4. Check step ordering
    step_ids = [s.step_id for s in doc.steps]
    if step_ids != sorted(step_ids):
        result.add_warning("Steps are not in order")

    if len(step_ids) != len(set(step_ids)):
        result.add_error("Duplicate step IDs found")

    # 5. Check each step
    for step in doc.steps:
        step_prefix = f"Step {step.step_id}"

        # Check description
        if not step.description:
            result.add_warning(f"{step_prefix}: Empty description")

        # Check actions
        if not step.actions:
            result.add_warning(f"{step_prefix}: No actions extracted")

        # Validate bounding boxes for primary_part if present
        if step.primary_part and step.primary_part.bbox:
            bbox = step.primary_part.bbox
            if bbox.x1 >= bbox.x2 or bbox.y1 >= bbox.y2:
                result.add_error(f"{step_prefix}: Invalid bounding box dimensions for primary_part")
            if bbox.x1 < 0 or bbox.y1 < 0:
                result.add_warning(f"{step_prefix}: primary_part bounding box has negative coordinates")

        # Validate parts_all and collect valid part_ids
        valid_part_ids = set()
        for part in step.parts_all:
            valid_part_ids.add(part.part_id)
            if part.bbox:
                bbox = part.bbox
                if bbox.x1 >= bbox.x2 or bbox.y1 >= bbox.y2:
                    result.add_warning(f"{step_prefix}: Part '{part.name}' (id={part.part_id}) has invalid bbox")

        # Check hands value and part_id references in action quadruples
        for idx, quadruple in enumerate(step.action_quadruples):
            if quadruple.hands not in (0, 1, 2):
                result.add_error(f"{step_prefix}: Action {idx+1} has invalid hands value: {quadruple.hands}")
            
            # Validate part_id reference if present
            if quadruple.part_id is not None and quadruple.part_id not in valid_part_ids:
                result.add_warning(
                    f"{step_prefix}: Action {idx+1} references part_id={quadruple.part_id} "
                    f"but it doesn't exist in parts_all"
                )

    # 6. Check toolbox consistency (tools are now in action_quadruples)
    used_tools = set()
    for step in doc.steps:
        for quadruple in step.action_quadruples:
            if quadruple.tool:
                used_tools.add(quadruple.tool.lower())

    toolbox_lower = {t.lower() for t in doc.header.toolbox}
    extra_tools = used_tools - toolbox_lower
    if extra_tools:
        result.add_warning(f"Tools used but not in toolbox: {extra_tools}")

    return result
