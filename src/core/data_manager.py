"""
Data Manager - Centralized service for managing analysis results, caching, and data operations.
"""
import json
from typing import Dict, Any, Optional

class DataManager:
    """Centralized service for managing analysis context operations."""

    def __init__(self, max_cache_size: int = 10):
        self._context: Optional[Dict[str, Any]] = None
            
    _CONTEXT_FORMAT_HINT = (
        'Expected JSON format:\n'
        '{\n'
        '  "outcomes": ["outcome1", "outcome2"],\n'
        '  "groups": ["group1", "group2"],\n'
        '  "units": {"outcome1": "kg"},\n'
        '  "error_bar_type": "SE"\n'
        '}'
    )

    def load_context(self, path: str) -> Dict[str, Any]:
        """Load and validate a context-of-interest JSON file."""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                ctx = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Invalid JSON in context file: {e}\n\n"
                f"{self._CONTEXT_FORMAT_HINT}"
            ) from e
        if not isinstance(ctx, dict):
            raise ValueError(
                f"Context file must contain a JSON object (dict), "
                f"got {type(ctx).__name__}.\n\n"
                f"{self._CONTEXT_FORMAT_HINT}"
            )
        if not isinstance(ctx.get('outcomes'), list):
            ctx['outcomes'] = []
        if not isinstance(ctx.get('groups'), list):
            ctx['groups'] = []
        self._context = ctx
        return ctx

    def get_context(self) -> Optional[Dict[str, Any]]:
        """Return the loaded context-of-interest, or None."""
        return self._context

    def clear_context(self):
        """Clear the loaded context."""
        self._context = None
                    