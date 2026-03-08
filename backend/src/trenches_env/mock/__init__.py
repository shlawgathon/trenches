"""Mock provider module – uses OpenRouter to stand in for entity models before deployment."""

from trenches_env.mock.config import build_mock_bindings, is_mock_enabled

__all__ = ["build_mock_bindings", "is_mock_enabled"]
