"""Common interface every wrapped tool (AMR-MoEGA, ProteinRAG, candidate tools) implements,
so the agent layer can call them uniformly without knowing their internals.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolResult:
    tool_name: str
    ok: bool
    output: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


class ScientificTool(ABC):
    name: str

    @abstractmethod
    def run(self, **kwargs) -> ToolResult:
        ...
