"""Local, open-source LLM backend via Ollama (no API key — see docs/design-principles.md's
open-source-model decision). Talks to Ollama's local REST API directly (no extra SDK
dependency needed)."""
from __future__ import annotations

import json

import requests


class OllamaBackend:
    def __init__(self, config: dict):
        cfg = config["agent"]["ollama"]
        self.host = cfg["host"]
        self.model = cfg["model"]
        self.temperature = cfg["temperature"]
        self.timeout_s = cfg["timeout_s"]

    def ping(self) -> bool:
        try:
            resp = requests.get(f"{self.host}/api/tags", timeout=5)
            return resp.ok
        except requests.RequestException:
            return False

    def chat_json(self, system: str, user: str) -> dict:
        """Sends a chat turn with JSON-mode forced output; returns the parsed dict.
        Raises on any failure — callers must handle a safe fallback (see policy_llm.py)."""
        resp = requests.post(
            f"{self.host}/api/chat",
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "format": "json",
                "stream": False,
                "options": {"temperature": self.temperature},
            },
            timeout=self.timeout_s,
        )
        resp.raise_for_status()
        content = resp.json()["message"]["content"]
        return json.loads(content)
