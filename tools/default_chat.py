from typing import List
from tools.base import BaseTool

class DefaultAgentChat(BaseTool):
    def execute(self, prompt: str, low_levels: List[str]) -> str:
        return (
            f"[CHAT] [DefaultAgentChat Activated]\n"
            f"   -> Chat Context: {low_levels}\n"
            f"   -> Response: Hello! How can I assist you with your project today?"
        )
