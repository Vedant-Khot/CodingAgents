from typing import List
from tools.base import BaseTool

class DocumentationTool(BaseTool):
    def execute(self, prompt: str, low_levels: List[str]) -> str:
        return (
            f"[DOCUMENT] [DocumentationTool Activated] generating documentation for: {low_levels}\n"
            f"   -> Document formats: README.md, Docstrings, OpenAPI specs\n"
            f"   -> Result: Created comprehensive documentation."
        )
