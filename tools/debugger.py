from typing import List
from tools.base import BaseTool

class DebuggerTool(BaseTool):
    def execute(self, prompt: str, low_levels: List[str]) -> str:
        return (
            f"[DEBUG] [DebuggerTool Activated] analyzing error report: {low_levels}\n"
            f"   -> Diagnosis: Identified runtime/syntax anomaly.\n"
            f"   -> Fix Recommendation: Inspect your parameters, clear node_modules, and update target references."
        )
