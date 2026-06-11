from typing import List
from tools.base import BaseTool

class TestGeneratorTool(BaseTool):
    def execute(self, prompt: str, low_levels: List[str]) -> str:
        return (
            f"[TEST] [TestGeneratorTool Activated] creating tests for: {low_levels}\n"
            f"   -> Strategy: Generated mock unit/integration assertions\n"
            f"   -> Output: Added test suite files containing test_cases matching input functions."
        )
