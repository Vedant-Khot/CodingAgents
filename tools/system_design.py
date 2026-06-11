from typing import List
from tools.base import BaseTool

class SystemDesignTool(BaseTool):
    def execute(self, prompt: str, low_levels: List[str]) -> str:
        return (
            f"[DESIGN] [SystemDesignTool Activated] generating architecture blueprints for: {low_levels}\n"
            f"   -> System Blueprint: Scalable Architecture Block\n"
            f"   -> Components: Client -> API Gateway -> Microservices -> Cache Layer -> Database"
        )
