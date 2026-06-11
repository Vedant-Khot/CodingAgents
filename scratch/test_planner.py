import sys
import os

# Ensure tools directory is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from router import ToolOrchestrator

def main():
    print("=== Testing Agentic Planner on Multi-Step Request ===")
    orchestrator = ToolOrchestrator()
    prompt = "Write a python function to compute TF-IDF, write unit tests for it, and then document the code."
    
    print(f"Prompt: \"{prompt}\"\n")
    res = orchestrator.process_and_execute(prompt)
    
    print("--- PLAN GENERATED ---")
    for i, step in enumerate(res.get("plan", []), 1):
        print(f"Step {i}: {step['description']}")
        print(f"  Tool:   {step['tool']}")
        print(f"  Prompt: {step['prompt']}")
        
    print("\n--- EXECUTION DETAILS ---")
    print(f"Dispatched Tools: {res.get('dispatched_tools')}")
    print(f"Is Fallback?:     {res.get('is_fallback')}")
    
    print("\n--- DETAILED STEP OUTPUTS ---")
    for i, step in enumerate(res.get("steps", []), 1):
        print(f"\n[Step {i}] {step['tool']} ({step['intent']}):")
        print(step["output"])
        print("-" * 50)

if __name__ == "__main__":
    main()
