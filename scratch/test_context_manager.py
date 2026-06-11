import os
import sys

# Ensure tools directory is in the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.code_reviewer import CodeReviewerTool, build_meaningful_map, flatten_meaningful_map, CodeSearchEngine, CodebaseContextManager

def main():
    print("=== Testing CodebaseContextManager integration inside CodeReviewerTool ===")
    
    # 1. Instantiate tool
    tool = CodeReviewerTool("code_reviewer", "Code Reviewer Tool")
    
    # 2. Define prompt
    prompt = "How does word splitting or tokenization work in this project?"
    
    # 3. Run tool execute (using low_levels containing 'Codebase QA')
    print(f"Running execute() for prompt: '{prompt}'")
    output = tool.execute(prompt, ["Codebase QA"])
    
    print("\n=== Tool Output ===")
    print(output)
    print("===================\n")

if __name__ == "__main__":
    main()
