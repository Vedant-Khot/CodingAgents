import os
import re
from typing import List
from tools.base import BaseTool, get_nvidia_api_key, query_nvidia

class CodeGeneratorTool(BaseTool):
    def execute(self, prompt: str, low_levels: List[str]) -> str:
        api_key = get_nvidia_api_key()
        if not api_key:
            return (
                f"[CODE] [CodeGeneratorTool Activated] Error: NVIDIA_API_KEY is not set.\n"
                f"Please set your NVIDIA_API_KEY in your environment or .env file to enable LLM code generation.\n\n"
                f"--- Fallback Mock Scaffold ---\n"
                f"// Boilerplate generated for: {prompt[:50]}...\n"
                f"class Boilerplate:\n"
                f"    def __init__(self):\n"
                f"        pass"
            )

        # Detect target file path to save code
        target_path = self._detect_target_path(prompt)

        system_prompt = (
            "You are an expert code generator. Your task is to write high-quality, clean, well-structured, "
            "and complete code as requested by the user. "
            "Return the code inside a standard markdown code block specifying the programming language (e.g., ```python ... ```).\n\n"
            "Ensure there is no introductory or concluding conversational text, just the code block. "
            "Include comments in the code to explain complex parts if requested."
        )

        try:
            # Query the NVIDIA Kimi LLM
            raw_response = query_nvidia(prompt, system_prompt)
            if "NVIDIA API Error" in raw_response:
                return f"[CODE] [CodeGeneratorTool Activated] LLM Generation failed: {raw_response}"

            cleaned_code = self._clean_code(raw_response)

            if target_path:
                try:
                    # Ensure parent directories exist
                    dir_name = os.path.dirname(target_path)
                    if dir_name:
                        os.makedirs(dir_name, exist_ok=True)

                    with open(target_path, "w", encoding="utf-8") as f:
                        f.write(cleaned_code)

                    return (
                        f"[CODE] [CodeGeneratorTool Activated] Code generated and saved to: {target_path}\n"
                        f"   -> Working Directory: {os.getcwd()}\n"
                        f"   -> File size: {len(cleaned_code)} bytes\n\n"
                        f"--- Code Preview ---\n"
                        f"{raw_response}"
                    )
                except Exception as e:
                    return (
                        f"[CODE] [CodeGeneratorTool Activated] Code generated but failed to save to file '{target_path}': {str(e)}\n\n"
                        f"--- Generated Code ---\n"
                        f"{raw_response}"
                    )
            else:
                return (
                    f"[CODE] [CodeGeneratorTool Activated] Code generated successfully:\n\n"
                    f"{raw_response}"
                )

        except Exception as e:
            return f"[CODE] [CodeGeneratorTool Activated] Error during code generation: {str(e)}"

    def _detect_target_path(self, prompt: str) -> str:
        # Match patterns like: save to main.py, write to src/index.js, output to file.html
        path_match = re.search(
            r'(?:save\s+to|write\s+to|output\s+to|create\s+file|save\s+file\s+to|file\s*:\s*)\s*([a-zA-Z0-9_\-\.\/\\:]+)',
            prompt,
            re.IGNORECASE
        )
        if path_match:
            target_path = path_match.group(1).strip()
            # Double check that it contains a file extension or explicitly looks like a file name
            if '.' in os.path.basename(target_path):
                return target_path
        return ""

    def _clean_code(self, raw_code: str) -> str:
        # Find first markdown code block (e.g. ```python ... ```)
        code_block_match = re.search(r'```(?:[a-zA-Z0-9_\-\+]+)?\n(.*?)\n```', raw_code, re.DOTALL)
        if code_block_match:
            return code_block_match.group(1).strip()
            
        # Fallback to search for generic ticks
        code_block_match2 = re.search(r'```(.*?)```', raw_code, re.DOTALL)
        if code_block_match2:
            return code_block_match2.group(1).strip()
            
        return raw_code.strip()
