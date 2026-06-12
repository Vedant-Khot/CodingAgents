import os
import re
from typing import List
from tools.base import BaseTool, get_nvidia_api_key, query_nvidia

class CodeGeneratorTool(BaseTool):
    def execute(self, prompt: str, low_levels: List[str]) -> str:
        # 1. Parse target file path if requested
        target_file = self._extract_target_file(prompt)
        
        api_key = get_nvidia_api_key()
        if not api_key:
            fallback_code = (
                f"class Handler:\n"
                f"    def __init__(self):\n"
                f"        # Fallback dummy logic for low levels: {low_levels}\n"
                f"        pass"
            )
            if target_file:
                try:
                    parent = os.path.dirname(target_file)
                    if parent:
                        os.makedirs(parent, exist_ok=True)
                    with open(target_file, "w", encoding="utf-8") as f:
                        f.write(fallback_code)
                    return (
                        f"[CODE] [CodeGeneratorTool Activated] Warning: NVIDIA_API_KEY is not set. "
                        f"Saved dummy fallback code to {target_file}.\n"
                        f"Code:\n{fallback_code}"
                    )
                except Exception as e:
                    return f"[CODE] [CodeGeneratorTool Activated] Warning: NVIDIA_API_KEY is not set. Failed to write fallback file: {str(e)}"
            return (
                f"[CODE] [CodeGeneratorTool Activated] Warning: NVIDIA_API_KEY is not set. "
                f"Returning dummy fallback code.\n"
                f"Code:\n{fallback_code}"
            )
            
        # 2. Call LLM to generate the code
        sys_prompt = (
            "You are an expert code generator. Generate clean, correct, and well-commented code "
            "based on the user's request. Output ONLY the code itself, without any conversational "
            "intro or outro. If you use markdown code blocks, make sure they are formatted correctly "
            "with language specifiers."
        )
        
        prompt_with_context = (
            f"Generate code for: \"{prompt}\".\n"
            f"Requirements/Context: {low_levels}."
        )
        
        generated = query_nvidia(prompt_with_context, sys_prompt)
        if "NVIDIA API Error" in generated:
            fallback_code = (
                f"# Fallback code generated due to API Error: {generated}\n"
                f"class Handler:\n"
                f"    def __init__(self):\n"
                f"        # Fallback dummy logic for low levels: {low_levels}\n"
                f"        pass"
            )
            if target_file:
                try:
                    target_abs = os.path.abspath(target_file)
                    parent = os.path.dirname(target_abs)
                    if parent:
                        os.makedirs(parent, exist_ok=True)
                    with open(target_abs, "w", encoding="utf-8") as f:
                        f.write(fallback_code)
                    return (
                        f"[CODE] [CodeGeneratorTool Activated] API Error: {generated}. "
                        f"Saved fallback code to {target_file}.\n"
                        f"Code:\n{fallback_code}"
                    )
                except Exception as e:
                    return f"[CODE] [CodeGeneratorTool Activated] API Error: {generated}. Failed to write fallback file: {str(e)}"
            return f"[CODE] [CodeGeneratorTool Activated] API Error: {generated}"
            
        raw_code = self._clean_code_output(generated)
        
        # 3. Save to file if target path specified
        if target_file:
            try:
                # Make path relative to workspace or absolute if specified absolute
                target_abs = os.path.abspath(target_file)
                parent = os.path.dirname(target_abs)
                if parent:
                    os.makedirs(parent, exist_ok=True)
                with open(target_abs, "w", encoding="utf-8") as f:
                    f.write(raw_code)
                    
                return (
                    f"[CODE] [CodeGeneratorTool Activated] Code generated successfully:\n"
                    f"   -> Saved to File: {target_file}\n"
                    f"   -> Absolute Path: {target_abs}\n\n"
                    f"--- File Contents ---\n"
                    f"{raw_code}"
                )
            except Exception as e:
                return (
                    f"[CODE] [CodeGeneratorTool Activated] Code generated successfully but failed to write file:\n"
                    f"   -> Target: {target_file}\n"
                    f"   -> Error: {str(e)}\n\n"
                    f"--- Generated Code ---\n"
                    f"{raw_code}"
                )
        else:
            return (
                f"[CODE] [CodeGeneratorTool Activated] Code generated successfully (no file path requested):\n\n"
                f"--- Generated Code ---\n"
                f"{raw_code}"
            )

    def _extract_target_file(self, prompt: str) -> str:
        # Regex-based extraction first (fast and precise)
        patterns = [
            r'(?:save|write|output|create|file|filename|filepath)(?:\s+to|\s+at)?\s*:\s*[\'"`]?([a-zA-Z0-9_\-\.\/\\:]+)[\'"`]?',
            r'(?:save|write|output|create|create\s+file)\s*(?:\s+to|\s+at)?\s*[\'"`]?([a-zA-Z0-9_\-\/\\:]+\.[a-zA-Z0-9]+|[a-zA-Z0-9_\-]+\/[a-zA-Z0-9_\-\.]+|[a-zA-Z0-9_\-]+\\[a-zA-Z0-9_\-\.]+)[\'"`]?',
            r'\b(?:into|in)\s+[\'"`]?([a-zA-Z0-9_\-\/\\:]+\.[a-zA-Z0-9]+|[a-zA-Z0-9_\-]+\/[a-zA-Z0-9_\-\.]+|[a-zA-Z0-9_\-]+\\[a-zA-Z0-9_\-\.]+)[\'"`]?'
        ]
        for pattern in patterns:
            match = re.search(pattern, prompt, re.IGNORECASE)
            if match:
                candidate = match.group(1).strip()
                # Confirm it looks like a file path
                if "." in candidate or "/" in candidate or "\\" in candidate:
                    return candidate
                    
        # LLM extraction fallback if API key is set
        api_key = get_nvidia_api_key()
        if api_key:
            extract_prompt = (
                f"Identify if the following prompt asks to save, write, or create code in a specific file path or filename. "
                f"If a specific file path is requested, return ONLY that file path (e.g. 'src/index.js' or 'app.py'). "
                f"If no specific file path is mentioned, return ONLY the word 'None'.\n\n"
                f"Prompt: {prompt}"
            )
            sys_prompt = "You are a precise filename extractor. Output only the filename/path or 'None'."
            try:
                res = query_nvidia(extract_prompt, sys_prompt).strip()
                if res and res.lower() != "none" and "nvidia api error" not in res.lower() and len(res) < 200:
                    return res.strip("'\"` ")
            except Exception:
                pass
                
        return ""

    def _clean_code_output(self, code: str) -> str:
        code_strip = code.strip()
        
        # 1. Match code blocks with language specifiers
        match = re.search(r'```[a-zA-Z0-9_\-\+\#]+\n(.*?)\n```', code_strip, re.DOTALL)
        if match:
            return match.group(1).strip()
            
        # 2. Match standard code blocks without language specifier
        match2 = re.search(r'```\n?(.*?)\n?```', code_strip, re.DOTALL)
        if match2:
            return match2.group(1).strip()
            
        return code_strip
