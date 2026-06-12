import subprocess
import os
import re
from typing import List
from tools.base import BaseTool, get_nvidia_api_key, query_nvidia

class TerminalExecutorTool(BaseTool):
    def execute(self, prompt: str, low_levels: List[str]) -> str:
        # Extract command
        command = self._extract_command(prompt)
        if not command:
            return (
                f"[EXECUTE] [TerminalExecutorTool Activated] Could not identify or extract command from prompt.\n"
                f"Prompt: {prompt}"
            )
            
        try:
            # Run the command in the shell
            # Set timeout to 30 seconds
            # Use subprocess.DEVNULL for stdin to avoid blocking on prompts
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                stdin=subprocess.DEVNULL,
                timeout=30
            )
            
            # Format the output beautifully
            output = (
                f"[EXECUTE] [TerminalExecutorTool Activated] Executed command: {command}\n"
                f"   -> Working Directory: {os.getcwd()}\n"
                f"   -> Exit Code: {result.returncode}\n"
            )
            
            if result.stdout.strip():
                output += f"   -> Output:\n{result.stdout}\n"
            else:
                output += "   -> Output: [Empty]\n"
                
            if result.stderr.strip():
                output += f"   -> Error/Stderr:\n{result.stderr}\n"
                
            return output
            
        except subprocess.TimeoutExpired:
            return (
                f"[EXECUTE] [TerminalExecutorTool Activated] Execution timed out (limit: 30s) for command: {command}"
            )
        except Exception as e:
            return (
                f"[EXECUTE] [TerminalExecutorTool Activated] Failed to run command: {command}\n"
                f"   -> Error details: {str(e)}"
            )

    def _extract_command(self, prompt: str) -> str:
        prompt_strip = prompt.strip()
        
        # 1. Look for markdown code blocks (e.g. ```bash ... ```)
        code_block_match = re.search(r'```(?:[a-zA-Z0-9_\-\+]+)?\n(.*?)\n```', prompt_strip, re.DOTALL)
        if code_block_match:
            return code_block_match.group(1).strip()
            
        # 2. Look for any code block without explicit language/newline prefix
        code_block_match2 = re.search(r'```(.*?)```', prompt_strip, re.DOTALL)
        if code_block_match2:
            return code_block_match2.group(1).strip()
            
        # 3. Look for inline backticks
        inline_match = re.search(r'`([^`]+)`', prompt_strip)
        if inline_match:
            return inline_match.group(1).strip()
            
        # 4. Use LLM if API key exists
        api_key = get_nvidia_api_key()
        if api_key:
            extract_prompt = (
                f"Extract the single exact terminal command to execute from the following request. "
                f"Return ONLY the plain command string without any surrounding quotes, markdown formatting, or extra explanation.\n\n"
                f"Request: {prompt}"
            )
            sys_prompt = "You are a precise command extractor. Output only the command itself, nothing else."
            command = query_nvidia(extract_prompt, sys_prompt).strip()
            if command and "NVIDIA API Error" not in command and len(command) < 500:
                # Clean up any potential markdown wrappers the model output anyway
                if command.startswith("`") and command.endswith("`"):
                    command = command.strip("`").strip()
                if command.startswith("```") and command.endswith("```"):
                    lines = command.splitlines()
                    if len(lines) >= 3:
                        command = "\n".join(lines[1:-1]).strip()
                return command
                
        # 5. Clean up prompt prefix and fall back
        cleaned = prompt_strip
        # Remove common request prefixes case-insensitively
        prefixes = [
            r'^run\s+command\s*:\s*',
            r'^run\s+command\s+',
            r'^run\s+the\s+command\s+',
            r'^run\s+',
            r'^execute\s+command\s*:\s*',
            r'^execute\s+command\s+',
            r'^execute\s+the\s+command\s+',
            r'^execute\s+',
            r'^terminal\s*:\s*',
            r'^shell\s*:\s*'
        ]
        for pattern in prefixes:
            match = re.match(pattern, cleaned, re.IGNORECASE)
            if match:
                cleaned = cleaned[match.end():].strip()
                break
                
        return cleaned
