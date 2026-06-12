from tools.base import get_nvidia_api_key, query_nvidia

api_key = get_nvidia_api_key()
print("API Key exists:", bool(api_key))

prompt = "Write a simple hello world Python script and save to scratch/hello.py"
extract_prompt = (
    f"Identify if the following prompt asks to save, write, or create code in a specific file path or filename. "
    f"If a specific file path is requested, return ONLY that file path (e.g. 'src/index.js' or 'app.py'). "
    f"If no specific file path is mentioned, return ONLY the word 'None'.\n\n"
    f"Prompt: {prompt}"
)
sys_prompt = "You are a precise filename extractor. Output only the filename/path or 'None'."
res = query_nvidia(extract_prompt, sys_prompt).strip()
print(f"LLM returned: '{res}'")
