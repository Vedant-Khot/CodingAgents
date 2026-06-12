from tools.base import query_nvidia, get_nvidia_api_key

print("API Key:", bool(get_nvidia_api_key()))

prompts = [
    "Create a simple HTML button styled with CSS.",
    "Write a hello world program in Python.",
    "Create a Flutter login screen."
]

for p in prompts:
    print(f"\n--- Testing Prompt: '{p}' ---")
    sys_prompt = "You are a software developer helper. Output the code for the request."
    res = query_nvidia(p, sys_prompt)
    print("Response:")
    print(res)
    print("=" * 60)
