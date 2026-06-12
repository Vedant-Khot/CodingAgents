import urllib.request
import json
import os
from tools.base import get_nvidia_api_key

api_key = get_nvidia_api_key()
url = "https://integrate.api.nvidia.com/v1/chat/completions"

models = [
    "meta/llama-3.1-8b-instruct",
    "meta/llama3-8b-instruct",
    "mistralai/mistral-7b-instruct-v0.3",
    "meta/llama-3.2-3b-instruct"
]

prompt = "Create a simple HTML button styled with CSS. Return ONLY the code."
sys_prompt = "You are a code generator."

for model in models:
    print(f"\n--- Testing Model: {model} ---")
    data = {
        "model": model,
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2,
        "max_tokens": 1024,
        "stream": False
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    req = urllib.request.Request(url, data=json.dumps(data).encode("utf-8"), headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            res = json.loads(r.read().decode("utf-8"))
            print("Response:")
            print(res["choices"][0]["message"]["content"][:300])
    except Exception as e:
        print(f"Error: {e}")
