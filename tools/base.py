import urllib.request
import urllib.parse
import re
import os
import ast
import json
import math
from collections import Counter
from typing import List, Dict

# ==========================================
# LOCAL QA PIPELINE HELPERS
# ==========================================

def stem(word: str) -> str:
    suffixes = ('ization', 'ize', 'ing', 'ly', 'ed', 'es', 's', 'er', 'ment')
    for suffix in suffixes:
        if word.endswith(suffix) and len(word) > len(suffix) + 2:
            return word[:-len(suffix)]
    return word

def tokenize(text: str) -> list:
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    tokens = text.split()
    stop_words = {
        'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are', 'arent', 'as', 'at',
        'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', 'can', 'cant', 'cannot',
        'could', 'couldnt', 'did', 'didnt', 'do', 'does', 'doesnt', 'doing', 'dont', 'down', 'during', 'each', 'few',
        'for', 'from', 'further', 'had', 'hadnt', 'has', 'hasnt', 'have', 'havent', 'having', 'he', 'hed', 'hell',
        'hes', 'her', 'here', 'heres', 'hers', 'herself', 'him', 'himself', 'his', 'how', 'hows', 'i', 'id', 'ill',
        'im', 'ive', 'if', 'in', 'into', 'is', 'isnt', 'it', 'its', 'itself', 'lets', 'me', 'more', 'most', 'mustnt',
        'my', 'myself', 'no', 'nor', 'not', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'ought', 'our', 'ours',
        'ourselves', 'out', 'over', 'own', 'same', 'shant', 'she', 'shed', 'shell', 'shes', 'should', 'shouldnt',
        'so', 'some', 'such', 'than', 'that', 'thats', 'the', 'their', 'theirs', 'them', 'themselves', 'then',
        'there', 'theres', 'these', 'they', 'theyd', 'theyll', 'theyre', 'theyve', 'this', 'those', 'through',
        'to', 'too', 'under', 'until', 'up', 'very', 'was', 'wasnt', 'we', 'wed', 'well', 'were', 'weve', 'werent',
        'what', 'whats', 'when', 'whens', 'where', 'wheres', 'which', 'while', 'who', 'whos', 'whom', 'why', 'whys',
        'with', 'wont', 'would', 'wouldnt', 'you', 'youd', 'youll', 'youre', 'youve', 'your', 'yours', 'yourself',
        'yourselves'
    }
    return [stem(t) for t in tokens if t not in stop_words and len(t) > 1]

def get_nvidia_api_key() -> str:
    key = os.environ.get("NVIDIA_API_KEY")
    if key:
        return key
    try:
        # Look for .env in current directory or parent directories
        if os.path.exists(".env"):
            with open(".env", "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("NVIDIA_API_KEY="):
                        return line.split("=", 1)[1].strip().strip('"').strip("'")
    except Exception:
        pass
    return ""

def query_nvidia(prompt: str, system_prompt: str = "") -> str:
    api_key = get_nvidia_api_key()
    if not api_key:
        return "NVIDIA API Error: NVIDIA_API_KEY is not set."
        
    try:
        url = "https://integrate.api.nvidia.com/v1/chat/completions"
        data = {
            "model": "meta/llama-3.1-8b-instruct",
            "messages": [
                {"role": "system", "content": system_prompt},
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
        with urllib.request.urlopen(req, timeout=60) as r:
            res = json.loads(r.read().decode("utf-8"))
            return res["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"NVIDIA API Error: {str(e)}"

class BaseTool:
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        
    def execute(self, prompt: str, low_levels: List[str]) -> str:
        raise NotImplementedError("Each tool must implement its own execute logic.")
