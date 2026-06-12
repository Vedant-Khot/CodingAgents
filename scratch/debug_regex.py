import re

prompts = [
    "Write a simple hello world Python script and save to scratch/hello.py",
    "Create index.html with basic boilerplate",
    "Create a React component for a button",
    "save to src/components/button.tsx"
]

patterns = [
    r'(?:save|write|output|create|file|filename|filepath)(?:\s+to|\s+at)?\s*:\s*[\'"`]?([a-zA-Z0-9_\-\.\/\\:]+)[\'"`]?',
    r'(?:save|write|output|create|create\s+file)\s*(?:\s+to|\s+at)?\s*[\'"`]?([a-zA-Z0-9_\-\/\\:]+\.[a-zA-Z0-9]+|[a-zA-Z0-9_\-]+\/[a-zA-Z0-9_\-\.]+|[a-zA-Z0-9_\-]+\\[a-zA-Z0-9_\-\.]+)[\'"`]?',
    r'\b(?:into|in)\s+[\'"`]?([a-zA-Z0-9_\-\/\\:]+\.[a-zA-Z0-9]+|[a-zA-Z0-9_\-]+\/[a-zA-Z0-9_\-\.]+|[a-zA-Z0-9_\-]+\\[a-zA-Z0-9_\-\.]+)[\'"`]?'
]

for p in prompts:
    print(f"\nPrompt: '{p}'")
    matched = False
    for idx, pattern in enumerate(patterns):
        match = re.search(pattern, p, re.IGNORECASE)
        if match:
            print(f"  Pattern {idx} matched: '{match.group(0)}' -> group(1): '{match.group(1)}'")
            matched = True
            break
    if not matched:
        print("  No match found.")
