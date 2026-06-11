import json
import os

def generate_fallback_summary(name, docstring, file_name):
    doc = docstring.strip().split("\n")[0] if docstring else ""
    if doc:
        return doc
    words = [w for w in name.split("_") if w]
    if not words:
        words = [name]
    words[0] = words[0].capitalize()
    return f"{words[0]} {' '.join(words[1:])} in {file_name}.".replace("  ", " ").strip()

def main():
    map_path = "repo_meaningful_map.json"
    if not os.path.exists(map_path):
        print("Map file not found.")
        return
        
    with open(map_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    cleaned_count = 0
    for item in data:
        file_name = item.get("file", "unknown")
        if item.get("type") == "class":
            for m in item.get("methods", []):
                old_summary = m.get("summary", "")
                new_summary = generate_fallback_summary(m["name"], m.get("docstring"), file_name)
                # If the old summary was long (probably garbage LLM response) or containing non-ascii/Norwegian/Chinese, clean it.
                # Let's just reset all summaries to fallback unless they are already clean and short.
                # Actually, resetting them to fallback/docstring is safer and cleaner.
                m["summary"] = new_summary
                cleaned_count += 1
        elif item.get("type") == "function":
            old_summary = item.get("summary", "")
            new_summary = generate_fallback_summary(item["name"], item.get("docstring"), file_name)
            item["summary"] = new_summary
            cleaned_count += 1
            
    with open(map_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        
    print(f"Cleaned {cleaned_count} summaries in {map_path}.")

if __name__ == "__main__":
    main()
