import urllib.request
import urllib.parse
import re
from typing import List
from tools.base import BaseTool

class WebSearchTool(BaseTool):
    def execute(self, prompt: str, low_levels: List[str]) -> str:
        try:
            query = prompt.strip()
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
            url = f"https://html.duckduckgo.com/html/?q={urllib.parse.quote(query)}"
            req = urllib.request.Request(url, headers=headers)
            
            with urllib.request.urlopen(req, timeout=10) as r:
                html = r.read().decode('utf-8', errors='ignore')
                
            snippets = re.findall(r'<a class="result__snippet"[^>]*>(.*?)</a>', html, re.DOTALL)
            if not snippets:
                snippets = re.findall(r'result__snippet[^>]*>(.*?)</a>', html, re.DOTALL)
                
            if not snippets:
                return f"[SEARCH] [WebSearchTool Activated] No live results found on web for: '{query}'."
                
            results = []
            for i, s in enumerate(snippets[:3]):
                clean_snippet = re.sub(r'<[^>]+>', '', s)
                clean_snippet = clean_snippet.replace('&quot;', '"').replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
                clean_snippet = clean_snippet.encode('ascii', 'ignore').decode('ascii')
                results.append(f"   {i+1}. {clean_snippet.strip()}")
                
            summary = "\n".join(results)
            return (
                f"[SEARCH] [WebSearchTool Activated] Live web results for query: '{query}'\n"
                f"{summary}"
            )
        except Exception as e:
            return f"[SEARCH] [WebSearchTool Activated] Error running web search: {str(e)}"
