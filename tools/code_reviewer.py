import os
import re
import ast
import json
import math
import xml.etree.ElementTree as ET
from xml.dom import minidom
from collections import Counter
from typing import List, Dict
from tools.base import BaseTool, get_nvidia_api_key, query_nvidia, tokenize

# ==========================================
# THE SCOUT: AST CODESPAN EXTRACTION
# ==========================================

def get_functions_from_file(filepath: str) -> list:
    functions = []
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            code = f.read()
            
        tree = ast.parse(code)
        lines = code.splitlines()
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                start_line = node.lineno - 1
                end_line = getattr(node, "end_lineno", len(lines))
                func_code = "\n".join(lines[start_line:end_line])
                
                try:
                    args_str = ast.unparse(node.args)
                except Exception:
                    args_str = ""
                ret_str = f" -> {ast.unparse(node.returns)}" if node.returns else ""
                sig = f"def {node.name}({args_str}){ret_str}"
                
                functions.append({
                    "name": node.name,
                    "signature": sig,
                    "docstring": ast.get_docstring(node) or "",
                    "code": func_code,
                    "file": os.path.basename(filepath),
                    "filepath": filepath
                })
    except Exception as e:
        pass
    return functions

# ==========================================
# THE LIBRARIAN: INDEXING & SUMMARIZATION
# ==========================================

def generate_fallback_summary(func: dict) -> str:
    doc = func["docstring"].strip().split("\n")[0] if func["docstring"] else ""
    if doc:
        return doc
    words = func["name"].split("_")
    words[0] = words[0].capitalize()
    return f"{' '.join(words)} in {func['file']}."

def build_meaningful_map(dirpath: str) -> list:
    ignored_dirs = {'.git', '__pycache__', '.gemini', 'venv', '.venv', 'env', '.env', 'build', 'dist'}
    all_functions = []
    
    for root, dirs, files in os.walk(dirpath):
        dirs[:] = [d for d in dirs if d not in ignored_dirs]
        for f in files:
            if f.endswith(".py"):
                filepath = os.path.join(root, f)
                all_functions.extend(get_functions_from_file(filepath))
                
    cache_path = os.path.join(dirpath, "repo_meaningful_map.json")
    cache_map = {}
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cached_data = json.load(f)
                for item in cached_data:
                    cache_map[(item.get("file"), item.get("name"))] = item.get("summary", "")
        except Exception:
            pass
            
    api_key = get_nvidia_api_key()
    print(api_key);
    for func in all_functions:
        cache_key = (func["file"], func["name"])
        if cache_key in cache_map and cache_map[cache_key]:
            func["summary"] = cache_map[cache_key]
        else:
            if api_key:
                prompt = f"Write a 1-sentence summary of what this python function does:\n\n{func['code']}"
                sys_prompt = "You are a compiler that outputs a single short sentence explaining a function."
                summary = query_nvidia(prompt, sys_prompt)
                if "NVIDIA API Error" in summary:
                    summary = generate_fallback_summary(func)
            else:
                summary = generate_fallback_summary(func)
            func["summary"] = summary
            
    try:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(all_functions, f, indent=2)
    except Exception:
        pass
        
    return all_functions

# ==========================================
# THE SEARCH: TF-IDF SEMANTIC MATCHING
# ==========================================

class CodeSearchEngine:
    def __init__(self, documents: list):
        self.documents = documents
        self.vocab = set()
        self.idf = {}
        self.vectors = []
        self._build()
        
    def _build(self):
        tokenized_docs = []
        for doc in self.documents:
            text_context = f"{doc['name']} {doc['summary']} {doc['code']}"
            tokens = tokenize(text_context)
            tokenized_docs.append(tokens)
            self.vocab.update(tokens)
            
        N = len(self.documents)
        for word in self.vocab:
            df = sum(1 for tokens in tokenized_docs if word in tokens)
            self.idf[word] = math.log(1 + N / (1 + df))
            
        for tokens in tokenized_docs:
            vector = self._vectorize(tokens)
            self.vectors.append(vector)
            
    def _vectorize(self, tokens: List[str]) -> Dict[str, float]:
        tf = Counter(tokens)
        vector = {}
        for word, count in tf.items():
            if word in self.idf:
                vector[word] = count * self.idf[word]
        sq_sum = sum(val**2 for val in vector.values())
        if sq_sum > 0:
            norm = math.sqrt(sq_sum)
            for word in vector:
                vector[word] /= norm
        return vector
        
    def search(self, query: str, top_k: int = 3) -> list:
        query_tokens = tokenize(query)
        query_vec = self._vectorize(query_tokens)
        
        results = []
        for i, doc in enumerate(self.documents):
            doc_vec = self.vectors[i]
            dot = sum(query_vec.get(w, 0) * doc_vec.get(w, 0) for w in query_vec)
            results.append((dot, doc))
            
        results.sort(key=lambda x: x[0], reverse=True)
        return [doc for score, doc in results if score > 0.05][:top_k]

# ==========================================
# CODEREVIEWERTOOL IMPLEMENTATION
# ==========================================

class CodeReviewerTool(BaseTool):
    def execute(self, prompt: str, low_levels: List[str]) -> str:
        target = "."
        words = re.findall(r'[a-zA-Z0-9_\.\-\/\\:]+', prompt)
        for word in words:
            if os.path.exists(word):
                target = word
                break
                
        target_abs = os.path.abspath(target)
        
        is_qa_request = "Codebase QA" in low_levels or any(
            kw in prompt.lower() for kw in ["our codebase", "this codebase", "this project", "in this repository", "how does our", "explain our", "how does the codebase", "how is our", "where does our"]
        )
        is_arch_request = "Architecture Review" in low_levels or any(
            kw in prompt.lower() for kw in ["skeleton", "skeletonize", "repomix", "pack", "map", "structure"]
        )
        
        if is_qa_request:
            dir_to_scan = target_abs if os.path.isdir(target_abs) else os.path.dirname(target_abs)
            meaningful_map = build_meaningful_map(dir_to_scan)
            
            search_engine = CodeSearchEngine(meaningful_map)
            matched_funcs = search_engine.search(prompt, top_k=3)
            
            header = f"[QA] [CodeReviewerTool QA Pipeline Activated] for codebase in: {os.path.basename(dir_to_scan)}\n"
            header += f"   -> Found {len(meaningful_map)} functions in codebase.\n"
            if matched_funcs:
                header += f"   -> Identified {len(matched_funcs)} functions matching query:\n"
                for f in matched_funcs:
                    header += f"      * File: {f['file']} | Signature: {f['signature']} | Summary: {f['summary']}\n"
            else:
                header += f"   -> No highly matching functions identified for search query.\n"
                
            api_key = get_nvidia_api_key()
            if api_key and matched_funcs:
                code_context = "<codebase_context>\n"
                for f in matched_funcs:
                    code_context += (
                        f"  <function file=\"{f['file']}\" name=\"{f['name']}\" signature=\"{f['signature']}\">\n"
                        f"    <summary>{f['summary']}</summary>\n"
                        f"    <code>\n{f['code']}\n    </code>\n"
                        f"  </function>\n"
                    )
                code_context += "</codebase_context>"
                
                llm_prompt = (
                    f"You are an expert software engineer. Based on the following source code context in XML, answer the user's question.\n\n"
                    f"<user_question>\n{prompt}\n</user_question>\n\n"
                    f"{code_context}\n\n"
                    f"Provide a clear, detailed, and accurate explanation."
                )
                sys_prompt = "You are a software architect explaining system design based on function implementations."
                
                analysis_res = query_nvidia(llm_prompt, sys_prompt)
                output = f"{header}\n--- Final Analysis (NVIDIA NIM) ---\n{analysis_res}"
            else:
                fallback_res = ""
                if not api_key:
                    fallback_res += "   -> Note: NVIDIA_API_KEY is not set. Displaying matching function source codes directly:\n\n"
                else:
                    fallback_res += "   -> Note: Displaying matching function source codes directly:\n\n"
                
                for f in matched_funcs:
                    fallback_res += f"*** File: {f['file']} | {f['signature']} ***\n{f['code']}\n" + "="*50 + "\n"
                output = f"{header}\n{fallback_res}"
                
        elif is_arch_request:
            if os.path.isfile(target_abs):
                return f"[DESIGN] [CodeReviewerTool Activated] Codebase Skeletonizer target must be a directory. Target specified is a file: {os.path.basename(target_abs)}"
            
            output_xml = self._generate_skeleton_xml(target_abs)
            output_file = os.path.join(target_abs, "repo_skeleton.xml")
            try:
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(output_xml)
                status_str = f"Saved XML context map to: {output_file}"
            except Exception as e:
                status_str = f"Failed to save XML context file: {str(e)}"
                
            preview_lines = output_xml.splitlines()[:25]
            preview = "\n".join(["      " + l for l in preview_lines])
            
            report = [
                f"[DESIGN] [CodeReviewerTool Activated] Codebase Skeletonizer completed for: {os.path.basename(target_abs)}",
                f"   -> Status: {status_str}",
                f"   -> XML Preview (First 25 lines):",
                preview,
                f"      ... (XML file truncated, check '{os.path.basename(output_file)}' for full content)"
            ]
            output = "\n".join(report)
        else:
            if os.path.isfile(target_abs):
                res = self.analyze_file(target_abs)
                report = [
                    f"[REVIEW] [CodeReviewerTool Activated] File Analysis for: {os.path.basename(target_abs)}",
                    f"   -> Total Lines: {res['lines']}",
                    f"   -> Classes: {res['classes']}",
                    f"   -> Functions: {res['functions']}",
                    f"   -> Security Alerts ({len(res['security_issues'])}):",
                ]
                if res['security_issues']:
                    for iss in res['security_issues']:
                        report.append(f"      [!] {iss}")
                else:
                    report.append("      No security threats found.")
                    
                report.append(f"   -> Readability Recommendations ({len(res['style_issues'])}):")
                if res['style_issues']:
                    for iss in res['style_issues']:
                        report.append(f"      [*] {iss}")
                else:
                    report.append("      Style conventions followed.")
                
                output = "\n".join(report)
            else:
                output = self.analyze_codebase(target_abs)
            
        return output.encode('ascii', 'ignore').decode('ascii')

    # --- SKELETONIZER HELPER METHODS ---
    
    def _get_ast_skeleton(self, filepath: str) -> dict:
        skeleton = {
            "summary": "",
            "imports": [],
            "classes": [],
            "functions": []
        }
        try:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                code = f.read()
            tree = ast.parse(code)
            
            docstring = ast.get_docstring(tree)
            if docstring:
                skeleton["summary"] = docstring.strip()
                
            for node in tree.body:
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        skeleton["imports"].append(f"import {alias.name}")
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    names = ", ".join(alias.name for alias in node.names)
                    skeleton["imports"].append(f"from {module} import {names}")
                elif isinstance(node, ast.ClassDef):
                    class_info = {
                        "name": node.name,
                        "bases": [ast.unparse(b) for b in node.bases],
                        "docstring": ast.get_docstring(node) or "",
                        "methods": []
                    }
                    for sub_node in node.body:
                        if isinstance(sub_node, ast.FunctionDef) or isinstance(sub_node, ast.AsyncFunctionDef):
                            sig = self._get_function_signature(sub_node)
                            class_info["methods"].append({
                                "signature": sig,
                                "docstring": ast.get_docstring(sub_node) or ""
                            })
                    skeleton["classes"].append(class_info)
                elif isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                    sig = self._get_function_signature(node)
                    skeleton["functions"].append({
                        "signature": sig,
                        "docstring": ast.get_docstring(node) or ""
                    })
        except Exception as e:
            skeleton["summary"] = f"Failed to parse AST: {str(e)}"
        return skeleton

    def _get_function_signature(self, node) -> str:
        is_async = isinstance(node, ast.AsyncFunctionDef)
        prefix = "async def " if is_async else "def "
        try:
            args_str = ast.unparse(node.args)
        except Exception:
            args_str = ""
        ret_str = ""
        if node.returns:
            try:
                ret_str = f" -> {ast.unparse(node.returns)}"
            except Exception:
                pass
        return f"{prefix}{node.name}({args_str}){ret_str}"

    def _generate_skeleton_xml(self, dirpath: str) -> str:
        root_el = ET.Element("repository_map")
        ignored_dirs = {'.git', '__pycache__', '.gemini', 'venv', '.venv', 'env', '.env', 'build', 'dist'}
        
        tree_lines = []
        py_files = []
        
        for root, dirs, files in os.walk(dirpath):
            dirs[:] = [d for d in dirs if d not in ignored_dirs]
            level = root.replace(dirpath, '').count(os.sep)
            indent = '  ' * level
            subfolder = os.path.basename(root)
            if subfolder and subfolder != os.path.basename(dirpath):
                tree_lines.append(f"{indent}- {subfolder}/")
                
            sub_indent = '  ' * (level + 1)
            for f in files:
                ext = os.path.splitext(f)[1].lower()
                tree_lines.append(f"{sub_indent}- {f}")
                if ext == '.py':
                    py_files.append(os.path.join(root, f))
                    
        map_el = ET.SubElement(root_el, "structure")
        map_el.text = "\n" + "\n".join(tree_lines) + "\n"
        
        for filepath in py_files:
            rel_path = os.path.relpath(filepath, dirpath)
            skeleton = self._get_ast_skeleton(filepath)
            
            file_el = ET.SubElement(root_el, "file", path=rel_path)
            summary_el = ET.SubElement(file_el, "summary")
            summary_el.text = skeleton["summary"] or "No file-level description."
            
            if skeleton["imports"]:
                imports_el = ET.SubElement(file_el, "imports")
                for imp in skeleton["imports"]:
                    imp_el = ET.SubElement(imports_el, "import_statement")
                    imp_el.text = imp
                    
            if skeleton["classes"]:
                classes_el = ET.SubElement(file_el, "classes")
                for cls in skeleton["classes"]:
                    cls_el = ET.SubElement(classes_el, "class")
                    sig_el = ET.SubElement(cls_el, "class_signature")
                    bases_str = f"({', '.join(cls['bases'])})" if cls['bases'] else ""
                    sig_el.text = f"class {cls['name']}{bases_str}"
                    
                    doc_el = ET.SubElement(cls_el, "docstring")
                    doc_el.text = cls["docstring"] or "No class description."
                    
                    if cls["methods"]:
                        methods_el = ET.SubElement(cls_el, "methods")
                        for method in cls["methods"]:
                            meth_el = ET.SubElement(methods_el, "method")
                            msig_el = ET.SubElement(meth_el, "method_signature")
                            msig_el.text = method["signature"]
                            mdoc_el = ET.SubElement(meth_el, "docstring")
                            mdoc_el.text = method["docstring"] or "No method description."
                            
            if skeleton["functions"]:
                funcs_el = ET.SubElement(file_el, "functions")
                for func in skeleton["functions"]:
                    func_el = ET.SubElement(funcs_el, "function")
                    fsig_el = ET.SubElement(func_el, "function_signature")
                    fsig_el.text = func["signature"]
                    fdoc_el = ET.SubElement(func_el, "docstring")
                    fdoc_el.text = func["docstring"] or "No function description."
                    
        xml_str = ET.tostring(root_el, encoding='utf-8')
        parsed_xml = minidom.parseString(xml_str)
        pretty_xml = parsed_xml.toprettyxml(indent="  ")
        return pretty_xml

    # --- STANDARD ANALYSIS HELPER METHODS ---
    
    def analyze_file(self, filepath: str) -> dict:
        metrics = {
            "lines": 0,
            "classes": 0,
            "functions": 0,
            "security_issues": [],
            "style_issues": []
        }
        try:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
                
            metrics["lines"] = len(lines)
            
            secret_pattern = re.compile(r'(api_key|secret|password|passwd|token|credentials)\s*=\s*[\'"][a-zA-Z0-9_\-]{8,}[\'"]', re.IGNORECASE)
            eval_pattern = re.compile(r'\b(eval|exec)\b')
            unsafe_subproc = re.compile(r'\bsubprocess\.(Popen|run|call)\(.*shell\s*=\s*True')
            weak_hash = re.compile(r'\b(md5|sha1)\b')
            
            in_function = False
            func_line_count = 0
            func_name = ""
            
            for i, line in enumerate(lines, 1):
                line_str = line.strip()
                if line_str.startswith("class "):
                    metrics["classes"] += 1
                elif line_str.startswith("def "):
                    metrics["functions"] += 1
                    func_name = line_str.split("(")[0].replace("def ", "").strip()
                    in_function = True
                    func_line_count = 0
                    
                    has_doc = False
                    for j in range(i, min(i + 3, len(lines))):
                        next_line = lines[j].strip()
                        if next_line.startswith('"""') or next_line.startswith("'''"):
                            has_doc = True
                            break
                    if not has_doc:
                        metrics["style_issues"].append(f"Line {i}: Function '{func_name}' is missing a docstring.")
                
                if in_function:
                    if line_str == "" or line_str.startswith("#"):
                        pass
                    else:
                        func_line_count += 1
                        if i < len(lines):
                            next_line = lines[i]
                            if next_line.strip() and not next_line.startswith(" ") and not next_line.startswith("\t") and not next_line.startswith("#"):
                                in_function = False
                                if func_line_count > 40:
                                    metrics["style_issues"].append(f"Function '{func_name}' is long ({func_line_count} logic lines). Consider refactoring.")
                
                if secret_pattern.search(line):
                    metrics["security_issues"].append(f"Line {i}: Potential hardcoded secret/key detected.")
                if eval_pattern.search(line):
                    metrics["security_issues"].append(f"Line {i}: Use of unsafe 'eval' or 'exec' detected.")
                if unsafe_subproc.search(line):
                    metrics["security_issues"].append(f"Line {i}: Unsafe shell execution via subprocess detected.")
                if weak_hash.search(line):
                    metrics["security_issues"].append(f"Line {i}: Weak hashing algorithm (md5/sha1) detected.")
        except Exception as e:
            metrics["security_issues"].append(f"Error reading file: {str(e)}")
        return metrics

    def analyze_codebase(self, dirpath: str) -> str:
        summary = []
        summary.append(f"[REVIEW] [CodeReviewerTool Activated] Codebase Analysis for: {os.path.basename(dirpath)}")
        
        file_types = {}
        total_files = 0
        total_lines = 0
        total_classes = 0
        total_funcs = 0
        
        all_security_issues = []
        all_style_issues = []
        
        ignored_dirs = {'.git', '__pycache__', '.gemini', 'venv', '.venv', 'env', '.env', 'build', 'dist'}
        
        tree_lines = []
        for root, dirs, files in os.walk(dirpath):
            dirs[:] = [d for d in dirs if d not in ignored_dirs]
            
            level = root.replace(dirpath, '').count(os.sep)
            indent = '  ' * level
            subfolder = os.path.basename(root)
            if subfolder and subfolder != os.path.basename(dirpath):
                tree_lines.append(f"{indent}[] {subfolder}/")
                
            sub_indent = '  ' * (level + 1)
            for f in files:
                total_files += 1
                ext = os.path.splitext(f)[1].lower() or "no_extension"
                file_types[ext] = file_types.get(ext, 0) + 1
                
                if level < 2:
                    tree_lines.append(f"{sub_indent}- {f}")
                    
                if ext == '.py':
                    filepath = os.path.join(root, f)
                    metrics = self.analyze_file(filepath)
                    total_lines += metrics["lines"]
                    total_classes += metrics["classes"]
                    total_funcs += metrics["functions"]
                    
                    for issue in metrics["security_issues"]:
                        all_security_issues.append(f"{f} -> {issue}")
                    for issue in metrics["style_issues"]:
                        all_style_issues.append(f"{f} -> {issue}")
                        
        summary.append("   -> Directory Layout (Max Depth 2):")
        summary.append("\n".join(["      " + l for l in tree_lines[:25]]))
        if len(tree_lines) > 25:
            summary.append("      ... (directory tree truncated)")
            
        summary.append("   -> File Extensions Summary:")
        for ext, count in sorted(file_types.items(), key=lambda x: x[1], reverse=True):
            summary.append(f"      {ext or 'no_ext'}: {count} files")
            
        summary.append(f"   -> Code Metrics:")
        summary.append(f"      Total Python Files: {file_types.get('.py', 0)}")
        summary.append(f"      Total Lines of Code: {total_lines}")
        summary.append(f"      Total Classes: {total_classes}")
        summary.append(f"      Total Functions: {total_funcs}")
        
        summary.append(f"   -> Security Flags Detected ({len(all_security_issues)}):")
        if all_security_issues:
            for issue in all_security_issues[:10]:
                summary.append(f"      [!] {issue}")
            if len(all_security_issues) > 10:
                summary.append(f"      ... and {len(all_security_issues) - 10} more security warnings.")
        else:
            summary.append("      No critical security issues found.")
            
        summary.append(f"   -> Code Style Recommendations ({len(all_style_issues)}):")
        if all_style_issues:
            for issue in all_style_issues[:10]:
                summary.append(f"      [*] {issue}")
            if len(all_style_issues) > 10:
                summary.append(f"      ... and {len(all_style_issues) - 10} more linter warnings.")
        else:
            summary.append("      Code readability looks good.")
            
        return "\n".join(summary)
