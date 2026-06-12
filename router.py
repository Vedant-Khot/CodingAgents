import re
import math
import json
from collections import Counter
from typing import Dict, Any, List, Optional, Set, Tuple
from tools import (
    BaseTool,
    WebSearchTool,
    CodeGeneratorTool,
    DebuggerTool,
    CodeOptimizerTool,
    CodeReviewerTool,
    CodeConverterTool,
    TestGeneratorTool,
    DeploymentTool,
    SystemDesignTool,
    DocumentationTool,
    TerminalExecutorTool,
    DefaultAgentChat,
    get_nvidia_api_key,
    query_nvidia
)

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def tokenize(text: str) -> List[str]:
    """Lowercases, removes punctuation, and splits text into cleaned tokens, filtering stop words."""
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
    return [t for t in tokens if t not in stop_words and len(t) > 1]


# ==========================================
# LEVEL 1: RULE-BASED INTENT RULES
# ==========================================

class IntentRule:
    def __init__(self, high_level: str, low_level: str, keywords: List[str], regex_patterns: List[str] = None):
        self.high_level = high_level
        self.low_level = low_level
        self.keywords = [kw.lower() for kw in keywords]
        self.regex_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in (regex_patterns or [])]

    def matches(self, prompt: str) -> bool:
        prompt_lower = prompt.lower()
        for kw in self.keywords:
            if kw in prompt_lower:
                return True
        for pattern in self.regex_patterns:
            if pattern.search(prompt):
                return True
        return False


# ==========================================
# LEVEL 2: TF-IDF SEMANTIC MATCHING
# ==========================================

class TfidfSimilarity:
    def __init__(self, exemplars: List[Dict[str, Any]]):
        self.exemplars = exemplars
        self.vocab = set()
        self.idf = {}
        self.exemplar_vectors = []
        self._build()

    def _build(self):
        tokenized_exemplars = []
        for ex in self.exemplars:
            tokens = tokenize(ex["text"])
            tokenized_exemplars.append(tokens)
            self.vocab.update(tokens)
            
        N = len(self.exemplars)
        for word in self.vocab:
            df = sum(1 for tokens in tokenized_exemplars if word in tokens)
            self.idf[word] = math.log(1 + N / (1 + df))
            
        for tokens in tokenized_exemplars:
            vector = self._vectorize(tokens)
            self.exemplar_vectors.append(vector)

    def _vectorize(self, tokens: List[str]) -> Dict[str, float]:
        tf = Counter(tokens)
        vector = {}
        for word, count in tf.items():
            if word in self.idf:
                vector[word] = count * self.idf[word]
        
        sq_sum = sum(val ** 2 for val in vector.values())
        if sq_sum > 0:
            norm = math.sqrt(sq_sum)
            for word in vector:
                vector[word] /= norm
        return vector

    def get_similarities(self, query: str) -> List[Dict[str, Any]]:
        query_tokens = tokenize(query)
        query_vec = self._vectorize(query_tokens)
        
        results = []
        for i, ex in enumerate(self.exemplars):
            ex_vec = self.exemplar_vectors[i]
            dot_product = sum(query_vec.get(word, 0) * ex_vec.get(word, 0) for word in query_vec)
            results.append({
                "exemplar": ex["text"],
                "high_level": ex["high"],
                "low_level": ex["low"],
                "similarity": dot_product
            })
            
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results


# ==========================================
# HIERARCHICAL MULTI-INTENT ROUTER
# ==========================================

class HierarchicalRouter:
    def __init__(self, similarity_threshold: float = 0.33):
        self.similarity_threshold = similarity_threshold
        self.rules: List[IntentRule] = []
        self._setup_rules()
        self._setup_exemplars()

    def _setup_rules(self):
        # Learn rules
        self.add_rule(IntentRule("Learn", "Concept", ["what is a", "what is", "explain the", "explain", "how does", "concept of"], [r"\bexplain\s+[a-zA-Z0-9_-]+"]))
        self.add_rule(IntentRule("Learn", "Comparison", ["vs", "versus", "difference between"]))
        self.add_rule(IntentRule("Learn", "Tutorial", ["teach me", "tutorial for", "step by step guide"]))
        
        # Build rules
        self.add_rule(IntentRule("Build", "Frontend", ["login screen", "login page", "angular component", "css styling", "react widget"]))
        self.add_rule(IntentRule("Build", "Backend", ["write python code", "build python script", "node express server", "flask backend"]))
        self.add_rule(IntentRule("Build", "API", ["rest api", "graphql query", "express route", "api endpoint"]))
        self.add_rule(IntentRule("Build", "Database", ["sql table", "prisma schema", "database migration"]))

        # Fix rules
        self.add_rule(IntentRule("Fix", "Error", ["why is this error", "app crashes", "uncaught exception", "traceback", "fails with"]))
        self.add_rule(IntentRule("Fix", "Bug", ["fix this bug", "incorrect output", "wrong calculations"]))
        self.add_rule(IntentRule("Fix", "Configuration", ["npm install error", "requirements.txt conflict", "webpack config"]))

        # Improve rules
        self.add_rule(IntentRule("Improve", "Refactor", ["refactor this", "make this cleaner", "clean code", "improve readability"]))
        self.add_rule(IntentRule("Improve", "Optimize", ["optimize this", "speed up", "run faster", "reduce memory"]))
        self.add_rule(IntentRule("Improve", "Secure", ["prevent sql injection", "xss sanitization", "security vulnerability"]))

        # Analyze rules
        self.add_rule(IntentRule("Analyze", "Code Review", ["review this code", "code review for", "lint check"]))
        self.add_rule(IntentRule("Analyze", "Architecture Review", ["architecture review", "evaluate design pattern", "skeleton", "skeletonize", "repomix", "codebase map", "repository map"]))
        self.add_rule(IntentRule("Analyze", "Codebase QA", ["our codebase", "this codebase", "this project", "in this repository", "how does our", "explain our", "how does the codebase", "how is our", "where does our", "work in this project", "works in this project"]))

        # Convert rules
        self.add_rule(IntentRule("Convert", "Language", ["convert to python", "translate to typescript", "port java to rust"]))
        self.add_rule(IntentRule("Convert", "Framework", ["migrate angularjs to angular", "convert react to vue"]))

        # Test rules
        self.add_rule(IntentRule("Test", "Unit", ["write unit tests", "generate test cases", "pytest for", "junit"]))

        # Deploy rules
        self.add_rule(IntentRule("Deploy", "Docker", ["dockerize", "dockerfile", "docker-compose"]))
        self.add_rule(IntentRule("Deploy", "CI/CD", ["github actions pipeline", "configure cicd", "jenkinsfile"]))

        # Design rules
        self.add_rule(IntentRule("Design", "System", ["design whatsapp", "system design of", "system architecture"]))
        self.add_rule(IntentRule("Design", "Database", ["database design for", "design schema"]))

        # Document rules
        self.add_rule(IntentRule("Document", "README", ["write readme", "readme.md template"]))
        self.add_rule(IntentRule("Document", "API Docs", ["generate api docs", "swagger documentation"]))
        self.add_rule(IntentRule("Document", "Comments", ["add comments", "add docstrings", "explain inline"]))

        # Execute rules
        self.add_rule(IntentRule("Execute", "Command", ["run command", "execute terminal", "run shell", "terminal command", "run script", "execute command", "bash", "powershell", "cmd", "npm run", "pip install", "pytest"]))

    def add_rule(self, rule: IntentRule):
        self.rules.append(rule)

    def _setup_exemplars(self):
        exemplars = [
            # 1. Learn
            {"text": "explain the concept of jwt token authentication oauth session login", "high": "Learn", "low": "Concept"},
            {"text": "what is a binary tree database index hashing load balancer", "high": "Learn", "low": "Concept"},
            {"text": "how does docker containerization virtual machine work internally", "high": "Learn", "low": "Concept"},
            {"text": "teach me react from scratch step by step guide book", "high": "Learn", "low": "Tutorial"},
            {"text": "guide tutorial to build a simple rest api with node js backend", "high": "Learn", "low": "Tutorial"},
            {"text": "compare react vs vue vs angular which is better comparison pros cons", "high": "Learn", "low": "Comparison"},
            {"text": "difference between sql and nosql postgres mongodb database engines", "high": "Learn", "low": "Comparison"},
            
            # 2. Build
            {"text": "create a beautiful login screen page html css tailwind flexbox responsive", "high": "Build", "low": "Frontend"},
            {"text": "generate a new angular component user interface dashboard card profile", "high": "Build", "low": "Frontend"},
            {"text": "write python backend django views controller server script", "high": "Build", "low": "Backend"},
            {"text": "implement express.js backend router logic node server code", "high": "Build", "low": "Backend"},
            {"text": "build a rest api web service endpoint url mapping search route", "high": "Build", "low": "API"},
            {"text": "create database tables schema model definition prisma relational diagram", "high": "Build", "low": "Database"},
            
            # 3. Fix
            {"text": "why is this exception occurring error crash failure traceback bug", "high": "Fix", "low": "Error"},
            {"text": "uncaught runtime exception typeerror undefined is not a function crash", "high": "Fix", "low": "Error"},
            {"text": "my react app shows ExpressionChangedAfterItHasBeenCheckedError crash runtime", "high": "Fix", "low": "Error"},
            {"text": "fix this logical bug sorting algorithm index out of bounds loop wrong value", "high": "Fix", "low": "Bug"},
            {"text": "there is a bug in my conditional check state updates incorrect behavior", "high": "Fix", "low": "Bug"},
            {"text": "fix dependency configuration issue package lock json node modules conflict", "high": "Fix", "low": "Configuration"},
            
            # 4. Improve
            {"text": "refactor this code to follow dry clean code patterns solid design principles", "high": "Improve", "low": "Refactor"},
            {"text": "make this class hierarchy cleaner easier to read code smell reuse", "high": "Improve", "low": "Refactor"},
            {"text": "optimize this python function run faster reduce latency scale performance cpu", "high": "Improve", "low": "Optimize"},
            {"text": "improve algorithm performance speed up execution decrease time complexity O(N)", "high": "Improve", "low": "Optimize"},
            {"text": "make this script secure against sql injection sanitizing database queries", "high": "Improve", "low": "Secure"},
            {"text": "sanitize inputs prevent cross site scripting xss vulnerability security leak", "high": "Improve", "low": "Secure"},
            
            # 5. Analyze
            {"text": "please review my code for potential bugs security flaws code smells", "high": "Analyze", "low": "Code Review"},
            {"text": "provide code review comments feedback pull request merge review", "high": "Analyze", "low": "Code Review"},
            {"text": "analyze system architecture diagrams cloud design microservices setup", "high": "Analyze", "low": "Architecture Review"},
            {"text": "review architecture of this node.js application components structure database", "high": "Analyze", "low": "Architecture Review"},
            {"text": "run security scan verify packages vulnerabilities audit dependencies", "high": "Analyze", "low": "Security Review"},
            {"text": "how does our login system work in this project", "high": "Analyze", "low": "Codebase QA"},
            {"text": "explain how tokenization or word splitting works in our codebase", "high": "Analyze", "low": "Codebase QA"},
            {"text": "where is the router defined in this repository", "high": "Analyze", "low": "Codebase QA"},
            {"text": "how does the codebase process and execute queries", "high": "Analyze", "low": "Codebase QA"},
            
            # 6. Convert
            {"text": "convert java code snippet logic function structure to python", "high": "Convert", "low": "Language"},
            {"text": "translate typescript function codebase files into clean go rust code", "high": "Convert", "low": "Language"},
            {"text": "migrate our angularjs legacy frontend application to angular 18 typescript", "high": "Convert", "low": "Framework"},
            {"text": "convert mysql tables schemas setup query syntax to mongodb collections", "high": "Convert", "low": "Database"},
            
            # 7. Test
            {"text": "write unit tests testing validation functions module using pytest unittest", "high": "Test", "low": "Unit"},
            {"text": "generate mock test cases assertions for backend service class functions", "high": "Test", "low": "Unit"},
            {"text": "write integration tests verifying database connection api endpoints routes pipeline", "high": "Test", "low": "Integration"},
            {"text": "create e2e end-to-end user browser tests cypress playwright automation", "high": "Test", "low": "E2E"},
            
            # 8. Deploy
            {"text": "how do i deploy host release this application on aws ec2 gcp cloud service", "high": "Deploy", "low": "Cloud"},
            {"text": "dockerize my web application write dockerfile multi stage build container setup", "high": "Deploy", "low": "Docker"},
            {"text": "configure ci cd automated pipeline workflow github actions jenkins deploy script", "high": "Deploy", "low": "CI/CD"},
            
            # 9. Design
            {"text": "design system architecture scaled whatsapp chat application netflix load balancer", "high": "Design", "low": "System"},
            {"text": "create database schema entity relationship diagram architecture ERD design", "high": "Design", "low": "Database"},
            {"text": "design REST API endpoints specs swagger schema blueprints paths design patterns", "high": "Design", "low": "API"},
            
            # 10. Document
            {"text": "write readme md markdown documentation file repository documentation structure", "high": "Document", "low": "README"},
            {"text": "generate swagger openapi specs api docs document paths outputs queries", "high": "Document", "low": "API Docs"},
            {"text": "add documentation code comments python docstrings explain logic inside methods", "high": "Document", "low": "Comments"},
            
            # 11. Execute
            {"text": "run npm test command to execute test suite", "high": "Execute", "low": "Command"},
            {"text": "execute python script check results", "high": "Execute", "low": "Command"},
            {"text": "run shell command list files directory", "high": "Execute", "low": "Command"},
            {"text": "run git status in terminal", "high": "Execute", "low": "Command"},
            {"text": "execute command to install pip requirements", "high": "Execute", "low": "Command"}
        ]
        self.similarity_engine = TfidfSimilarity(exemplars)

    def route(self, prompt: str) -> Dict[str, Any]:
        high_intents: Set[str] = set()
        low_intents: Set[str] = set()
        
        # 1. Level 1 Matching
        for rule in self.rules:
            if rule.matches(prompt):
                high_intents.add(rule.high_level)
                low_intents.add(rule.low_level)
                
        # 2. Level 2 Matching
        sim_results = self.similarity_engine.get_similarities(prompt)
        matches_above_threshold = [r for r in sim_results if r["similarity"] >= self.similarity_threshold]
        
        for match in matches_above_threshold:
            high_intents.add(match["high_level"])
            low_intents.add(match["low_level"])
            
        # Fallback if empty
        if not high_intents and sim_results:
            best_match = sim_results[0]
            if best_match["similarity"] > 0.05:
                high_intents.add(best_match["high_level"])
                low_intents.add(best_match["low_level"])
            else:
                high_intents.add("Learn")
                low_intents.add("Concept")

        # Convert to list
        high_list = sorted(list(high_intents))
        low_list = sorted(list(low_intents))
        
        return {
            "high_level": high_list,
            "low_level": low_list,
            "best_score": round(sim_results[0]["similarity"], 3) if sim_results else 0.0
        }


# ==========================================
# AGENTIC PLANNER
# ==========================================

class AgenticPlanner:
    SYSTEM_PROMPT = (
        "You are an expert agent planner. Given a user's coding request and a list of available tools, "
        "your job is to break down the request into a structured sequence of execution steps.\n\n"
        "Each step in the plan must correspond to ONE tool. For each step, you must output:\n"
        "- 'tool': The high-level intent key of the tool to use (must be one of: Learn, Build, Fix, Improve, Analyze, Convert, Test, Deploy, Design, Document, Execute, Default).\n"
        "- 'description': A short sentence explaining what this step does.\n"
        "- 'prompt': The refined sub-prompt tailored for this specific tool.\n\n"
        "Return ONLY a valid JSON object with a single top-level key 'plan' containing the array of steps. "
        "Do not include any markdown formatting (e.g., no ```json block wrapper), no explanations, no text before or after the JSON. "
        "Ensure the JSON is strictly valid.\n\n"
        "Example output:\n"
        "{\n"
        "  \"plan\": [\n"
        "    {\n"
        "      \"tool\": \"Build\",\n"
        "      \"description\": \"Generate the backend express server script\",\n"
        "      \"prompt\": \"Create a backend node express server with standard CRUD endpoints.\"\n"
        "    },\n"
        "    {\n"
        "      \"tool\": \"Test\",\n"
        "      \"description\": \"Write unit tests for the express server\",\n"
        "      \"prompt\": \"Write integration/unit tests using mocha/chai for the express server crud endpoints.\"\n"
        "    }\n"
        "  ]\n"
        "}"
    )

    def plan(self, prompt: str, tools_info: list) -> list:
        api_key = get_nvidia_api_key()
        if not api_key:
            return []
            
        tools_str = json.dumps(tools_info, indent=2)
        user_prompt = (
            f"User Coding Request: \"{prompt}\"\n\n"
            f"Available Tools:\n{tools_str}\n\n"
            f"Please generate the JSON plan containing the sequence of steps to fulfill this request."
        )
        
        try:
            res = query_nvidia(user_prompt, self.SYSTEM_PROMPT).strip()
            # Clean up potential markdown wrappers
            if res.startswith("```"):
                res = re.sub(r"^```(?:json)?\n", "", res)
                res = re.sub(r"\n```$", "", res)
            res = res.strip()
            
            plan_data = json.loads(res)
            return plan_data.get("plan", [])
        except Exception:
            return []


# ==========================================
# TOOL ORCHESTRATOR & DISPATCHER
# ==========================================

class ToolOrchestrator:
    def __init__(self):
        self.router = HierarchicalRouter()
        self.tools: Dict[str, BaseTool] = {}
        self._register_tools()

    def _register_tools(self):
        # Register tools
        self.tools["Learn"] = WebSearchTool("LearnTool", "Searches documentation and tutorials.")
        self.tools["Build"] = CodeGeneratorTool("BuilderTool", "Creates frontend/backend/database code.")
        self.tools["Fix"] = DebuggerTool("DebuggerTool", "Diagnoses errors and suggests patches.")
        self.tools["Improve"] = CodeOptimizerTool("OptimizerTool", "Refactors and speeds up execution.")
        self.tools["Analyze"] = CodeReviewerTool("ReviewerTool", "Performs code audits and quality reviews.")
        self.tools["Convert"] = CodeConverterTool("ConverterTool", "Translates code languages and migrates frameworks.")
        self.tools["Test"] = TestGeneratorTool("TestGenTool", "Generates unit and integration tests.")
        self.tools["Deploy"] = DeploymentTool("DeployerTool", "Creates deployment configuration pipelines.")
        self.tools["Design"] = SystemDesignTool("DesignTool", "Generates architectural systems designs.")
        self.tools["Document"] = DocumentationTool("DocTool", "Generates READMEs and API documentation.")
        self.tools["Execute"] = TerminalExecutorTool("TerminalExecutorTool", "Executes terminal/shell commands and reads output.")
        # Fallback tool
        self.tools["Default"] = DefaultAgentChat("DefaultChat", "Standard conversational chatbot.")

    def get_tool_for_intent(self, high_level_intent: str) -> BaseTool:
        return self.tools.get(high_level_intent, self.tools["Default"])

    def process_and_execute(self, prompt: str) -> Dict[str, Any]:
        # Gather tool info for the planner
        tools_info = []
        for intent, tool in self.tools.items():
            if intent == "Default":
                continue
            tools_info.append({
                "tool": intent,
                "name": tool.name,
                "description": tool.description
            })
            
        planner = AgenticPlanner()
        plan = planner.plan(prompt, tools_info)
        
        is_fallback = False
        if not plan:
            # Fall back to Hierarchical Router
            is_fallback = True
            routing_result = self.router.route(prompt)
            high_intents = routing_result["high_level"]
            
            plan = []
            for intent in high_intents:
                plan.append({
                    "tool": intent,
                    "description": f"Fallback route to {intent} tool",
                    "prompt": prompt
                })
                
        execution_outputs = []
        dispatched_tools = []
        step_details = []
        
        for step in plan:
            intent = step["tool"]
            step_prompt = step["prompt"]
            step_desc = step["description"]
            
            tool = self.get_tool_for_intent(intent)
            if tool.name not in dispatched_tools:
                dispatched_tools.append(tool.name)
            
            # Retrieve low-level intents for this step's sub-prompt
            step_routing = self.router.route(step_prompt)
            step_low_levels = step_routing["low_level"]
            
            # Execute
            output = tool.execute(step_prompt, step_low_levels)
            execution_outputs.append(output)
            
            step_details.append({
                "tool": tool.name,
                "intent": intent,
                "description": step_desc,
                "prompt": step_prompt,
                "output": output
            })
            
        if plan:
            first_step_routing = self.router.route(plan[0]["prompt"])
            high_level = first_step_routing["high_level"]
            low_level = first_step_routing["low_level"]
            confidence = first_step_routing["best_score"]
        else:
            high_level = ["Default"]
            low_level = ["Concept"]
            confidence = 0.0
            
        return {
            "prompt": prompt,
            "plan": plan,
            "is_fallback": is_fallback,
            "high_level": high_level[0] if len(high_level) == 1 else high_level,
            "low_level": low_level[0] if len(low_level) == 1 else low_level,
            "confidence": confidence,
            "dispatched_tools": dispatched_tools,
            "execution_results": execution_outputs,
            "steps": step_details
        }


# ==========================================
# INTERACTIVE CLI EXECUTION
# ==========================================
if __name__ == "__main__":
    orchestrator = ToolOrchestrator()
    
    print("=" * 70)
    print("ROUTER AGENT ORCHESTRATOR & TOOL EXECUTION CLI")
    print("=" * 70)
    print("Enter a coding task to route intent and trigger the correct tool.")
    print("Type 'exit' or 'quit' to end.\n")

    # Run quick startup tests
    startup_tests = [
        "Create a Flutter login screen",
        "My App crashes with null pointer traceback",
        "Convert Java code to Python and optimize it",
        "Run command `echo router_self_test`"
    ]
    
    print("--- Running Startup Self-Test ---")
    for test in startup_tests:
        res = orchestrator.process_and_execute(test)
        print(f"\nPrompt:      \"{test}\"")
        if "plan" in res:
            print("Generated Plan:")
            for i, step in enumerate(res["plan"], 1):
                fallback_flag = " (Fallback)" if res.get("is_fallback") else ""
                print(f"  Step {i}{fallback_flag}: {step['description']} [{step['tool']}]")
        print(f"Intents:     {res['high_level']} -> {res['low_level']}")
        print(f"Tools Run:   {res['dispatched_tools']}")
        print(f"Execution Output:")
        for output in res['execution_results']:
            print(output)
        print("-" * 50)
    print("======================================================================\n")

    while True:
        try:
            prompt = input("Enter prompt > ").strip()
            if not prompt:
                continue
            if prompt.lower() in ("exit", "quit"):
                print("Exiting. Goodbye!")
                break
                
            result = orchestrator.process_and_execute(prompt)
            print("-" * 70)
            if "plan" in result:
                print("Generated Plan:")
                for i, step in enumerate(result["plan"], 1):
                    fallback_flag = " (Fallback)" if result.get("is_fallback") else ""
                    print(f"  Step {i}{fallback_flag}: {step['description']} [{step['tool']}]")
                    print(f"    Prompt: {step['prompt']}")
                print("-" * 70)
            print(f"Identified Intent:  {result['high_level']} -> {result['low_level']}")
            print(f"Triggered Tools:    {result['dispatched_tools']}")
            print(f"Confidence score:   {result['confidence']}")
            print("-" * 70)
            print("Execution Outputs:")
            if "steps" in result:
                for i, step in enumerate(result["steps"], 1):
                    print(f"\n[Step {i}] Execution Result of {step['tool']}:")
                    print(step["output"])
            else:
                for output in result["execution_results"]:
                    print(output)
            print("-" * 70 + "\n")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting. Goodbye!")
            break
