ALLOWED_MODULES = {
    "pandas",
    "matplotlib",
    "matplotlib.pyplot"
}

FORBIDDEN_KEYWORDS = [
    "import os",
    "import sys",
    "subprocess",
    "socket",
    "eval(",
    "exec(",
]

def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name in ALLOWED_MODULES:
        return __import__(name, globals, locals, fromlist, level)
    raise ImportError(f"Import not allowed: {name}")

def extract_python_code(llm_output: str) -> str:
    if "```" not in llm_output:
        return llm_output.strip()

    parts = llm_output.split("```")
    for part in parts:
        if "import pandas" in part:
            return part.replace("python", "").strip()

    raise ValueError("No executable Python code found.")

def validate_code(code: str):
    for keyword in FORBIDDEN_KEYWORDS:
        if keyword in code:
            raise ValueError(f"Forbidden operation detected: {keyword}")

def execute_code(code: str):
    clean_code = extract_python_code(code)
    validate_code(clean_code)

    safe_globals = {
        "__builtins__": {
            "print": print,
            "range": range,
            "len": len,
            "str": str,
            "int": int,
            "float": float,
            "list": list,
            "dict": dict,
            "set": set,
            "__import__": safe_import
        }
    }

    safe_locals = {}

    print("===== EXECUTING CLEAN CODE =====")
    print(clean_code)
    print("================================")

    exec(clean_code, safe_globals, safe_locals)

    return {
        "result_df": safe_locals.get("result_df", None)
    }
