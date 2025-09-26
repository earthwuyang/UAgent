TOOL_SPEC = {
    "tools": {
        "create_if_absent": {
            "desc": "Create a new text file only if it does not exist (idempotent).",
            "args": {"path": "string", "content": "string"},
            "returns": {"success": "bool", "message": "string"},
        },
        "write": {
            "desc": "Write/overwrite a text file (atomic).",
            "args": {"path": "string", "content": "string", "overwrite": "bool (default true)"},
            "returns": {"success": "bool", "message": "string"},
        },
        "read": {
            "desc": "Read a text file.",
            "args": {"path": "string"},
            "returns": {"success": "bool", "content": "string"},
        },
        "run": {
            "desc": "Execute a shell command in a fresh process. Prefer ./.venv/bin/python and ./.venv/bin/pip.",
            "args": {"cmd": "string", "timeout_sec": "int (default 300)"},
            "returns": {"success": "bool", "exit_code": "int", "stdout": "string", "stderr": "string"},
        },
        "finish": {
            "desc": "Return final result as strict JSON (no code fences).",
            "args": {"result_json": "object"},
            "returns": {"ok": "bool"},
        },
    },
    "rules": [
        "If a file must exist and is absent, call create_if_absent.",
        "If a file exists and must change, call write(overwrite=true).",
        "Never loop: if an action returns success, move on.",
        "Always use ./.venv/bin/python and ./.venv/bin/pip once the venv exists.",
    ],
}

