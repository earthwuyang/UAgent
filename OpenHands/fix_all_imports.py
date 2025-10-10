#!/usr/bin/env python3
"""
CRITICAL BLOCKER FIX: Standardize all imports to absolute imports

This script converts all relative and bare imports to absolute imports
with the extensions.uagent_research prefix.

Usage:
    python fix_all_imports.py
"""

import os
import re
from pathlib import Path
from typing import List, Tuple

# Define all the import transformations
# Format: (file_pattern, old_import, new_import)
IMPORT_FIXES = [
    # orchestrator/tree_orchestrator.py
    ("extensions/uagent_research/orchestrator/tree_orchestrator.py", [
        ("from ..uagent_research.models.research_tree import", "from extensions.uagent_research.uagent_research.models.research_tree import"),
        ("from ..uagent_research.models.events import", "from extensions.uagent_research.uagent_research.models.events import"),
        ("from ..adapters.base.agent_adapter import", "from extensions.uagent_research.adapters.base.agent_adapter import"),
        ("from ..uagent_research.engines", "from extensions.uagent_research.uagent_research.engines"),
        ("from ..api.websocket_routes import", "from extensions.uagent_research.api.websocket_routes import"),
        ("from ..services.idea_generation_service import", "from extensions.uagent_research.services.idea_generation_service import"),
    ]),
    
    # middleware/research_middleware.py
    ("extensions/uagent_research/middleware/research_middleware.py", [
        ("from classifier.task_classifier import", "from extensions.uagent_research.classifier.task_classifier import"),
        ("from orchestrator.tree_orchestrator import", "from extensions.uagent_research.orchestrator.tree_orchestrator import"),
        ("from uagent_research.models.research_tree import", "from extensions.uagent_research.uagent_research.models.research_tree import"),
        ("from config import", "from extensions.uagent_research.config import"),
        ("from control.control_bus import", "from extensions.uagent_research.control.control_bus import"),
        ("from services.research_session_manager import", "from extensions.uagent_research.services.research_session_manager import"),
        ("from orchestrator.event_bus import", "from extensions.uagent_research.orchestrator.event_bus import"),
        ("from adapters.ensure_adapters import", "from extensions.uagent_research.adapters.ensure_adapters import"),
    ]),
    
    # adapters
    ("extensions/uagent_research/adapters/deepresearch/adapter.py", [
        ("from ...adapters.base.agent_adapter import", "from extensions.uagent_research.adapters.base.agent_adapter import"),
        ("from ...uagent_research", "from extensions.uagent_research.uagent_research"),
    ]),
    
    ("extensions/uagent_research/adapters/repomaster/adapter.py", [
        ("from ...adapters.base.agent_adapter import", "from extensions.uagent_research.adapters.base.agent_adapter import"),
        ("from ...uagent_research", "from extensions.uagent_research.uagent_research"),
    ]),
    
    ("extensions/uagent_research/adapters/codeact/adapter.py", [
        ("from ...adapters.base.agent_adapter import", "from extensions.uagent_research.adapters.base.agent_adapter import"),
        ("from ...uagent_research", "from extensions.uagent_research.uagent_research"),
    ]),
    
    ("extensions/uagent_research/adapters/base/agent_adapter.py", [
        ("from ...uagent_research.models", "from extensions.uagent_research.uagent_research.models"),
    ]),
    
    # services
    ("extensions/uagent_research/services/idea_generation_service.py", [
        ("from ..uagent_research.engines", "from extensions.uagent_research.uagent_research.engines"),
        ("from ..uagent_research.models", "from extensions.uagent_research.uagent_research.models"),
    ]),
    
    # tools
    ("extensions/uagent_research/tools/search/bing_search_tool.py", [
        ("from ..common", "from extensions.uagent_research.tools.common"),
    ]),
    
    ("extensions/uagent_research/tools/browse/web_browse_tool.py", [
        ("from ..common", "from extensions.uagent_research.tools.common"),
    ]),
    
    # orchestrator/event_bus.py
    ("extensions/uagent_research/orchestrator/event_bus.py", [
        ("from ..uagent_research.models.events import", "from extensions.uagent_research.uagent_research.models.events import"),
        ("from ..api.websocket_routes import", "from extensions.uagent_research.api.websocket_routes import"),
    ]),
    
    # router
    ("extensions/uagent_research/router/skill_router.py", [
        ("from ..uagent_research", "from extensions.uagent_research.uagent_research"),
        ("from ..adapters", "from extensions.uagent_research.adapters"),
    ]),
    
    # bridges
    ("extensions/uagent_research/bridges/openhands_bridge.py", [
        ("from ..uagent_research", "from extensions.uagent_research.uagent_research"),
        ("from ..adapters", "from extensions.uagent_research.adapters"),
    ]),
]

# session.py fix
SESSION_FIX = ("openhands/server/session/session.py", [
    ("from middleware.research_middleware import research_middleware", 
     "from extensions.uagent_research.middleware.research_middleware import research_middleware"),
    ("from extensions.uagent_research.middleware.research_middleware import research_middleware",
     "from extensions.uagent_research.middleware.research_middleware import research_middleware"),  # Idempotent
])

def fix_imports_in_file(filepath: str, replacements: List[Tuple[str, str]]) -> int:
    """Fix imports in a single file"""
    if not os.path.exists(filepath):
        print(f"   ⚠️  File not found: {filepath}")
        return 0
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    original_content = content
    changes_made = 0
    
    for old_import, new_import in replacements:
        if old_import in content:
            content = content.replace(old_import, new_import)
            changes_made += 1
            print(f"      ✓ {old_import[:50]}... → {new_import[:50]}...")
    
    if content != original_content:
        with open(filepath, 'w') as f:
            f.write(content)
        return changes_made
    
    return 0

def main():
    print("=" * 80)
    print("CRITICAL BLOCKER FIX: Standardizing All Imports to Absolute")
    print("=" * 80)
    print()
    
    total_files = 0
    total_changes = 0
    
    # Fix all files
    all_fixes = IMPORT_FIXES + [SESSION_FIX]
    
    for filepath, replacements in all_fixes:
        print(f"📝 Processing: {filepath}")
        changes = fix_imports_in_file(filepath, replacements)
        if changes > 0:
            total_files += 1
            total_changes += changes
            print(f"   ✅ Made {changes} changes")
        else:
            print(f"   ℹ️  No changes needed (or file not found)")
        print()
    
    print("=" * 80)
    print(f"✅ COMPLETE: Fixed {total_changes} imports in {total_files} files")
    print("=" * 80)
    print()
    print("Next steps:")
    print("1. Run: python -c \"from extensions.uagent_research.middleware.research_middleware import research_middleware; print('✅ Import works!')\"")
    print("2. Start OpenHands server: poetry run python openhands/server/listen.py")
    print("3. Test auto-trigger by sending a researchy prompt in chat")

if __name__ == "__main__":
    main()
