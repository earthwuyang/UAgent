#!/usr/bin/env python3
"""
Sync dependencies from OpenHands/pyproject.toml to UAgent/requirements.txt
Handles version conflicts gracefully by preferring newer versions.
"""

import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import toml


def parse_poetry_version(version_spec: str) -> Tuple[str, str]:
    """
    Parse Poetry version specification to pip-compatible format.
    
    Examples:
        "^1.2.3" -> (">=", "1.2.3")
        ">=1.2.3,<2.0.0" -> (">=", "1.2.3,<2.0.0")
        "*" -> ("", "")
        "1.2.3" -> ("==", "1.2.3")
    """
    version_spec = version_spec.strip()
    
    if version_spec == "*":
        return ("", "")
    
    # Handle caret (^) - compatible version
    if version_spec.startswith("^"):
        version = version_spec[1:]
        return (">=", version)
    
    # Handle tilde (~) - approximately equivalent
    if version_spec.startswith("~"):
        version = version_spec[1:]
        return ("~=", version)
    
    # Handle comparison operators
    if any(version_spec.startswith(op) for op in [">=", "<=", "!=", "==", ">", "<"]):
        return ("", version_spec)
    
    # Plain version means exact match
    return ("==", version_spec)


def extract_pyproject_deps(pyproject_path: Path) -> Dict[str, str]:
    """
    Extract dependencies from pyproject.toml.
    
    Returns:
        Dict mapping package name (lowercase) to version spec
    """
    data = toml.load(pyproject_path)
    deps = {}
    
    # Main dependencies
    main_deps = data.get("tool", {}).get("poetry", {}).get("dependencies", {})
    for pkg, spec in main_deps.items():
        if pkg == "python":
            continue
            
        # Handle complex dependency specifications
        if isinstance(spec, dict):
            # Optional dependencies or git dependencies
            if spec.get("optional", False):
                print(f"⏭️  Skipping optional: {pkg}")
                continue
            if "git" in spec:
                print(f"⏭️  Skipping git dependency: {pkg}")
                continue
            # Extract version if available
            version = spec.get("version", "*")
        else:
            version = spec
        
        op, ver = parse_poetry_version(str(version))
        if ver:
            deps[pkg.lower()] = f"{op}{ver}"
        else:
            deps[pkg.lower()] = ""
    
    # Include dev dependencies (optional, uncomment if needed)
    # dev_deps = data.get("tool", {}).get("poetry", {}).get("group", {}).get("dev", {}).get("dependencies", {})
    # for pkg, spec in dev_deps.items():
    #     if isinstance(spec, str):
    #         op, ver = parse_poetry_version(spec)
    #         deps[pkg.lower()] = f"{op}{ver}" if ver else ""
    
    return deps


def parse_requirements_line(line: str) -> Optional[Tuple[str, str]]:
    """
    Parse a requirements.txt line.
    
    Returns:
        (package_name_lower, version_spec) or None if invalid
    """
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    
    # Match package name and version
    match = re.match(r"^([a-zA-Z0-9_-]+)(.*?)$", line)
    if not match:
        return None
    
    pkg_name = match.group(1).lower()
    version_spec = match.group(2).strip()
    
    return (pkg_name, version_spec)


def compare_versions(v1: str, v2: str) -> str:
    """
    Compare two version specifications and return the more restrictive one.
    
    For simplicity, we prefer:
    1. Explicit versions over ranges
    2. Newer versions when both are explicit
    3. v2 (pyproject.toml) in case of conflict
    """
    # If one is empty, prefer the other
    if not v1:
        return v2
    if not v2:
        return v1
    
    # If versions are the same, keep as is
    if v1 == v2:
        return v1
    
    # Extract numeric version for comparison
    v1_num = re.search(r"(\d+\.\d+(?:\.\d+)?)", v1)
    v2_num = re.search(r"(\d+\.\d+(?:\.\d+)?)", v2)
    
    if v1_num and v2_num:
        v1_parts = [int(x) for x in v1_num.group(1).split(".")]
        v2_parts = [int(x) for x in v2_num.group(1).split(".")]
        
        # Pad to same length
        max_len = max(len(v1_parts), len(v2_parts))
        v1_parts += [0] * (max_len - len(v1_parts))
        v2_parts += [0] * (max_len - len(v2_parts))
        
        # Prefer newer version (from pyproject.toml)
        if v2_parts > v1_parts:
            print(f"      ⬆️  Upgrading: {v1} → {v2}")
            return v2
        elif v1_parts > v2_parts:
            print(f"      ⬇️  Keeping newer: {v1} (skipping {v2})")
            return v1
    
    # Default: prefer pyproject.toml version
    print(f"      🔄 Version conflict, preferring pyproject.toml: {v1} → {v2}")
    return v2


def merge_requirements(
    existing_reqs: Dict[str, str],
    pyproject_deps: Dict[str, str]
) -> Dict[str, str]:
    """
    Merge existing requirements with pyproject dependencies.
    """
    merged = existing_reqs.copy()
    
    for pkg, new_ver in pyproject_deps.items():
        if pkg in merged:
            old_ver = merged[pkg]
            merged[pkg] = compare_versions(old_ver, new_ver)
        else:
            print(f"   ➕ Adding new package: {pkg}{new_ver}")
            merged[pkg] = new_ver
    
    return merged


def main():
    # Paths
    base_dir = Path(__file__).parent
    pyproject_path = base_dir / "OpenHands" / "pyproject.toml"
    requirements_path = base_dir / "requirements.txt"
    requirements_backup = base_dir / "requirements.txt.backup"
    
    print("=" * 80)
    print("🔄 Syncing OpenHands dependencies to UAgent requirements.txt")
    print("=" * 80)
    print()
    
    # Read existing requirements
    print("📖 Reading existing requirements.txt...")
    existing_reqs = {}
    if requirements_path.exists():
        with open(requirements_path, "r") as f:
            for line in f:
                parsed = parse_requirements_line(line)
                if parsed:
                    pkg, ver = parsed
                    existing_reqs[pkg] = ver
    
    print(f"   Found {len(existing_reqs)} existing packages")
    print()
    
    # Extract pyproject.toml dependencies
    print("📖 Reading OpenHands/pyproject.toml...")
    pyproject_deps = extract_pyproject_deps(pyproject_path)
    print(f"   Found {len(pyproject_deps)} dependencies")
    print()
    
    # Merge
    print("🔀 Merging dependencies...")
    merged = merge_requirements(existing_reqs, pyproject_deps)
    print(f"   Total packages after merge: {len(merged)}")
    print()
    
    # Backup existing requirements
    if requirements_path.exists():
        print(f"💾 Backing up existing requirements.txt to {requirements_backup.name}")
        requirements_path.rename(requirements_backup)
    
    # Write new requirements.txt
    print(f"✍️  Writing new requirements.txt...")
    with open(requirements_path, "w") as f:
        # Sort by package name
        for pkg in sorted(merged.keys()):
            ver = merged[pkg]
            line = f"{pkg}{ver}\n"
            f.write(line)
    
    print()
    print("=" * 80)
    print("✅ Successfully synced dependencies!")
    print("=" * 80)
    print()
    print("📋 Summary:")
    print(f"   • Total packages: {len(merged)}")
    print(f"   • Backup saved: {requirements_backup}")
    print(f"   • New file: {requirements_path}")
    print()
    print("🔧 Next step: Install with pip")
    print(f"   cd {base_dir}")
    print("   source .venv/bin/activate")
    print("   pip install -r requirements.txt")


if __name__ == "__main__":
    main()
