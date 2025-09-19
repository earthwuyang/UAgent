#!/usr/bin/env python3
"""
Test Tree View Components and Integration
Verifies that all tree view components are properly integrated
"""

import os
import json
import requests
from pathlib import Path

def analyze_tree_components():
    """Analyze tree view components in the frontend"""
    frontend_path = Path("/workspace/project/UAgent/frontend/src")
    
    components = {
        "tree_visualization": frontend_path / "components/research/ResearchTreeVisualization.tsx",
        "research_dashboard": frontend_path / "components/research/ResearchDashboard.tsx", 
        "progress_stream": frontend_path / "components/research/ResearchProgressStream.tsx",
        "main_app": frontend_path / "App.tsx",
        "layout": frontend_path / "components/Layout.tsx"
    }
    
    analysis = {}
    
    for name, path in components.items():
        if path.exists():
            with open(path, 'r') as f:
                content = f.read()
                
            analysis[name] = {
                "exists": True,
                "size": len(content),
                "has_reactflow": "ReactFlow" in content,
                "has_websocket": "WebSocket" in content or "ws://" in content,
                "has_tree_integration": "ResearchTreeVisualization" in content,
                "has_api_calls": "fetch(" in content or "axios" in content or "api" in content.lower(),
                "lines": len(content.split('\n'))
            }
        else:
            analysis[name] = {"exists": False}
    
    return analysis

def test_backend_tree_endpoints():
    """Test backend endpoints that provide tree data"""
    endpoints = {
        "health": "http://localhost:12000/health",
        "research_sessions": "http://localhost:12000/api/research/sessions",
        "deep_research": "http://localhost:12000/api/research/deep"
    }
    
    results = {}
    
    for name, url in endpoints.items():
        try:
            if name == "deep_research":
                # POST request for deep research
                response = requests.post(url, json={
                    "query": "Test tree view functionality",
                    "max_iterations": 1
                }, timeout=10)
            else:
                # GET request
                response = requests.get(url, timeout=10)
            
            results[name] = {
                "status_code": response.status_code,
                "success": response.status_code == 200,
                "response_size": len(response.text) if response.text else 0
            }
            
            if response.status_code == 200:
                try:
                    data = response.json()
                    if name == "research_sessions":
                        results[name]["session_count"] = len(data) if isinstance(data, list) else 0
                    elif name == "deep_research":
                        results[name]["research_id"] = data.get("research_id", "No ID")
                        results[name]["status"] = data.get("status", "Unknown")
                except:
                    results[name]["json_parseable"] = False
            
        except Exception as e:
            results[name] = {
                "success": False,
                "error": str(e)
            }
    
    return results

def check_package_dependencies():
    """Check if required packages are installed"""
    package_json_path = Path("/workspace/project/UAgent/frontend/package.json")
    
    if not package_json_path.exists():
        return {"error": "package.json not found"}
    
    with open(package_json_path, 'r') as f:
        package_data = json.load(f)
    
    dependencies = package_data.get("dependencies", {})
    dev_dependencies = package_data.get("devDependencies", {})
    
    required_packages = [
        "react", "react-dom", "reactflow", "typescript", 
        "vite", "@types/react", "tailwindcss"
    ]
    
    analysis = {
        "total_dependencies": len(dependencies) + len(dev_dependencies),
        "has_reactflow": "reactflow" in dependencies,
        "has_react": "react" in dependencies,
        "has_typescript": "typescript" in dev_dependencies,
        "has_vite": "vite" in dev_dependencies,
        "missing_packages": []
    }
    
    for pkg in required_packages:
        if pkg not in dependencies and pkg not in dev_dependencies:
            analysis["missing_packages"].append(pkg)
    
    return analysis

def create_test_report():
    """Create comprehensive test report"""
    print("🌳 UAgent Tree View Component Analysis")
    print("=" * 50)
    
    # Test 1: Component Analysis
    print("\n📁 Component Analysis:")
    components = analyze_tree_components()
    
    for name, info in components.items():
        if info.get("exists"):
            print(f"✅ {name}: {info['lines']} lines")
            if info.get("has_reactflow"):
                print(f"   🔄 ReactFlow integration: YES")
            if info.get("has_websocket"):
                print(f"   🌐 WebSocket support: YES")
            if info.get("has_tree_integration"):
                print(f"   🌳 Tree integration: YES")
        else:
            print(f"❌ {name}: NOT FOUND")
    
    # Test 2: Backend Endpoints
    print("\n🔧 Backend Endpoint Tests:")
    backend_results = test_backend_tree_endpoints()
    
    for name, result in backend_results.items():
        if result.get("success"):
            print(f"✅ {name}: Status {result['status_code']}")
            if "session_count" in result:
                print(f"   📊 Sessions: {result['session_count']}")
            if "research_id" in result:
                print(f"   🆔 Research ID: {result['research_id'][:8]}...")
        else:
            print(f"❌ {name}: {result.get('error', 'Failed')}")
    
    # Test 3: Package Dependencies
    print("\n📦 Package Dependencies:")
    deps = check_package_dependencies()
    
    if "error" not in deps:
        print(f"✅ Total packages: {deps['total_dependencies']}")
        print(f"✅ ReactFlow: {'YES' if deps['has_reactflow'] else 'NO'}")
        print(f"✅ React: {'YES' if deps['has_react'] else 'NO'}")
        print(f"✅ TypeScript: {'YES' if deps['has_typescript'] else 'NO'}")
        
        if deps['missing_packages']:
            print(f"⚠️  Missing packages: {', '.join(deps['missing_packages'])}")
        else:
            print("✅ All required packages present")
    else:
        print(f"❌ {deps['error']}")
    
    # Summary
    print("\n📊 Summary:")
    component_count = sum(1 for info in components.values() if info.get("exists"))
    backend_success = sum(1 for result in backend_results.values() if result.get("success"))
    
    print(f"Components found: {component_count}/{len(components)}")
    print(f"Backend endpoints working: {backend_success}/{len(backend_results)}")
    
    if component_count >= 3 and backend_success >= 2:
        print("🎉 Tree view functionality appears to be properly integrated!")
        
        # Create a research task for testing
        if backend_results.get("deep_research", {}).get("success"):
            research_id = backend_results["deep_research"].get("research_id")
            print(f"\n🧪 Test Research Created: {research_id}")
            print("📋 To test tree view:")
            print("1. Open frontend in browser")
            print("2. Navigate to Research Dashboard")
            print("3. Click on 'Tree View' tab")
            print("4. Observe the tree visualization")
    else:
        print("⚠️  Tree view functionality may have issues")
    
    return {
        "components": components,
        "backend": backend_results,
        "dependencies": deps
    }

if __name__ == "__main__":
    create_test_report()