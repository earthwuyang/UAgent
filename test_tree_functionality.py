#!/usr/bin/env python3
"""
Comprehensive test script for UAgent Tree View functionality.
This script tests the tree view when a user performs a research task.
"""

import requests
import json
import time
import sys
from typing import Dict, Any, List

# Configuration
BACKEND_URL = "http://localhost:12000"
FRONTEND_URL = "http://localhost:12001"
GOAL_ID = "cf55a5d2-5e4d-4eef-84c3-c63d48554d44"

class TreeViewTester:
    def __init__(self):
        self.backend_url = BACKEND_URL
        self.frontend_url = FRONTEND_URL
        self.goal_id = GOAL_ID
        self.test_results = []

    def log_test(self, test_name: str, success: bool, message: str, data: Any = None):
        """Log test results"""
        result = {
            "test": test_name,
            "success": success,
            "message": message,
            "data": data,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        self.test_results.append(result)
        
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}: {message}")
        if data and not success:
            print(f"   Details: {data}")

    def test_backend_health(self) -> bool:
        """Test backend health by checking docs endpoint"""
        try:
            response = requests.get(f"{self.backend_url}/docs", timeout=5)
            if response.status_code == 200:
                self.log_test("Backend Health", True, "Backend is healthy (docs endpoint accessible)")
                return True
            else:
                self.log_test("Backend Health", False, f"HTTP {response.status_code}", response.text)
                return False
        except Exception as e:
            self.log_test("Backend Health", False, f"Connection failed: {str(e)}")
            return False

    def test_research_goal_status(self) -> Dict[str, Any]:
        """Test research goal status endpoint"""
        try:
            response = requests.get(f"{self.backend_url}/api/research-tree/goals/{self.goal_id}/status", timeout=10)
            if response.status_code == 200:
                data = response.json()
                goal_title = data.get('goal', {}).get('title', 'Unknown')
                total_nodes = data.get('tree_stats', {}).get('total_nodes', 0)
                completed_nodes = data.get('tree_stats', {}).get('completed_nodes', 0)
                success_rate = data.get('tree_stats', {}).get('success_rate', 0) * 100
                
                self.log_test("Research Goal Status", True, 
                             f"Goal: {goal_title}, Nodes: {total_nodes}, Completed: {completed_nodes}, Success: {success_rate:.1f}%")
                return data
            else:
                self.log_test("Research Goal Status", False, f"HTTP {response.status_code}", response.text)
                return {}
        except Exception as e:
            self.log_test("Research Goal Status", False, f"Request failed: {str(e)}")
            return {}

    def test_tree_visualization_data(self) -> Dict[str, Any]:
        """Test tree visualization data endpoint"""
        try:
            response = requests.get(f"{self.backend_url}/api/research-tree/goals/{self.goal_id}/visualization", timeout=10)
            if response.status_code == 200:
                data = response.json()
                
                # Analyze the tree structure
                nodes = data.get('all_nodes', {})
                graphs = data.get('graphs', {})
                edges = graphs.get('main_graph', {}).get('edges', [])
                
                node_count = len(nodes)
                edge_count = len(edges)
                project_goal = data.get('overall_project_goal', 'Unknown')
                
                # Analyze node types and statuses
                node_types = {}
                node_statuses = {}
                layers = set()
                
                for node_id, node in nodes.items():
                    node_type = node.get('node_type', 'unknown')
                    status = node.get('status', 'unknown')
                    layer = node.get('layer', 0)
                    
                    node_types[node_type] = node_types.get(node_type, 0) + 1
                    node_statuses[status] = node_statuses.get(status, 0) + 1
                    layers.add(layer)
                
                analysis = {
                    'project_goal': project_goal,
                    'node_count': node_count,
                    'edge_count': edge_count,
                    'node_types': node_types,
                    'node_statuses': node_statuses,
                    'layers': sorted(list(layers))
                }
                
                self.log_test("Tree Visualization Data", True, 
                             f"Retrieved tree with {node_count} nodes, {edge_count} edges, {len(layers)} layers")
                
                return data
            else:
                self.log_test("Tree Visualization Data", False, f"HTTP {response.status_code}", response.text)
                return {}
        except Exception as e:
            self.log_test("Tree Visualization Data", False, f"Request failed: {str(e)}")
            return {}

    def test_tree_structure_validity(self, tree_data: Dict[str, Any]) -> bool:
        """Test if tree structure is valid for visualization"""
        if not tree_data:
            self.log_test("Tree Structure Validity", False, "No tree data provided")
            return False
        
        nodes = tree_data.get('all_nodes', {})
        graphs = tree_data.get('graphs', {})
        
        if not nodes:
            self.log_test("Tree Structure Validity", False, "No nodes found in tree data")
            return False
        
        # Check for root node
        root_nodes = [node for node in nodes.values() if not node.get('parent_node_id')]
        if not root_nodes:
            self.log_test("Tree Structure Validity", False, "No root node found")
            return False
        
        # Check node structure
        required_fields = ['task_id', 'goal', 'node_type', 'status', 'layer']
        for node_id, node in nodes.items():
            missing_fields = [field for field in required_fields if field not in node]
            if missing_fields:
                self.log_test("Tree Structure Validity", False, 
                             f"Node {node_id} missing fields: {missing_fields}")
                return False
        
        # Check edges
        edges = graphs.get('main_graph', {}).get('edges', [])
        for edge in edges:
            if 'source' not in edge or 'target' not in edge:
                self.log_test("Tree Structure Validity", False, "Edge missing source or target")
                return False
            
            if edge['source'] not in nodes or edge['target'] not in nodes:
                self.log_test("Tree Structure Validity", False, 
                             f"Edge references non-existent nodes: {edge['source']} -> {edge['target']}")
                return False
        
        self.log_test("Tree Structure Validity", True, 
                     f"Tree structure is valid with {len(root_nodes)} root node(s)")
        return True

    def test_frontend_accessibility(self) -> bool:
        """Test if frontend is accessible"""
        try:
            response = requests.get(f"{self.frontend_url}/", timeout=5)
            if response.status_code == 200:
                self.log_test("Frontend Accessibility", True, "Frontend is accessible")
                return True
            else:
                self.log_test("Frontend Accessibility", False, f"HTTP {response.status_code}")
                return False
        except Exception as e:
            self.log_test("Frontend Accessibility", False, f"Connection failed: {str(e)}")
            return False

    def display_tree_structure(self, tree_data: Dict[str, Any]):
        """Display tree structure in a readable format"""
        if not tree_data:
            print("\n❌ No tree data to display")
            return
        
        nodes = tree_data.get('all_nodes', {})
        graphs = tree_data.get('graphs', {})
        edges = graphs.get('main_graph', {}).get('edges', [])
        
        print(f"\n🌳 TREE STRUCTURE VISUALIZATION")
        print(f"Project Goal: {tree_data.get('overall_project_goal', 'Unknown')}")
        print(f"Total Nodes: {len(nodes)}")
        print(f"Total Edges: {len(edges)}")
        
        # Group nodes by layer
        layers = {}
        for node_id, node in nodes.items():
            layer = node.get('layer', 0)
            if layer not in layers:
                layers[layer] = []
            layers[layer].append(node)
        
        # Display nodes by layer
        for layer in sorted(layers.keys()):
            print(f"\n📊 Layer {layer}:")
            for node in layers[layer]:
                status_icon = self.get_status_icon(node.get('status', 'unknown'))
                node_icon = self.get_node_icon(node.get('node_type', 'unknown'))
                print(f"  {node_icon} {status_icon} {node.get('goal', 'No goal')[:60]}...")
                print(f"    ID: {node.get('task_id', 'unknown')}")
                print(f"    Type: {node.get('node_type', 'unknown')}")
                print(f"    Status: {node.get('status', 'unknown')}")
                if node.get('agent_name'):
                    print(f"    Agent: {node.get('agent_name')}")
                if node.get('output_summary'):
                    print(f"    Summary: {node.get('output_summary', '')[:80]}...")
                print()
        
        # Display edges
        if edges:
            print("🔗 CONNECTIONS:")
            for edge in edges:
                source_node = nodes.get(edge['source'], {})
                target_node = nodes.get(edge['target'], {})
                print(f"  {source_node.get('goal', 'Unknown')[:30]}... → {target_node.get('goal', 'Unknown')[:30]}...")

    def get_status_icon(self, status: str) -> str:
        """Get icon for node status"""
        icons = {
            'DONE': '✅',
            'RUNNING': '🔄',
            'FAILED': '❌',
            'READY': '▶️',
            'PENDING': '⏳',
            'PLAN_DONE': '📋'
        }
        return icons.get(status, '⚪')

    def get_node_icon(self, node_type: str) -> str:
        """Get icon for node type"""
        icons = {
            'PLAN': '🧠',
            'EXECUTE': '⚙️',
            'research': '🔍',
            'computational': '💻'
        }
        return icons.get(node_type, '⚙️')

    def run_all_tests(self):
        """Run all tests"""
        print("🚀 Starting UAgent Tree View Functionality Tests")
        print("=" * 60)
        
        # Test 1: Backend Health
        backend_healthy = self.test_backend_health()
        
        if not backend_healthy:
            print("\n❌ Backend is not healthy. Cannot proceed with other tests.")
            return False
        
        # Test 2: Research Goal Status
        goal_status = self.test_research_goal_status()
        
        # Test 3: Tree Visualization Data
        tree_data = self.test_tree_visualization_data()
        
        # Test 4: Tree Structure Validity
        structure_valid = self.test_tree_structure_validity(tree_data)
        
        # Test 5: Frontend Accessibility
        frontend_accessible = self.test_frontend_accessibility()
        
        # Display tree structure
        if tree_data:
            self.display_tree_structure(tree_data)
        
        # Summary
        print("\n" + "=" * 60)
        print("📊 TEST SUMMARY")
        print("=" * 60)
        
        passed = sum(1 for result in self.test_results if result['success'])
        total = len(self.test_results)
        
        print(f"Tests Passed: {passed}/{total}")
        print(f"Success Rate: {(passed/total*100):.1f}%")
        
        if passed == total:
            print("🎉 All tests passed! Tree view functionality is working correctly.")
        else:
            print("⚠️  Some tests failed. Check the details above.")
        
        return passed == total

def main():
    """Main function"""
    tester = TreeViewTester()
    success = tester.run_all_tests()
    
    if success:
        print("\n✅ CONCLUSION: Tree view functionality is working correctly!")
        print("The system can:")
        print("  • Connect to the backend API")
        print("  • Retrieve research goal status")
        print("  • Get tree visualization data")
        print("  • Display hierarchical node structure")
        print("  • Show node relationships and status")
        sys.exit(0)
    else:
        print("\n❌ CONCLUSION: Some issues were found with tree view functionality.")
        sys.exit(1)

if __name__ == "__main__":
    main()