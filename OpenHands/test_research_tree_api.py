#!/usr/bin/env python3
"""
Test script to verify research tree API is working
"""

import requests
import json

# Your experiment ID
experiment_id = "exp_6411470ef5854f71bfecdfd7b6689330_1759761226_4991e0f1"

# API endpoint
url = f"http://120.46.207.248:3000/api/research/experiments/{experiment_id}/tree"

print(f"Testing research tree API...")
print(f"URL: {url}")
print()

try:
    response = requests.get(url)
    print(f"Status Code: {response.status_code}")
    print(f"Headers: {dict(response.headers)}")
    print()

    if response.status_code == 200:
        data = response.json()
        print("Response Data:")
        print(json.dumps(data, indent=2))
        print()

        # Check if tree has data
        tree_data = data.get('data', {})
        nodes = tree_data.get('nodes', [])
        edges = tree_data.get('edges', [])
        stats = tree_data.get('stats', {})

        print(f"Summary:")
        print(f"  Nodes: {len(nodes)}")
        print(f"  Edges: {len(edges)}")
        print(f"  Stats: {stats}")

        if nodes:
            print(f"\n  First node:")
            print(f"    {json.dumps(nodes[0], indent=4)}")
        else:
            print(f"\n  ⚠️  No nodes in tree!")

    else:
        print(f"Error: {response.text}")

except Exception as e:
    print(f"Exception: {e}")
    import traceback
    traceback.print_exc()
