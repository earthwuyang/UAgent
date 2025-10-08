#!/bin/bash
# Verify that all node types have required MCTS fields after the fix

CONVERSATION_ID="${1:-18014f5c05c04e7ba93b75b3e8d6f5ff}"
API_URL="http://120.46.207.248:3001/api/research/${CONVERSATION_ID}/tree"

echo "🔍 Verifying MCTS fields for conversation: ${CONVERSATION_ID}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Fetch tree data
response=$(curl -s "$API_URL")

if [ -z "$response" ]; then
    echo "❌ Failed to fetch tree data from API"
    echo "   Make sure the server is running at http://120.46.207.248:3001"
    exit 1
fi

# Check if response is valid JSON
if ! echo "$response" | python3 -m json.tool > /dev/null 2>&1; then
    echo "❌ Invalid JSON response from API"
    echo "Response: $response"
    exit 1
fi

echo "✅ API is responding with valid JSON"
echo ""

# Extract node types and check MCTS fields
echo "Checking node types and MCTS fields:"
echo "─────────────────────────────────────"

for node_type in root idea hypothesis; do
    echo ""
    echo "📊 ${node_type^^} nodes:"
    
    # Count nodes of this type
    count=$(echo "$response" | python3 -c "
import sys, json
data = json.load(sys.stdin)
nodes = data.get('data', {}).get('nodes', [])
count = sum(1 for n in nodes if n.get('type') == '${node_type}')
print(count)
" 2>/dev/null)
    
    echo "   Total: $count"
    
    if [ "$count" -gt 0 ]; then
        # Check if all have required fields
        echo "$response" | python3 << PYTHON
import sys, json

data = json.load(sys.stdin)
nodes = data.get('data', {}).get('nodes', [])
type_nodes = [n for n in nodes if n.get('type') == '${node_type}']

required_fields = ['avg_value', 'prior', 'visit_count', 'puct_score']
all_good = True

for i, node in enumerate(type_nodes, 1):
    node_data = node.get('data', {})
    missing = []
    invalid = []
    
    for field in required_fields:
        if field not in node_data:
            missing.append(field)
            all_good = False
        elif node_data[field] is None or (isinstance(node_data[field], float) and node_data[field] != node_data[field]):  # Check for NaN
            invalid.append(field)
            all_good = False
    
    if missing or invalid:
        print(f"   ❌ Node {i} ({node.get('id', 'unknown')[:15]}...)")
        if missing:
            print(f"      Missing: {', '.join(missing)}")
        if invalid:
            print(f"      Invalid: {', '.join(invalid)}")
    else:
        # Show values for first node
        if i == 1:
            print(f"   ✅ Sample values (node 1):")
            for field in required_fields:
                value = node_data[field]
                if isinstance(value, float):
                    print(f"      {field}: {value:.3f}")
                else:
                    print(f"      {field}: {value}")

if all_good and len(type_nodes) > 0:
    if len(type_nodes) > 1:
        print(f"   ✅ All {len(type_nodes)} nodes have valid MCTS fields")
PYTHON
    fi
done

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🎉 Verification complete!"
echo ""
echo "If all checks passed, the frontend should now render without errors."
echo "Visit: http://120.46.207.248:3000/conversations/${CONVERSATION_ID}"
