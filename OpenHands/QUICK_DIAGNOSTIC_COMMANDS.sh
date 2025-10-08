#!/bin/bash
# Quick Diagnostic Commands for Parallel Research Debugging

echo "=== OpenHands Parallel Research Diagnostic Tool ==="
echo ""

LOG_FILE="${1:-logs/openhands.log}"

if [ ! -f "$LOG_FILE" ]; then
    echo "❌ Log file not found: $LOG_FILE"
    echo "Usage: $0 [path/to/logfile.log]"
    exit 1
fi

echo "📁 Analyzing log file: $LOG_FILE"
echo ""

# Check 1: Orchestrator Started
echo "1️⃣  Checking if Orchestrator Started..."
if grep -q "\[ORCHESTRATOR\] run() called" "$LOG_FILE"; then
    echo "   ✅ Orchestrator started"
    grep "\[ORCHESTRATOR\] run() called" "$LOG_FILE" | tail -1
else
    echo "   ❌ Orchestrator NOT started - Check middleware initialization"
fi
echo ""

# Check 2: Adapters Registered
echo "2️⃣  Checking Adapter Registration..."
if grep -q "\[ADAPTER_REGISTRY\].*adapters" "$LOG_FILE"; then
    adapter_count=$(grep "\[ADAPTER_REGISTRY\] Current state:" "$LOG_FILE" | tail -1 | grep -oP '\d+' | head -1)
    if [ -n "$adapter_count" ] && [ "$adapter_count" -ge 3 ]; then
        echo "   ✅ $adapter_count adapters registered"
    else
        echo "   ⚠️  Only $adapter_count adapters registered (expected 3)"
    fi
    grep "\[ADAPTER_REGISTRY\]" "$LOG_FILE" | tail -5
else
    echo "   ❌ No adapter registration logs found"
fi
echo ""

# Check 3: PUCT Loop Running
echo "3️⃣  Checking PUCT Loop Execution..."
puct_count=$(grep -c "\[ORCHESTRATOR\] PUCT iteration" "$LOG_FILE")
if [ "$puct_count" -gt 0 ]; then
    echo "   ✅ PUCT loop ran $puct_count iterations"
    grep "\[ORCHESTRATOR\] PUCT iteration" "$LOG_FILE" | tail -3
else
    echo "   ❌ PUCT loop NOT running - Check orchestrator.run()"
fi
echo ""

# Check 4: Node Expansion
echo "4️⃣  Checking Node Expansion..."
if grep -q "\[EXPAND\] Generated.*children" "$LOG_FILE"; then
    echo "   ✅ Nodes expanding"
    grep "\[EXPAND\] Generated.*children" "$LOG_FILE" | tail -3
else
    echo "   ❌ Nodes NOT expanding - Check _expand_node() and LLM"
fi
echo ""

# Check 5: Parallel Task Spawning
echo "5️⃣  Checking Parallel Task Execution..."
if grep -q "\[EXECUTE\] Starting parallel execution" "$LOG_FILE"; then
    echo "   ✅ Parallel tasks spawning"
    grep "\[EXECUTE\] Starting parallel execution" "$LOG_FILE" | tail -3
else
    echo "   ❌ Parallel tasks NOT spawning - Check _execute_children_parallel()"
fi
echo ""

# Check 6: Adapter Execution
echo "6️⃣  Checking Adapter Execution..."
adapter_calls=$(grep -c "DEEPRESEARCH\|REPOMASTER\|CODEACT.*run() called" "$LOG_FILE")
if [ "$adapter_calls" -gt 0 ]; then
    echo "   ✅ Adapters executed $adapter_calls times"
    grep "DEEPRESEARCH\|REPOMASTER\|CODEACT.*run() called" "$LOG_FILE" | tail -5
else
    echo "   ❌ Adapters NOT executing - Check adapter routing"
fi
echo ""

# Check 7: Errors
echo "7️⃣  Checking for Errors..."
error_count=$(grep -c "ERROR\|CRITICAL\|Exception" "$LOG_FILE")
if [ "$error_count" -gt 0 ]; then
    echo "   ⚠️  Found $error_count errors - Review below:"
    grep "ERROR\|CRITICAL" "$LOG_FILE" | tail -10
else
    echo "   ✅ No errors found"
fi
echo ""

# Summary
echo "=== Summary ==="
echo "Orchestrator: $(grep -q '\[ORCHESTRATOR\] run() called' $LOG_FILE && echo '✅' || echo '❌')"
echo "Adapters: $(grep -q '\[ADAPTER_REGISTRY\].*3.*adapters' $LOG_FILE && echo '✅' || echo '⚠️')"
echo "PUCT Loop: $([ $puct_count -gt 0 ] && echo "✅ ($puct_count)" || echo '❌')"
echo "Expansion: $(grep -q '\[EXPAND\] Generated.*children' $LOG_FILE && echo '✅' || echo '❌')"
echo "Parallel: $(grep -q '\[EXECUTE\] Starting' $LOG_FILE && echo '✅' || echo '❌')"
echo "Adapters Running: $([ $adapter_calls -gt 0 ] && echo "✅ ($adapter_calls)" || echo '❌')"
echo "Errors: $([ $error_count -gt 0 ] && echo "⚠️ ($error_count)" || echo '✅')"
echo ""
echo "💡 Run individual checks:"
echo "   grep '\[ORCHESTRATOR\]' $LOG_FILE | less"
echo "   grep '\[EXPAND\]' $LOG_FILE | less"
echo "   grep '\[EXECUTE\]' $LOG_FILE | less"
