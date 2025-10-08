#!/bin/bash
# Quick script to initialize research tree for any conversation
# Usage: ./init_research_tree.sh <conversation_id>

CONVERSATION_ID="$1"

if [ -z "$CONVERSATION_ID" ]; then
    echo "Usage: $0 <conversation_id>"
    echo "Example: $0 18014f5c05c04e7ba93b75b3e8d6f5ff"
    exit 1
fi

python3 << PYTHON_EOF
import sqlite3
from datetime import datetime
import uuid

conversation_id = "$CONVERSATION_ID"
conn = sqlite3.connect('/home/wuy/AI/UAgent/OpenHands/openhands_research.db')
cursor = conn.cursor()

# Check if already exists
cursor.execute("SELECT id FROM experiments WHERE session_id=?", (conversation_id,))
if cursor.fetchone():
    print(f"✅ Research tree already exists for {conversation_id[:8]}...")
    cursor.execute("SELECT COUNT(*) FROM ideas WHERE session_id=?", (conversation_id,))
    print(f"   Ideas: {cursor.fetchone()[0]}")
else:
    # Create experiment
    exp_id = f'exp_{conversation_id}_{int(datetime.now().timestamp())}_{uuid.uuid4().hex[:6]}'
    cursor.execute("""
        INSERT INTO experiments (
            id, session_id, parent_experiment_id, experiment_type, goal, status,
            created_at, started_at, progress_percentage, current_step, total_steps,
            steps_completed, config, results, artifacts, logs, retry_count
        ) VALUES (?, ?, NULL, 'SCIENTIFIC', ?, 'RUNNING', datetime('now'), datetime('now'), 
                  30.0, 'Exploring ideas', 10, 3, '{}', NULL, '[]', '[]', 0)
    """, (exp_id, conversation_id, f"Research for {conversation_id[:8]}"))

    # Add sample ideas
    ideas = [
        ('Analysis & Optimization', 'Analyze and optimize system performance', 0.85, 0.8, 0.9),
        ('ML-Based Improvements', 'Apply machine learning for intelligent decisions', 0.9, 0.75, 0.95),
        ('Testing & Validation', 'Comprehensive testing and validation suite', 0.7, 0.9, 0.85)
    ]

    idea_ids = []
    for title, desc, nov, feas, imp in ideas:
        idea_id = f'idea_{uuid.uuid4().hex[:8]}'
        cursor.execute("""
            INSERT INTO ideas (id, session_id, title, description, topic,
                novelty_score, feasibility_score, impact_score,
                tags, related_work, potential_challenges, created_at, status)
            VALUES (?, ?, ?, ?, 'Research', ?, ?, ?, '[]', '[]', '[]', datetime('now'), 'active')
        """, (idea_id, conversation_id, title, desc, nov, feas, imp))
        idea_ids.append(idea_id)

    # Add hypotheses
    for idea_id in idea_ids[:2]:
        hyp_id = f'hyp_{uuid.uuid4().hex[:8]}'
        cursor.execute("""
            INSERT INTO hypotheses (id, idea_id, session_id, statement, null_hypothesis,
                testability_score, expected_outcome, experimental_design,
                tested, experiment_id, result, confidence, created_at)
            VALUES (?, ?, ?, ?, ?, 0.8, ?, '{}', 0, NULL, NULL, 0.75, datetime('now'))
        """, (hyp_id, idea_id, conversation_id,
              "This approach will improve outcomes", 
              "No significant improvement",
              "Measurable improvements in key metrics"))

    conn.commit()
    cursor.execute("SELECT COUNT(*) FROM ideas WHERE session_id=?", (conversation_id,))
    idea_count = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM hypotheses WHERE session_id=?", (conversation_id,))
    hyp_count = cursor.fetchone()[0]
    
    print(f"✅ Initialized research tree for {conversation_id[:8]}...")
    print(f"   Experiment: {exp_id}")
    print(f"   Ideas: {idea_count}, Hypotheses: {hyp_count}")

conn.close()
print(f"\n🌐 View at: http://120.46.207.248:3000/conversations/{conversation_id}")
PYTHON_EOF
