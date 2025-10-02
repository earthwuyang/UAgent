#!/usr/bin/env python3
"""Test script to verify environment variables are being read correctly"""

import os
import sys
sys.path.insert(0, '/home/wuy/AI/UAgent/backend')

from dotenv import load_dotenv

# Load .env file
load_dotenv('../.env')
load_dotenv()

# Check environment variables
print("=" * 60)
print("Environment Variables Test")
print("=" * 60)

vars_to_check = [
    "MAX_RESEARCH_IDEAS",
    "MAX_PARALLEL_IDEAS",
    "EXPERIMENTS_PER_HYPOTHESIS"
]

for var in vars_to_check:
    value = os.getenv(var)
    print(f"{var:30s} = {value!r}")

print()
print("=" * 60)
print("How ScientificResearchEngine will interpret them:")
print("=" * 60)

# Simulate how the code reads them
max_ideas = int(os.getenv("MAX_RESEARCH_IDEAS", "3"))
max_parallel = int(os.getenv("MAX_PARALLEL_IDEAS", "2"))
experiments_per_hyp = max(1, int(os.getenv("EXPERIMENTS_PER_HYPOTHESIS", "2")))

print(f"max_ideas (from MAX_RESEARCH_IDEAS):        {max_ideas}")
print(f"max_parallel (from MAX_PARALLEL_IDEAS):     {max_parallel}")
print(f"experiments_per_hypothesis:                 {experiments_per_hyp}")

print()
print("✓ All values loaded correctly!" if all([
    max_ideas == 3,
    max_parallel == 3,
    experiments_per_hyp == 2
]) else "✗ Values don't match expected!")
