# UAgent Experimental Failures: Root Cause Analysis & Generic Solution

## Executive Summary

Your UAgent system's scientific research experiments fail because the agent doesn't know how to work around container limitations. **The solution is NOT to pre-install domain-specific tools** (PostgreSQL, DuckDB, etc.) but to **teach the agent to download and build ANY software from source locally**.

This document explains:
1. Why experiments fail (root cause)
2. The **generic solution** that works for ANY research domain
3. How to implement it

## Root Cause Analysis

### The Real Problem (From Your Paper)

Your reproducibility paper states:
> "No access to install system packages or modify core database functionality"
> "Container environment limitations prevented actual Docker deployment"
> "Currently does not support sending input or signals to active processes"

The agent **gives up** when it can't `apt install postgresql` or modify `/usr/bin/postgres`.

### Why This Happens

1. **Container Restrictions**: Docker containers can't `apt install` or modify system software
2. **Missing Guidance**: Prompts say "don't simulate" but don't explain HOW to get real data
3. **No Alternative Strategy**: Agent doesn't know it can build software locally

### The Mistake in Previous Solutions

**❌ Wrong Approach**: Pre-install PostgreSQL, DuckDB, Redis, etc. in Docker image
- **Problem**: Hardcodes specific research domains
- **Limitation**: Only works for database research
- **Not Scalable**: What about ML frameworks? Compilers? Web servers?

**✅ Right Approach**: Provide generic build tools + teach local compilation strategy
- **Benefit**: Works for ANY open-source software
- **Scalability**: One solution for all research domains
- **Flexibility**: Agent can modify source code as needed

## The Generic Solution

### Core Insight

**If the agent needs software X but can't install it system-wide:**
1. Clone X's source code (`git clone`)
2. Build X locally (`./configure --prefix=/workspace/local && make`)
3. Modify X's source if needed
4. Use the local build (`export PATH=/workspace/local/bin:$PATH`)

This works for:
- ✅ PostgreSQL (database kernel research)
- ✅ DuckDB (columnar database research)
- ✅ Redis (cache research)
- ✅ NGINX (web server research)
- ✅ Python interpreters (language research)
- ✅ Compilers (systems research)
- ✅ ML frameworks (AI research)
- ✅ **Any open-source software**

### Implementation: 3 Components

#### Component 1: Generic Build Tools Docker Image

**File**: `docker/research-runtime.Dockerfile`

Includes ONLY generic tools:
- Build tools: gcc, g++, cmake, autoconf, make
- Version control: git, svn, mercurial
- Download tools: wget, curl
- Dev libraries: libssl-dev, zlib1g-dev, libreadline-dev, etc.
- Python: numpy, pandas, scipy, scikit-learn (common)

**Does NOT include domain-specific software** like PostgreSQL, DuckDB, Redis.

Build it:
```bash
cd /Users/wuy/Desktop/code/UAgent
docker build -t earthwuyang/uagent-research:v1.0 -f docker/research-runtime.Dockerfile .
```

Update `.env`:
```bash
UAGENT_OPENHANDS_IMAGE=earthwuyang/uagent-research:v1.0
```

#### Component 2: Generic Experiment Guidance

**File**: `backend/app/core/research_engines/scientific_research.py` (lines 1178-1272)

Added comprehensive guidance that:
- Lists available build tools
- Explains container limitations
- **Teaches the local build strategy step-by-step**
- Provides examples for common software
- Requires detailed reproducibility documentation

**Key Section** (from the code):
```
If your experiment requires software that:
  - Is not pre-installed system-wide
  - Needs source code modifications
  - Requires custom compilation flags

Then DOWNLOAD and BUILD it locally:

1. Clone the source repository
2. Configure with local install prefix
3. Build and install locally
4. Modify source code as needed
5. Use the locally-built version
```

#### Component 3: Reproducibility Requirements

**Enhanced `final.json` Structure**:

```json
{
    "success": true,
    "data": {
        "raw_measurements": [...],
        "experimental_conditions": {...}
    },
    "analysis": {
        "approach": "Built PostgreSQL v16.1 from source, modified src/backend/optimizer/path/costsize.c",
        "build_artifacts": ["postgresql-16.1", "pg_duckdb-v0.1"],
        "modifications_made": [
            "Added query feature extraction in costsize.c",
            "Integrated ML model for cost estimation"
        ]
    },
    "reproducibility": {
        "source_repositories": [
            "https://github.com/postgres/postgres.git@REL_16_1",
            "https://github.com/duckdb/pg_duckdb.git@main"
        ],
        "build_commands": [
            "./configure --prefix=/workspace/local",
            "make -j8",
            "make install"
        ],
        "can_reproduce": true
    }
}
```

## Example: PostgreSQL/DuckDB Query Routing Experiment

### What the Agent Should Do Now

**Step 1**: Recognize it needs PostgreSQL and DuckDB

**Step 2**: Clone and build locally
```bash
# Clone PostgreSQL
git clone --depth 1 --branch REL_16_1 https://github.com/postgres/postgres.git
cd postgres

# Configure for local install
./configure --prefix=/workspace/local/postgres --without-readline

# Build
make -j$(nproc)
make install

# Clone DuckDB
cd /workspace
git clone https://github.com/duckdb/duckdb.git
cd duckdb
mkdir build && cd build
cmake -DCMAKE_INSTALL_PREFIX=/workspace/local/duckdb ..
make -j$(nproc)
make install
```

**Step 3**: Extract query features (modify source if needed)
```bash
# Example: Add feature extraction to PostgreSQL planner
cd /workspace/postgres/src/backend/optimizer/path

# Edit costsize.c to log query features
vim costsize.c
# Add: extract_query_features(root, query_plan)

# Rebuild
cd /workspace/postgres
make -j$(nproc)
make install
```

**Step 4**: Run dual-execution experiment
```python
import subprocess
import json

results = []

for query in test_queries:
    # Execute on local PostgreSQL
    pg_time = measure_postgres_query(query, "/workspace/local/postgres/bin/psql")

    # Execute on local DuckDB
    duckdb_time = measure_duckdb_query(query, "/workspace/local/duckdb/bin/duckdb")

    results.append({
        "query": query,
        "postgres_time": pg_time,
        "duckdb_time": duckdb_time,
        "faster_engine": "postgres" if pg_time < duckdb_time else "duckdb"
    })

# Train ML model
train_routing_model(results)
```

**Step 5**: Document everything in `final.json`

## Advantages of This Generic Approach

### 1. Domain Agnostic
Works for **any** research that needs software:
- Database systems
- Compilers
- Web servers
- ML frameworks
- Operating systems
- Programming languages

### 2. True Source Code Modification
Agent can:
- Clone exact version from git
- Edit source files
- Add instrumentation
- Apply patches
- Test modifications

### 3. Reproducible
`final.json` contains:
- Exact git commits used
- Build commands executed
- Source modifications made
- Anyone can reproduce

### 4. No Pre-installation Needed
- Single Docker image for all research
- No domain-specific maintenance
- Smaller image size

### 5. Teaches Real Research Skills
Agent learns to:
- Clone repositories
- Build software from source
- Modify and rebuild
- Document changes
- **This is how real CS research works!**

## Testing the Solution

### Step 1: Build Generic Image
```bash
cd /Users/wuy/Desktop/code/UAgent
docker build -t earthwuyang/uagent-research:v1.0 -f docker/research-runtime.Dockerfile .
```

### Step 2: Update Configuration
```bash
# Edit .env
sed -i '' 's/UAGENT_OPENHANDS_IMAGE=.*/UAGENT_OPENHANDS_IMAGE=earthwuyang\/uagent-research:v1.0/' .env
```

### Step 3: Restart Backend
```bash
pkill -f "uvicorn app.main:app"
source .venv/bin/activate
uvicorn app.main:app --reload --port 8001 --app-dir backend
```

### Step 4: Submit Test Query
```bash
curl -X POST http://localhost:8001/api/research/scientific \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Reproduce the ML-based query routing experiment: build PostgreSQL and DuckDB from source, extract query features, collect dual-execution times, and train a classifier",
    "research_id": "test_generic_build"
  }'
```

### Step 5: Monitor Execution
```bash
tail -f /tmp/uagent-workspace/uagent_workspaces/*/logs/openhands_live/live_combined.log
```

### Expected Behavior

The agent should:
1. ✅ Read the generic guidance
2. ✅ Recognize it needs PostgreSQL and DuckDB
3. ✅ Clone both from GitHub
4. ✅ Configure with `--prefix=/workspace/local`
5. ✅ Build with `make -j$(nproc)`
6. ✅ Install locally
7. ✅ Run experiments using local builds
8. ✅ Document build process in `final.json`

## What Changed in the Code

### Modified Files

1. **`docker/research-runtime.Dockerfile`**
   - Generic build tools (gcc, cmake, git)
   - Common dev libraries
   - NO domain-specific software

2. **`backend/app/core/research_engines/scientific_research.py`**
   - Lines 1178-1272: Generic guidance
   - Explains local build strategy
   - Provides step-by-step instructions
   - Examples for common software

### Deleted Files

1. ~~`backend/app/core/research_engines/environment_spec.py`~~ - Too domain-specific
2. ~~`backend/app/core/research_engines/environment_validator.py`~~ - Too domain-specific

These were **removed** because they hardcoded PostgreSQL/DuckDB knowledge, which defeats the purpose of a **generic** research system.

## Success Criteria

An experiment is successful if:
- ✅ `final.json` contains `"success": true`
- ✅ `analysis.build_artifacts` lists software built from source
- ✅ `analysis.modifications_made` documents source changes (if any)
- ✅ `reproducibility.source_repositories` lists exact git URLs and commits
- ✅ `reproducibility.build_commands` contains actual build commands used
- ✅ No simulation or placeholder data

## Comparison: Before vs After

### Before (Failing)
```
Agent: "I need PostgreSQL with kernel modifications"
Agent: "apt install postgresql" → FAILS (no root)
Agent: "Can't get real data"
Agent: "Using simulation instead" ❌
Result: Paper reports "container environment limitations"
```

### After (Working)
```
Agent: "I need PostgreSQL with kernel modifications"
Agent: Reads guidance → "Clone and build locally"
Agent: "git clone https://github.com/postgres/postgres.git"
Agent: "./configure --prefix=/workspace/local && make -j8"
Agent: Edits src/backend/optimizer/path/costsize.c
Agent: "make -j8" to rebuild
Agent: Runs experiments with local build ✅
Result: Real data, documented build process, reproducible
```

## Why This Solves Your Paper's Failures

Your paper listed these failures:

| Issue | How Generic Solution Fixes It |
|-------|-------------------------------|
| "No access to install system packages" | **Clone and build locally, no system install needed** |
| "Unable to collect real measurements" | **Local builds provide real software, real measurements** |
| "Cannot modify core database functionality" | **Edit cloned source code, rebuild as needed** |
| "Container environment limitations" | **Work within /workspace, all tools available** |

## Limitations

### What This Does NOT Solve

1. **Closed-source software**: Can't clone or modify (e.g., Oracle, Microsoft SQL Server)
   - **Workaround**: Use open-source alternatives or connect to external instances

2. **Hardware requirements**: Container can't add GPUs, special hardware
   - **Workaround**: Document hardware limitations, use what's available

3. **Very long builds**: Some software takes hours to compile (e.g., LLVM, Chromium)
   - **Workaround**: Use incremental builds, cache binaries, or adjust timeouts

4. **Build failures**: Not all software builds cleanly with generic tools
   - **Workaround**: Document build issues, try alternative versions, file bug reports

### When Estimation is Acceptable

If building from source is truly impossible, document it:

```json
{
    "analysis": {
        "approach": "PostgreSQL build failed after 3 attempts (missing dependency XYZ)",
        "fallback": "Used DuckDB for real measurements, estimated PostgreSQL times based on published benchmarks",
        "estimation_methodology": "Used TPC-H results from postgres.org, scaled by query complexity",
        "limitations": ["PostgreSQL times are estimates, not real measurements"]
    }
}
```

## Conclusion

The **generic local-build strategy** transforms UAgent from a system that only works with pre-installed software to one that can handle **any open-source research project**.

**Key Principle**:
> Don't tell the agent "don't simulate" without explaining HOW to get real data.
> Instead, teach it: "If you can't install X, clone X's source and build it locally."

This is:
- ✅ **Generic**: Works for any domain
- ✅ **Powerful**: Enables source code modifications
- ✅ **Reproducible**: Documents exact build process
- ✅ **Scalable**: One Docker image for all research
- ✅ **Educational**: Teaches real research methodology

### Implementation Checklist

- [x] Create generic build tools Dockerfile
- [x] Add generic guidance to experiment goals
- [x] Remove domain-specific code
- [ ] Build Docker image locally
- [ ] Update .env configuration
- [ ] Restart backend
- [ ] Test with PostgreSQL/DuckDB experiment
- [ ] Verify `final.json` contains build artifacts
- [ ] Test with different domain (e.g., compiler research)

## Next Steps

1. **Build the image**: `docker build -t earthwuyang/uagent-research:v1.0 -f docker/research-runtime.Dockerfile .`
2. **Update .env**: Change `UAGENT_OPENHANDS_IMAGE` to new image
3. **Test**: Submit a research query that requires building from source
4. **Validate**: Check that `final.json` documents the build process
5. **Iterate**: If experiments still fail, examine logs and refine guidance

The solution is now **domain-agnostic** and **research-methodology-correct**.