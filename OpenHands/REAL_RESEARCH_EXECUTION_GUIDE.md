# Real Research Execution: ML-Based Query Routing

This guide explains how to execute **real research** (not mocks) using the UAgent parallel research system.

## Overview

The script `execute_ml_routing_research.py` triggers actual parallel research to implement and evaluate an ML-based query routing system for PostgreSQL + DuckDB.

## Research Task

**Goal**: Develop an intelligent query router that decides whether to execute queries on PostgreSQL or DuckDB based on predicted performance using machine learning.

**Key Requirements**:
1. Download and build PostgreSQL + pg_duckdb from source
2. Extract pre-optimization features from PostgreSQL kernel
3. Collect dual-execution performance data
4. Train LightGBM model for engine selection
5. Embed ML model in C code
6. Implement threshold-based baselines
7. Run comprehensive experiments
8. Document everything for reproducibility

## Prerequisites

### 1. Start OpenHands Server

The research requires the OpenHands server with research extension:

```bash
cd /home/wuy/AI/UAgent/OpenHands
./start_openhands_research.sh
```

Wait for server to start (check `http://localhost:3000/api/research/health`)

### 2. Configure Proxy (Already Set)

The script is pre-configured to use proxy on `localhost:7890` for downloading source code.

## Execution

### Quick Start

```bash
cd /home/wuy/AI/UAgent/OpenHands
python3 execute_ml_routing_research.py
```

### What Happens

1. **Initialization**: 
   - Creates workspace directory
   - Initializes research middleware
   - Generates unique session ID

2. **Research Launch**:
   - Starts parallel tree orchestrator
   - Explores multiple approaches simultaneously
   - Uses PUCT algorithm for intelligent exploration

3. **Parallel Execution**:
   - Up to 3 concurrent research threads
   - Each explores different aspects (download, feature extraction, ML training, etc.)
   - Automatically handles dependencies

4. **Output Generation**:
   - Source code in `workspace/ml_routing_research/`
   - README.md with reproducible commands
   - requirements.txt with Python dependencies
   - Experiment results and logs

## Monitoring Progress

### Via API

```bash
# Get experiment status
curl http://localhost:3000/api/research/experiments/<experiment_id>/status

# Get research tree
curl http://localhost:3000/api/research/experiments/<experiment_id>/tree
```

### Via Web UI

Navigate to:
```
http://120.46.207.248:3000/conversations/<session_id>
```

Click on the "Research" tab to see the tree visualization.

### Via Logs

```bash
# Check workspace output
ls -la workspace/ml_routing_research/

# View README
cat workspace/ml_routing_research/README.md

# Check requirements
cat workspace/ml_routing_research/requirements.txt
```

## Research Configuration

The script uses these parameters:

```python
config = {
    'experiment_name': 'ml_query_routing_postgres_duckdb',
    'max_iterations': 50,      # Complex task needs more iterations
    'max_cost': 20.0,          # Higher budget for real research
    'max_parallel': 3,         # Parallel exploration
    'max_tokens': 200000,      # Large token budget
    'workspace_dir': '/home/wuy/AI/UAgent/OpenHands/workspace/ml_routing_research',
    'proxy': 'http://localhost:7890',
}
```

You can modify these in the script if needed.

## Expected Outputs

### Directory Structure

```
workspace/ml_routing_research/
├── README.md                    # Reproducibility guide
├── requirements.txt            # Python dependencies
├── postgres_src/               # PostgreSQL source
├── pg_duckdb_src/             # pg_duckdb extension source
├── feature_extraction/         # Feature extraction code
│   ├── extract_features.c
│   └── feature_logs/
├── data_collection/            # Dual-execution data
│   ├── collect_data.py
│   └── query_features.csv
├── ml_model/                   # ML training
│   ├── train_lightgbm.py
│   ├── model.txt
│   └── model_c_embedding.c
├── routing_implementation/     # Router code
│   ├── intelligent_router.c
│   └── threshold_router.c
├── experiments/                # Experiment scripts
│   ├── run_experiments.py
│   └── compare_methods.py
└── results/                    # Experiment results
    ├── postgres_only.csv
    ├── duckdb_only.csv
    ├── threshold_10000.csv
    ├── threshold_50000.csv
    └── lightgbm_routing.csv
```

### README.md Content

The generated README.md will contain:
- Build commands for PostgreSQL and pg_duckdb
- Feature extraction procedures
- Data collection scripts
- ML model training steps
- C code compilation
- Experiment execution commands
- Results analysis

### requirements.txt Content

```
lightgbm>=3.3.0
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=0.24.0
psycopg2-binary>=2.9.0
matplotlib>=3.4.0
seaborn>=0.11.0
```

## Research Workflow

The parallel research system will:

1. **Phase 1: Source Code Acquisition** (Parallel)
   - Download PostgreSQL source
   - Download pg_duckdb extension
   - Build both projects

2. **Phase 2: Feature Extraction** (Parallel)
   - Modify PostgreSQL kernel to extract features
   - Implement logging to files
   - Test feature extraction

3. **Phase 3: Data Collection** (Parallel)
   - Run queries on both engines
   - Collect execution times
   - Log features and performance

4. **Phase 4: ML Model Training**
   - Train LightGBM model
   - Validate model accuracy
   - Export model for C embedding

5. **Phase 5: Implementation** (Parallel)
   - Embed ML model in C
   - Implement threshold baselines
   - Integrate with PostgreSQL

6. **Phase 6: Experiments** (Parallel)
   - PostgreSQL-only baseline
   - DuckDB-only baseline
   - Threshold-based routing (multiple thresholds)
   - LightGBM-based routing
   - Compare all methods

7. **Phase 7: Documentation**
   - Generate README.md
   - Create requirements.txt
   - Document all commands

## Troubleshooting

### Server Not Running

```bash
# Check server status
curl http://localhost:3000/api/research/health

# If not running, start it
./start_openhands_research.sh
```

### Proxy Issues

```bash
# Test proxy
curl -x http://localhost:7890 https://github.com

# If proxy not working, update in script or remove
```

### Memory/Resource Issues

Reduce parallelism in config:

```python
'max_parallel': 2,  # or even 1
'max_iterations': 30,
```

### Check Progress

```bash
# View experiment status
python3 << EOF
import asyncio
import requests

response = requests.get('http://localhost:3000/api/research/experiments/<experiment_id>/status')
print(response.json())
