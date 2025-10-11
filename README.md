# UAgent Research Session Notes

## Session Commands
- `tmux new-session -d -s uagent-backend`
- `tmux send-keys -t uagent-backend "cd /Users/wuy/Desktop/code/UAgent" C-m`
- `tmux send-keys -t uagent-backend "source .venv/bin/activate" C-m`
- `tmux send-keys -t uagent-backend "./start_openhands_research.sh" C-m`

## Research Prompt
The following prompt was submitted via the OpenHands UI to initiate the research workflow:

```
research goal: modify postgres and pg_duckdb source code （ to download source code you can utilize the proxy on port localhost:7890, do not use the system-wide postgresql）, first extract pre-opt features from postgres kernel and log to files, then collect dual-execution data (pre-optimization query features that can be found in kernel structures and execution times on dual engine) and train a machine learning model to predict whether postgres engine or duckdb engine executes a query fast and embed the machine learning model into database source code (using the language of the database for example c language) to online route each query to the faster engine, and execute end-to-end experiments to test the ml-based system’s performance. A baseline method called threshold-based method should also be implemented, which routes query based on threshold, for example threshold can be 10000 or 50000 or any other value, if postgres estimates the cost of a query is above threshold, then send to duckdb, otherwise send to postgres, and compare the postgres-only, duckdb-only, different threshold-based methods and lightgbm-based method. please record every successful  necessary commands in README.md so that later people can reproduce your results. also record your python packages dependencies in requirements.txt.
```

## Current Observations
- Backend reports successful research activation (experiment `exp_1d1ad58d9b7948d4b2231ef46472d6ad_1760163024_981fdc`).
- Frontend encounters repeated WebSocket error 1006 leading to React error #185; research panel fails to render.
