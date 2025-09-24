FROM docker.all-hands.dev/all-hands-ai/runtime:latest

# Pre-install database drivers and common data libs used by the experiments
RUN pip install --no-cache-dir \
    "duckdb>=0.10.2" \
    "psycopg[binary]>=3.2" \
    "pandas>=2.1" \
    "numpy>=1.26"
