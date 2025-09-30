# Generic Research Runtime Docker Image

This Docker image provides **generic build tools** for compiling ANY open-source software from source. It does NOT pre-install domain-specific tools like PostgreSQL, DuckDB, Redis, etc.

## Philosophy

**Wrong Approach** ❌: Pre-install PostgreSQL, DuckDB, Redis, ML frameworks, compilers...
- Problem: Hardcodes research domains
- Not scalable: Need different images for different research
- Brittle: Version conflicts, large image size

**Right Approach** ✅: Provide generic build tools + teach local compilation
- Works for ANY open-source software
- Agent clones source and builds locally
- Can modify source code as needed
- Single image for all research domains

## What's Included

### Build Tools
- **gcc, g++** - C/C++ compilers
- **cmake** - Modern build system
- **autoconf, automake, libtool** - Classic build tools
- **make** - Build automation
- **pkg-config** - Library configuration

### Version Control
- **git** - Primary VCS
- **subversion** - Legacy projects
- **mercurial** - Some projects still use it

### Download & Compression
- **wget, curl** - Download files
- **zip, unzip, bzip2, xz-utils** - Handle various archives

### Common Development Libraries
- **libssl-dev** - OpenSSL (HTTPS, crypto)
- **zlib1g-dev** - Compression
- **libreadline-dev** - Interactive CLI
- **libffi-dev** - Foreign function interface
- **libsqlite3-dev** - SQLite database
- **libpq-dev** - PostgreSQL client library
- **libmysqlclient-dev** - MySQL client library

### Python Scientific Stack
- **numpy** - Numerical computing
- **scipy** - Scientific computing
- **pandas** - Data analysis
- **matplotlib** - Plotting
- **scikit-learn** - Machine learning
- **requests** - HTTP library

### Debugging Tools
- **gdb** - GNU debugger
- **valgrind** - Memory profiler
- **strace** - System call tracer

## What's NOT Included

This image deliberately does NOT include:
- ❌ PostgreSQL server
- ❌ DuckDB
- ❌ Redis
- ❌ NGINX
- ❌ Domain-specific databases
- ❌ Precompiled frameworks

**Why?** Because agents should **build these from source** when needed.

## Quick Start

### 1. Build the Image

```bash
cd /Users/wuy/Desktop/code/UAgent
docker build -t earthwuyang/uagent-research:v1.0 -f docker/research-runtime.Dockerfile .
```

Build time: ~10-15 minutes (depends on internet speed)

### 2. Update Configuration

Edit `.env`:
```bash
# Change this line:
UAGENT_OPENHANDS_IMAGE=earthwuyang/uagent:v0.1

# To this:
UAGENT_OPENHANDS_IMAGE=earthwuyang/uagent-research:v1.0
```

Or use sed:
```bash
sed -i '' 's/UAGENT_OPENHANDS_IMAGE=.*/UAGENT_OPENHANDS_IMAGE=earthwuyang\/uagent-research:v1.0/' .env
```

### 3. Restart Backend

```bash
pkill -f "uvicorn app.main:app"
cd /Users/wuy/Desktop/code/UAgent
source .venv/bin/activate
uvicorn app.main:app --reload --port 8001 --app-dir backend
```

## How Agents Use This Image

When an experiment needs software (e.g., PostgreSQL):

### Step 1: Agent reads guidance
```
"If your experiment requires software that is not pre-installed,
 DOWNLOAD and BUILD it locally."
```

### Step 2: Agent clones source
```bash
git clone https://github.com/postgres/postgres.git
cd postgres
```

### Step 3: Agent configures for local install
```bash
./configure --prefix=/workspace/local/postgres
```

### Step 4: Agent builds
```bash
make -j$(nproc)
make install
```

### Step 5: Agent modifies source (if needed)
```bash
vim src/backend/optimizer/path/costsize.c
# Add custom feature extraction
make -j$(nproc)  # Rebuild
```

### Step 6: Agent uses local build
```bash
export PATH=/workspace/local/postgres/bin:$PATH
psql --version  # Uses locally-built PostgreSQL
```

### Step 7: Agent documents in final.json
```json
{
    "reproducibility": {
        "source_repositories": ["https://github.com/postgres/postgres.git@REL_16_1"],
        "build_commands": [
            "./configure --prefix=/workspace/local/postgres",
            "make -j8",
            "make install"
        ],
        "modifications_made": ["Added query feature extraction to costsize.c"]
    }
}
```

## Testing the Image

### Manual Test

```bash
# Start container
docker run -it --rm earthwuyang/uagent-research:v1.0 bash

# Inside container, test build tools
gcc --version
cmake --version
git --version
python3 -c "import numpy, pandas, sklearn; print('✓ Python packages OK')"

# Test building something from source
cd /workspace
git clone https://github.com/antirez/redis.git
cd redis
make -j$(nproc)
./src/redis-server --version

exit
```

### Automated Test

```bash
# Submit test research query
curl -X POST http://localhost:8001/api/research/scientific \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Build Redis from source and benchmark SET/GET operations",
    "research_id": "test_generic_build_redis"
  }'

# Monitor execution
tail -f /tmp/uagent-workspace/uagent_workspaces/*/logs/openhands_live/live_combined.log
```

Expected behavior: Agent clones Redis, builds it, runs benchmarks, documents everything.

## Examples of Supported Research

### Database Research
```bash
git clone https://github.com/postgres/postgres.git
git clone https://github.com/duckdb/duckdb.git
git clone https://github.com/clickhouse/clickhouse.git
```

### Systems Research
```bash
git clone https://github.com/redis/redis.git
git clone https://github.com/nginx/nginx.git
git clone https://github.com/memcached/memcached.git
```

### Compiler Research
```bash
git clone https://github.com/llvm/llvm-project.git
git clone https://github.com/gcc-mirror/gcc.git
```

### Language Research
```bash
git clone https://github.com/python/cpython.git
git clone https://github.com/ruby/ruby.git
git clone https://github.com/nodejs/node.git
```

### ML Framework Research
```bash
git clone https://github.com/pytorch/pytorch.git
git clone https://github.com/tensorflow/tensorflow.git
```

**Key Point**: The image doesn't include any of these, but has all the tools to BUILD them.

## Troubleshooting

### Build Fails with "Cannot find X"

The image includes common dev libraries, but not all. If a build fails:

1. **Check error message** for missing dependency
2. **Document limitation** in final.json
3. **Try alternative approach** (different version, configure flags)

Example:
```json
{
    "analysis": {
        "approach": "Attempted PostgreSQL build, failed due to missing libicu-dev",
        "fallback": "Used DuckDB instead (builds without additional deps)",
        "limitations": ["Could not test PostgreSQL, used alternative"]
    }
}
```

### Image Too Large

Current size: ~1.5 GB (compressed)

To reduce:
- Remove debugging tools (gdb, valgrind, strace)
- Remove less common libraries
- Use multi-stage builds

But remember: **Generic flexibility is worth the size**.

### Very Long Builds

Some software takes hours (LLVM, Chromium, TensorFlow).

Solutions:
- Use `--depth 1` for shallow clones
- Increase `OPENHANDS_MAX_MINUTES` in .env
- Cache builds between experiments
- Use precompiled versions when available

## Comparison to Previous Approach

### Old Approach (Domain-Specific)
```dockerfile
# Pre-install PostgreSQL
RUN apt-get install postgresql-15

# Pre-install DuckDB
RUN wget duckdb && install

# Pre-install Redis
RUN apt-get install redis

# ... (dozens more packages)
```

**Problems**:
- Only works for database research
- Can't modify source code
- Huge image size
- Version conflicts
- Not extensible

### New Approach (Generic)
```dockerfile
# Install build tools
RUN apt-get install build-essential cmake git

# Install dev libraries
RUN apt-get install libssl-dev zlib1g-dev ...

# That's it!
```

**Benefits**:
- Works for ANY research domain
- Can modify source code
- Smaller image size
- No version conflicts
- Infinitely extensible

## Advanced Usage

### Caching Builds

To speed up repeated experiments:

```bash
# First experiment: build PostgreSQL
git clone https://github.com/postgres/postgres.git
cd postgres && ./configure --prefix=/workspace/cache/postgres
make -j$(nproc) && make install

# Later experiments: reuse build
export PATH=/workspace/cache/postgres/bin:$PATH
# No rebuild needed!
```

### Cross-Compilation

The image includes gcc/g++ for the host architecture. For cross-compilation:

```bash
apt-get install gcc-aarch64-linux-gnu  # Won't work (no apt)
# Instead, build a cross-compiler from source!
git clone https://github.com/crosstool-ng/crosstool-ng.git
```

## Push to Docker Hub (Optional)

```bash
docker login
docker tag earthwuyang/uagent-research:v1.0 earthwuyang/uagent-research:latest
docker push earthwuyang/uagent-research:v1.0
docker push earthwuyang/uagent-research:latest
```

## See Also

- **Full Analysis**: `../EXPERIMENTAL_FAILURES_ANALYSIS.md`
- **Experiment Guidance**: `../backend/app/core/research_engines/scientific_research.py` (lines 1178-1272)
- **Base Image**: https://github.com/All-Hands-AI/OpenHands

## License

Same as UAgent project (MIT)