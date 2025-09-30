# Building Multi-Platform Docker Image

## Quick Build (Local Testing Only)

The current build creates a **local-only** multi-platform image:

```bash
cd /Users/wuy/Desktop/code/UAgent

docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -t earthwuyang/uagent-research:v1.0 \
  -f docker/research-runtime.Dockerfile \
  .
```

This creates images for:
- **linux/amd64** (x86_64) - Intel/AMD CPUs
- **linux/arm64** (aarch64) - Apple Silicon, AWS Graviton

## Push to Docker Hub

To make the multi-platform image available publicly:

### Step 1: Login to Docker Hub

```bash
docker login
# Enter username: earthwuyang
# Enter password: [your Docker Hub password]
```

### Step 2: Build and Push

```bash
cd /Users/wuy/Desktop/code/UAgent

docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -t earthwuyang/uagent-research:v1.0 \
  -t earthwuyang/uagent-research:latest \
  -f docker/research-runtime.Dockerfile \
  --push \
  .
```

The `--push` flag automatically pushes to Docker Hub after building.

## Verify Multi-Platform Support

After pushing, verify both architectures are available:

```bash
docker manifest inspect earthwuyang/uagent-research:v1.0
```

Should show:
```json
{
  "manifests": [
    {
      "platform": {
        "architecture": "amd64",
        "os": "linux"
      }
    },
    {
      "platform": {
        "architecture": "arm64",
        "os": "linux"
      }
    }
  ]
}
```

## Pull and Test on Different Platforms

### On x86_64 Linux Server

```bash
docker pull earthwuyang/uagent-research:v1.0
# Automatically pulls linux/amd64 version
docker run --rm earthwuyang/uagent-research:v1.0 uname -m
# Output: x86_64
```

### On Apple Silicon Mac

```bash
docker pull earthwuyang/uagent-research:v1.0
# Automatically pulls linux/arm64 version
docker run --rm earthwuyang/uagent-research:v1.0 uname -m
# Output: aarch64
```

### On Intel Mac (Rosetta)

```bash
docker pull earthwuyang/uagent-research:v1.0
# Pulls linux/amd64 version, runs via Rosetta emulation
docker run --rm earthwuyang/uagent-research:v1.0 uname -m
# Output: x86_64 (emulated)
```

## Build Time Estimates

- **ARM64 only**: ~3-5 minutes (your current Mac)
- **AMD64 only**: ~5-7 minutes (cross-compilation)
- **Both platforms**: ~12-15 minutes (builds both in parallel)

## Troubleshooting

### Error: "multiple platforms feature is currently not supported"

Enable Docker BuildKit:
```bash
export DOCKER_BUILDKIT=1
```

### Error: "no builder instance found"

Create a new builder:
```bash
docker buildx create --name multiplatform --use
docker buildx inspect --bootstrap
```

### Build is Very Slow

Docker Desktop on Mac uses QEMU emulation for cross-platform builds.
This is normal. The amd64 build on ARM Mac takes 2-3x longer.

To speed up:
- Build on native hardware (x86_64 Linux server for amd64)
- Use GitHub Actions with matrix builds
- Use Docker Hub Automated Builds

## Alternative: GitHub Actions

Create `.github/workflows/docker-build.yml`:

```yaml
name: Docker Multi-Platform Build

on:
  push:
    branches: [ main ]
    paths:
      - 'docker/**'

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Set up QEMU
        uses: docker/setup-qemu-action@v2

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v2

      - name: Login to Docker Hub
        uses: docker/login-action@v2
        with:
          username: ${{ secrets.DOCKERHUB_USERNAME }}
          password: ${{ secrets.DOCKERHUB_TOKEN }}

      - name: Build and push
        uses: docker/build-push-action@v4
        with:
          context: .
          file: docker/research-runtime.Dockerfile
          platforms: linux/amd64,linux/arm64
          push: true
          tags: |
            earthwuyang/uagent-research:v1.0
            earthwuyang/uagent-research:latest
```

This builds on GitHub's infrastructure (much faster).

## Size Comparison

- **Single platform** (arm64 only): ~11.6 GB
- **Multi-platform manifest**: ~23 GB total (11.6 GB × 2)
- **Downloaded per user**: 11.6 GB (only their platform)

The multi-platform manifest is a pointer to both images.
Users only download the version for their architecture.

## Current Status

✅ Built: linux/arm64 (your Mac)
⏳ Building: linux/amd64 (cross-compiled)
❌ Not pushed: Needs `--push` flag or manual `docker push`

To push manually after local build:
```bash
docker push earthwuyang/uagent-research:v1.0
```

## See Also

- Docker Buildx docs: https://docs.docker.com/buildx/
- Multi-platform images: https://docs.docker.com/build/building/multi-platform/