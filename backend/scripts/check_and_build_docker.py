#!/usr/bin/env python3
"""
Check if required Docker image exists locally or remotely.
If not found, build it automatically.
"""

import subprocess
import sys
import os
from pathlib import Path


def run_command(cmd: list[str], check: bool = True) -> tuple[int, str, str]:
    """Run a command and return (returncode, stdout, stderr)"""
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False
    )
    if check and result.returncode != 0:
        print(f"Error running command: {' '.join(cmd)}", file=sys.stderr)
        print(f"Stderr: {result.stderr}", file=sys.stderr)
    return result.returncode, result.stdout, result.stderr


def check_image_exists_locally(image_name: str) -> bool:
    """Check if Docker image exists locally"""
    returncode, stdout, _ = run_command(
        ["docker", "images", "-q", image_name],
        check=False
    )
    return returncode == 0 and bool(stdout.strip())


def check_image_exists_remotely(image_name: str) -> bool:
    """Check if Docker image exists on Docker Hub"""
    returncode, stdout, _ = run_command(
        ["docker", "manifest", "inspect", image_name],
        check=False
    )
    return returncode == 0


def build_image(project_root: Path, image_name: str, dockerfile: str) -> bool:
    """Build Docker image locally"""
    print(f"\n{'='*70}")
    print(f"Building Docker image: {image_name}")
    print(f"Dockerfile: {dockerfile}")
    print(f"This may take 10-15 minutes...")
    print(f"{'='*70}\n")

    # Build the image
    cmd = [
        "docker", "build",
        "-t", image_name,
        "-f", dockerfile,
        str(project_root)
    ]

    print(f"Running: {' '.join(cmd)}\n")

    # Run build with real-time output
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=str(project_root)
    )

    # Stream output
    for line in iter(process.stdout.readline, ''):
        if line:
            print(line, end='')

    process.wait()

    if process.returncode == 0:
        print(f"\n✅ Successfully built: {image_name}")
        return True
    else:
        print(f"\n❌ Failed to build: {image_name}", file=sys.stderr)
        return False


def get_image_from_env() -> str:
    """Get Docker image name from environment variable"""
    image = os.getenv("UAGENT_OPENHANDS_IMAGE")
    if not image:
        raise RuntimeError(
            "UAGENT_OPENHANDS_IMAGE environment variable is not set. "
            "Please set it in .env file."
        )
    return image


def main():
    """Main entry point"""
    # Get project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent

    # Get image name from environment
    try:
        image_name = get_image_from_env()
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Checking Docker image: {image_name}")

    # Check if image exists locally
    if check_image_exists_locally(image_name):
        print(f"✅ Image found locally: {image_name}")
        return 0

    print(f"⚠️  Image not found locally: {image_name}")

    # Check if image exists remotely
    print(f"Checking Docker Hub for: {image_name}")
    if check_image_exists_remotely(image_name):
        print(f"✅ Image found on Docker Hub: {image_name}")
        print(f"Pulling image...")
        returncode, _, _ = run_command(["docker", "pull", image_name])
        if returncode == 0:
            print(f"✅ Successfully pulled: {image_name}")
            return 0
        else:
            print(f"⚠️  Failed to pull image. Will build locally.", file=sys.stderr)

    # Image not found locally or remotely - build it
    print(f"\n⚠️  Image not found on Docker Hub: {image_name}")
    print(f"Will build image locally from source...")

    dockerfile = project_root / "docker" / "research-runtime.Dockerfile"
    if not dockerfile.exists():
        print(f"Error: Dockerfile not found at {dockerfile}", file=sys.stderr)
        sys.exit(1)

    # Build the image
    success = build_image(project_root, image_name, str(dockerfile))

    if success:
        print(f"\n✅ Image ready: {image_name}")
        return 0
    else:
        print(f"\n❌ Failed to build image", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())