#!/usr/bin/env python3
"""CLI script to manually cleanup Docker containers"""

import sys
import os
import argparse

# Add the backend directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.core.docker_container_manager import get_container_manager

def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(description='Clean up UAgent Docker containers')
    parser.add_argument('--all', action='store_true', help='Clean up all managed containers')
    parser.add_argument('--openhands', action='store_true', help='Clean up all OpenHands containers')
    parser.add_argument('--container-id', type=str, help='Clean up specific container by ID')

    args = parser.parse_args()

    if not any([args.all, args.openhands, args.container_id]):
        parser.print_help()
        return 1

    try:
        container_manager = get_container_manager()

        if args.container_id:
            print(f"🧹 Cleaning up container: {args.container_id}")
            success = container_manager.cleanup_container(args.container_id)
            if success:
                print("✅ Container cleanup successful")
            else:
                print("❌ Container cleanup failed")
                return 1

        if args.openhands:
            print("🧹 Cleaning up all OpenHands containers...")
            container_manager.cleanup_openhands_containers()
            print("✅ OpenHands container cleanup completed")

        if args.all:
            print("🧹 Cleaning up all managed containers...")
            container_manager.cleanup_all_containers()
            print("✅ All container cleanup completed")

        return 0

    except Exception as e:
        print(f"❌ Error during cleanup: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())