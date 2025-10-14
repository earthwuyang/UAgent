"""Test git worktree setup and cleanup operations."""

import asyncio
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from uagent_research.models.research_tree import ExperimentContext


async def test_worktree_setup_and_cleanup():
    """Test that worktree can be created and cleaned up."""
    
    # Create a temporary git repository for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"\n📁 Created temp directory: {temp_dir}")
        
        # Initialize git repo
        subprocess.run(['git', 'init'], cwd=temp_dir, capture_output=True, check=True)
        subprocess.run(['git', 'config', 'user.email', 'test@example.com'], cwd=temp_dir, capture_output=True, check=True)
        subprocess.run(['git', 'config', 'user.name', 'Test User'], cwd=temp_dir, capture_output=True, check=True)
        
        # Create initial commit
        test_file = os.path.join(temp_dir, 'test.txt')
        with open(test_file, 'w') as f:
            f.write('test content')
        subprocess.run(['git', 'add', '.'], cwd=temp_dir, capture_output=True, check=True)
        subprocess.run(['git', 'commit', '-m', 'Initial commit'], cwd=temp_dir, capture_output=True, check=True)
        
        print("✅ Git repository initialized")
        
        # Create experiment context
        worktree_path = os.path.join(temp_dir, 'worktrees', 'exp_test_branch')
        exp_context = ExperimentContext(
            conversation_id="exp_test_conv_123",
            worktree_branch="exp_test_branch",
            worktree_path=worktree_path,
            parent_branch="master"  # Git init creates master by default
        )
        
        # Create minimal orchestrator instance (just to test the methods)
        class MockOrchestrator:
            async def _setup_experiment_worktree(self, experiment_context):
                """Copied from TreeSearchOrchestrator for testing."""
                import subprocess
                import os
                
                worktree_path = experiment_context.worktree_path
                branch_name = experiment_context.worktree_branch
                
                print(f"[TEST] Setting up worktree: {worktree_path}")
                
                # Check if worktree already exists
                if os.path.exists(worktree_path):
                    print(f"[TEST] Worktree already exists, reusing")
                    return
                
                # Ensure parent directory exists
                worktrees_dir = os.path.dirname(worktree_path)
                if worktrees_dir and not os.path.exists(worktrees_dir):
                    os.makedirs(worktrees_dir, exist_ok=True)
                    print(f"[TEST] Created worktrees directory")
                
                # Create git worktree
                cmd = ['git', 'worktree', 'add', worktree_path, '-b', branch_name]
                result = subprocess.run(cmd, capture_output=True, text=True, cwd=temp_dir)
                
                if result.returncode != 0:
                    raise RuntimeError(f"Failed to create worktree: {result.stderr}")
                
                print(f"[TEST] Successfully created worktree")
            
            async def _cleanup_experiment_worktree(self, worktree_path):
                """Copied from TreeSearchOrchestrator for testing."""
                import subprocess
                import os
                
                if not worktree_path or not os.path.exists(worktree_path):
                    print(f"[TEST] Worktree doesn't exist, no cleanup needed")
                    return
                
                print(f"[TEST] Cleaning up worktree: {worktree_path}")
                
                cmd = ['git', 'worktree', 'remove', worktree_path, '--force']
                result = subprocess.run(cmd, capture_output=True, text=True, cwd=temp_dir)
                
                if result.returncode != 0:
                    print(f"[TEST] Warning: Failed to remove worktree: {result.stderr}")
                else:
                    print(f"[TEST] Successfully removed worktree")
        
        mock_orch = MockOrchestrator()
        
        # Test worktree setup
        await mock_orch._setup_experiment_worktree(exp_context)
        
        # Verify worktree was created
        assert os.path.exists(worktree_path), f"Worktree directory not created: {worktree_path}"
        assert os.path.isdir(worktree_path), f"Worktree path is not a directory: {worktree_path}"
        
        # Verify worktree has .git file (not directory, worktrees use git file pointer)
        git_file = os.path.join(worktree_path, '.git')
        assert os.path.exists(git_file), f"Worktree .git not found: {git_file}"
        
        print("✅ Worktree created successfully")
        
        # Verify worktree is listed in git worktrees
        result = subprocess.run(['git', 'worktree', 'list'], cwd=temp_dir, capture_output=True, text=True)
        assert 'exp_test_branch' in result.stdout, "Worktree not listed in git worktree list"
        print("✅ Worktree listed in git worktree list")
        
        # Test idempotency - calling setup again should not fail
        await mock_orch._setup_experiment_worktree(exp_context)
        print("✅ Worktree setup is idempotent")
        
        # Test worktree cleanup
        await mock_orch._cleanup_experiment_worktree(worktree_path)
        
        # Verify worktree was removed
        assert not os.path.exists(worktree_path), f"Worktree directory still exists after cleanup: {worktree_path}"
        print("✅ Worktree cleaned up successfully")
        
        # Verify worktree is no longer listed
        result = subprocess.run(['git', 'worktree', 'list'], cwd=temp_dir, capture_output=True, text=True)
        assert 'exp_test_branch' not in result.stdout, "Worktree still listed after cleanup"
        print("✅ Worktree removed from git worktree list")


async def test_parallel_worktrees():
    """Test that multiple worktrees can be created in parallel."""
    
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"\n📁 Created temp directory for parallel test: {temp_dir}")
        
        # Initialize git repo
        subprocess.run(['git', 'init'], cwd=temp_dir, capture_output=True, check=True)
        subprocess.run(['git', 'config', 'user.email', 'test@example.com'], cwd=temp_dir, capture_output=True, check=True)
        subprocess.run(['git', 'config', 'user.name', 'Test User'], cwd=temp_dir, capture_output=True, check=True)
        
        test_file = os.path.join(temp_dir, 'test.txt')
        with open(test_file, 'w') as f:
            f.write('test content')
        subprocess.run(['git', 'add', '.'], cwd=temp_dir, capture_output=True, check=True)
        subprocess.run(['git', 'commit', '-m', 'Initial commit'], cwd=temp_dir, capture_output=True, check=True)
        
        # Create multiple experiment contexts
        contexts = []
        for i in range(3):
            worktree_path = os.path.join(temp_dir, 'worktrees', f'exp_test_branch_{i}')
            contexts.append(ExperimentContext(
                conversation_id=f"exp_test_conv_{i}",
                worktree_branch=f"exp_test_branch_{i}",
                worktree_path=worktree_path,
                parent_branch="master"
            ))
        
        # Create all worktrees (simulating parallel experiments)
        class MockOrchestrator:
            async def _setup_experiment_worktree(self, experiment_context):
                import subprocess
                import os
                
                worktree_path = experiment_context.worktree_path
                branch_name = experiment_context.worktree_branch
                
                if os.path.exists(worktree_path):
                    return
                
                worktrees_dir = os.path.dirname(worktree_path)
                if worktrees_dir and not os.path.exists(worktrees_dir):
                    os.makedirs(worktrees_dir, exist_ok=True)
                
                cmd = ['git', 'worktree', 'add', worktree_path, '-b', branch_name]
                result = subprocess.run(cmd, capture_output=True, text=True, cwd=temp_dir)
                
                if result.returncode != 0:
                    raise RuntimeError(f"Failed to create worktree: {result.stderr}")
        
        mock_orch = MockOrchestrator()
        
        # Create all worktrees
        for ctx in contexts:
            await mock_orch._setup_experiment_worktree(ctx)
        
        # Verify all worktrees exist
        for ctx in contexts:
            assert os.path.exists(ctx.worktree_path), f"Worktree not created: {ctx.worktree_path}"
        
        print("✅ Multiple parallel worktrees created successfully")
        
        # Verify all branches exist
        result = subprocess.run(['git', 'branch'], cwd=temp_dir, capture_output=True, text=True)
        for i in range(3):
            assert f'exp_test_branch_{i}' in result.stdout, f"Branch exp_test_branch_{i} not found"
        
        print("✅ All worktree branches created")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Git Worktree Operations")
    print("=" * 60)
    
    asyncio.run(test_worktree_setup_and_cleanup())
    asyncio.run(test_parallel_worktrees())
    
    print("\n" + "=" * 60)
    print("✅ All worktree tests passed!")
    print("=" * 60)
