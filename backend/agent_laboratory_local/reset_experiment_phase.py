#!/usr/bin/env python3
"""
Reset the experiment phase so Agent Laboratory regenerates the experiment code
with the improved anti-dummy prompts
"""
import pickle
import sys
from pathlib import Path

# Import the necessary classes for unpickling
sys.path.append('.')
from ai_lab_repo import LaboratoryWorkflow

def reset_experiment_phase(state_file_path: str):
    """Reset the experiment phase in the state file"""
    print(f"🔧 Resetting experiment phase in {state_file_path}")

    try:
        # Load the state
        with open(state_file_path, 'rb') as f:
            state = pickle.load(f)

        print("✅ Loaded state file successfully")

        # Check what's in the state
        if hasattr(state, 'phase_status'):
            print(f"📊 Current phase status: {state.phase_status}")

            # Reset the experiment-related phases
            phases_to_reset = [
                "data preparation",
                "running experiments",
                "results interpretation",
                "report writing",
                "report refinement"
            ]

            for phase in phases_to_reset:
                if phase in state.phase_status:
                    state.phase_status[phase] = False
                    print(f"🔄 Reset phase: {phase}")

            # Save the modified state
            with open(state_file_path, 'wb') as f:
                pickle.dump(state, f)

            print("✅ State file updated successfully")
            print("🎯 Agent Laboratory will now regenerate experiment code with improved prompts")

        else:
            print("❌ No phase_status found in state file")
            return False

    except Exception as e:
        print(f"❌ Error resetting state: {e}")
        return False

    return True

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python reset_experiment_phase.py <state_file_path>")
        print("Example: python reset_experiment_phase.py state_saves/Paper0.pkl")
        sys.exit(1)

    state_file = sys.argv[1]

    if not Path(state_file).exists():
        print(f"❌ State file not found: {state_file}")
        sys.exit(1)

    success = reset_experiment_phase(state_file)

    if success:
        print("\n✅ Experiment phase reset completed!")
        print("You can now restart the Agent Laboratory and it will regenerate the experiment code.")
    else:
        print("\n❌ Failed to reset experiment phase")
        sys.exit(1)