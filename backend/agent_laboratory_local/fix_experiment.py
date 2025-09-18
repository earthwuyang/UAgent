#!/usr/bin/env python3
"""
Fix existing broken experiment that has dummy implementations
"""
import os
import sys
from pathlib import Path

def fix_dummy_experiment(lab_dir: str):
    """Fix the dummy experiment implementation"""
    lab_path = Path(lab_dir)
    experiment_file = lab_path / "src" / "run_experiments.py"

    if not experiment_file.exists():
        print(f"❌ Experiment file not found: {experiment_file}")
        return False

    print(f"🔧 Fixing dummy implementation in {experiment_file}")

    # Read the current file
    with open(experiment_file, 'r') as f:
        content = f.read()

    # Check if it has the dummy implementation
    if "Dummy Answer" not in content:
        print("✅ No dummy implementation found - file appears correct")
        return True

    # Replace the dummy query_model function with real implementation
    dummy_function = '''# Define a function to query the model
def query_model(model_str, prompt, system_prompt, dashscope_api_key):
    # This is a placeholder for the actual model querying logic
    # In a real scenario, you would use an API or a library to interact with the model
    # For this example, we will just return a dummy response
    return f"\\boxed{Dummy Answer}"'''

    real_function = '''# Import the real query_model function
import sys
sys.path.append('..')
from inference import query_model as real_query_model

# Define a function to query the model
def query_model(model_str, prompt, system_prompt, dashscope_api_key):
    """Query the real qwen3-max-preview model"""
    try:
        response = real_query_model(
            model_str=model_str,
            prompt=prompt,
            system_prompt=system_prompt,
            dashscope_api_key=dashscope_api_key
        )
        return response
    except Exception as e:
        print(f"Error querying model: {e}")
        return f"\\boxed{{Error: {e}}}"'''

    # Replace the dummy function
    if dummy_function in content:
        content = content.replace(dummy_function, real_function)
        print("✅ Replaced dummy function with real API implementation")
    else:
        # Alternative approach - replace any function that returns "Dummy Answer"
        import re
        # Find and replace the dummy return statement
        content = re.sub(
            r'return f?"\\boxed\{Dummy Answer\}"',
            '''try:
        response = real_query_model(
            model_str=model_str,
            prompt=prompt,
            system_prompt=system_prompt,
            dashscope_api_key=dashscope_api_key
        )
        return response
    except Exception as e:
        print(f"Error querying model: {e}")
        return f"\\boxed{{Error: {e}}}"''',
            content
        )
        print("✅ Fixed dummy return statements")

    # Add the import if it's missing
    if "from inference import query_model as real_query_model" not in content:
        # Add import at the top after existing imports
        import_insertion_point = content.find("from datasets import load_dataset") + len("from datasets import load_dataset\n")
        import_line = "import sys\nsys.path.append('..')\nfrom inference import query_model as real_query_model\n"
        content = content[:import_insertion_point] + import_line + content[import_insertion_point:]
        print("✅ Added proper imports")

    # Write the fixed content back
    with open(experiment_file, 'w') as f:
        f.write(content)

    print(f"🎉 Fixed experiment file: {experiment_file}")

    # Test the first example to make sure it works
    print("🧪 Testing the fix with a sample problem...")
    try:
        # Create a simple test
        test_code = '''
import os
import sys
sys.path.append('..')
sys.path.append('../..')
os.environ["DASHSCOPE_API_KEY"] = os.getenv("DASHSCOPE_API_KEY", "test-key")

from inference import query_model
response = query_model(
    model_str="qwen3-max-preview",
    prompt="What is 2+2? Answer with \\boxed{4}",
    system_prompt="You are a mathematician.",
    dashscope_api_key=os.getenv("DASHSCOPE_API_KEY")
)
print(f"Test response: {response}")
'''
        exec(test_code)
        print("✅ Test passed - real API integration working")
    except Exception as e:
        print(f"⚠️  Test failed (but fix was applied): {e}")

    return True

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python fix_experiment.py <lab_directory>")
        print("Example: python fix_experiment.py MATH_research_dir/research_dir_0_lab_1")
        sys.exit(1)

    lab_dir = sys.argv[1]
    success = fix_dummy_experiment(lab_dir)

    if success:
        print("\n✅ Experiment fix completed!")
        print("You can now restart the Agent Laboratory or continue the current run.")
    else:
        print("\n❌ Failed to fix experiment")
        sys.exit(1)