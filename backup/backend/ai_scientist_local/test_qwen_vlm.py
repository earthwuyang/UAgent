#!/usr/bin/env python3
"""
Test script to verify qwen-vl-max model functionality
"""
import os
import sys
import traceback
from ai_scientist.vlm import create_client, get_response_from_vlm

def test_qwen_vlm():
    """Test if qwen-vl-max model is working correctly"""

    # Check if DASHSCOPE_API_KEY is set
    if "DASHSCOPE_API_KEY" not in os.environ:
        print("❌ ERROR: DASHSCOPE_API_KEY environment variable not set")
        return False

    try:
        print("🔧 Testing qwen-vl-max model creation...")

        # Test client creation
        client, model = create_client("qwen-vl-max")
        print(f"✅ Successfully created client for model: {model}")

        # Create a simple test image (we'll use a figure from the experiment if available)
        test_image_path = None

        # Look for any PNG file in the experiments directory
        import glob
        png_files = glob.glob("/home/wuy/AI/AI-Scientist-v2/experiments/*/figures/*.png")
        if png_files:
            test_image_path = png_files[0]
            print(f"📷 Using test image: {test_image_path}")
        else:
            print("⚠️  No test image found in experiments, creating a simple test...")
            # Create a simple matplotlib figure for testing
            import matplotlib.pyplot as plt
            import numpy as np

            fig, ax = plt.subplots(figsize=(6, 4))
            x = np.linspace(0, 10, 100)
            y = np.sin(x)
            ax.plot(x, y, label='sin(x)')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_title('Test Plot for VLM')
            ax.legend()

            test_image_path = "/tmp/test_vlm_plot.png"
            plt.savefig(test_image_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"📷 Created test image: {test_image_path}")

        # Test VLM response
        print("🧠 Testing VLM response...")
        test_message = "Please describe what you see in this image. What kind of plot or visualization is this?"
        system_message = "You are a helpful assistant that analyzes images and provides detailed descriptions."

        response, _ = get_response_from_vlm(
            msg=test_message,
            image_paths=test_image_path,
            client=client,
            model=model,
            system_message=system_message,
            temperature=0.7
        )

        print("✅ VLM Response received:")
        print(f"📝 Response: {response[:200]}..." if len(response) > 200 else f"📝 Response: {response}")

        return True

    except Exception as e:
        print(f"❌ ERROR testing qwen-vl-max: {str(e)}")
        print("📋 Full traceback:")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Starting qwen-vl-max model test...")
    success = test_qwen_vlm()

    if success:
        print("\n✅ qwen-vl-max model test PASSED!")
        sys.exit(0)
    else:
        print("\n❌ qwen-vl-max model test FAILED!")
        sys.exit(1)