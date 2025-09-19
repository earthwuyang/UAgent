"""
Pytest configuration file to set up proper Python path
"""
import sys
import os

# Add the backend directory to the Python path so 'app' module can be imported
backend_dir = os.path.dirname(os.path.abspath(__file__))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)