#!/usr/bin/env python3
"""
Entry point script for running the Streamlit app with TensorFlow.
This script provides an easy way to start the Streamlit app from the project root.
"""

import sys
import os
import subprocess

if __name__ == "__main__":
    print("Starting Streamlit App with TensorFlow...")
    print("=" * 50)
    print("App will be available at: http://localhost:8501")
    print("Press Ctrl+C to stop the server")
    print("=" * 50)
    
    # Run streamlit app from apps directory
    app_path = os.path.join("apps", "streamlit_app_with_tensorflow.py")
    subprocess.run([sys.executable, "-m", "streamlit", "run", app_path])
