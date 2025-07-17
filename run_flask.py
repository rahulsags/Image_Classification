#!/usr/bin/env python3
"""
Entry point script for running the Flask API server.
This script provides an easy way to start the Flask app from the project root.
"""

import sys
import os

# Add apps directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'apps'))

if __name__ == "__main__":
    print("Starting Flask API Server...")
    print("=" * 50)
    print("Server will be available at: http://localhost:5000")
    print("Press Ctrl+C to stop the server")
    print("=" * 50)
    
    # Import and run the Flask app
    from apps.app import app
    app.run(debug=True, host='0.0.0.0', port=5000)
