#!/usr/bin/env python3
"""
Entry point script for training the CIFAR-10 CNN model.
This script provides an easy way to run the training from the project root.
"""

import sys
import os

# Add src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from train_cifar import SimpleCIFARClassifier

if __name__ == "__main__":
    print("Starting CIFAR-10 CNN Training...")
    print("=" * 50)
    
    # Initialize and train the model
    classifier = SimpleCIFARClassifier()
    classifier.train()
    
    print("=" * 50)
    print("Training completed! Check the outputs/ and models/ directories for results.")
