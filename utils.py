"""
Utility functions for image classification project
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from PIL import Image
import requests
from pathlib import Path

def create_directory_structure():
    """Create necessary directories for the project"""
    directories = [
        'data',
        'models',
        'logs',
        'static',
        'templates'
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
    
    print("Directory structure created successfully!")

def download_sample_images():
    """Download sample images for testing"""
    sample_urls = [
        ("https://images.unsplash.com/photo-1546422904-90eab23c3d8e?w=400", "cat_sample.jpg"),
        ("https://images.unsplash.com/photo-1552053831-71594a27632d?w=400", "dog_sample.jpg")
    ]
    
    sample_dir = Path("sample_images")
    sample_dir.mkdir(exist_ok=True)
    
    for url, filename in sample_urls:
        filepath = sample_dir / filename
        if not filepath.exists():
            try:
                response = requests.get(url)
                with open(filepath, 'wb') as f:
                    f.write(response.content)
                print(f"Downloaded: {filename}")
            except:
                print(f"Failed to download: {filename}")

def plot_model_architecture(model, filename='model_architecture.png'):
    """Plot and save model architecture"""
    try:
        tf.keras.utils.plot_model(
            model,
            to_file=filename,
            show_shapes=True,
            show_layer_names=True,
            rankdir='TB',
            expand_nested=False,
            dpi=96
        )
        print(f"Model architecture saved as {filename}")
    except Exception as e:
        print(f"Could not plot model architecture: {e}")

def visualize_predictions(images, predictions, true_labels, class_names, num_images=8):
    """Visualize model predictions"""
    plt.figure(figsize=(15, 10))
    
    for i in range(min(num_images, len(images))):
        plt.subplot(2, 4, i + 1)
        plt.imshow(images[i])
        plt.axis('off')
        
        pred_class = class_names[np.argmax(predictions[i])]
        true_class = class_names[true_labels[i]] if true_labels is not None else "Unknown"
        confidence = np.max(predictions[i])
        
        color = 'green' if pred_class == true_class else 'red'
        plt.title(f'Pred: {pred_class}\nTrue: {true_class}\nConf: {confidence:.2f}', 
                 color=color, fontsize=10)
    
    plt.tight_layout()
    plt.savefig('prediction_samples.png', dpi=300, bbox_inches='tight')
    plt.show()

def analyze_model_performance(history):
    """Analyze and visualize model training performance"""
    # Training metrics summary
    final_train_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    final_train_loss = history.history['loss'][-1]
    final_val_loss = history.history['val_loss'][-1]
    
    print("\n" + "="*50)
    print("TRAINING PERFORMANCE SUMMARY")
    print("="*50)
    print(f"Final Training Accuracy:   {final_train_acc:.4f}")
    print(f"Final Validation Accuracy: {final_val_acc:.4f}")
    print(f"Final Training Loss:       {final_train_loss:.4f}")
    print(f"Final Validation Loss:     {final_val_loss:.4f}")
    print(f"Best Validation Accuracy:  {max(history.history['val_accuracy']):.4f}")
    print(f"Training Epochs:           {len(history.history['loss'])}")
    
    # Check for overfitting
    acc_diff = final_train_acc - final_val_acc
    loss_diff = final_val_loss - final_train_loss
    
    print("\n" + "="*50)
    print("OVERFITTING ANALYSIS")
    print("="*50)
    
    if acc_diff > 0.1:
        print("⚠️  WARNING: Potential overfitting detected!")
        print(f"   Training accuracy is {acc_diff:.3f} higher than validation")
    else:
        print("✅ Good generalization - no significant overfitting")
    
    if loss_diff > 0.5:
        print("⚠️  WARNING: Validation loss significantly higher than training loss")
    else:
        print("✅ Loss values look healthy")

def create_deployment_files():
    """Create additional deployment configuration files"""
    
    # Docker file for containerized deployment
    dockerfile_content = """FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
"""
    
    with open('Dockerfile', 'w') as f:
        f.write(dockerfile_content)
    
    # .gitignore file
    gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
venv/
env/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Project specific
data/
*.h5
*.pkl
logs/
*.png
*.jpg
*.jpeg

# Jupyter
.ipynb_checkpoints/

# Environment variables
.env
"""
    
    with open('.gitignore', 'w') as f:
        f.write(gitignore_content)
    
    print("Deployment files created:")
    print("- Dockerfile")
    print("- .gitignore")

def check_system_requirements():
    """Check system requirements and compatibility"""
    print("🔍 Checking System Requirements...")
    print("="*50)
    
    # Python version
    import sys
    print(f"Python Version: {sys.version}")
    
    # Available packages
    packages = [
        'tensorflow', 'numpy', 'matplotlib', 'sklearn', 
        'PIL', 'flask', 'pandas', 'seaborn'
    ]
    
    for package in packages:
        try:
            __import__(package)
            print(f"✅ {package}: Available")
        except ImportError:
            print(f"❌ {package}: Not found - install with 'pip install {package}'")
    
    # Check GPU availability
    try:
        import tensorflow as tf
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            print(f"🚀 GPU Support: {len(gpus)} GPU(s) detected")
            for i, gpu in enumerate(gpus):
                print(f"   GPU {i}: {gpu.name}")
        else:
            print("💻 GPU Support: Not available (using CPU)")
    except:
        print("❓ GPU Support: Cannot determine")
    
    print("="*50)

if __name__ == "__main__":
    # Run system check
    check_system_requirements()
    
    # Create project structure
    create_directory_structure()
    
    # Create deployment files
    create_deployment_files()
    
    print("\n✅ Project utilities initialized successfully!")
    print("You can now run 'python train_model.py' to start training.")
