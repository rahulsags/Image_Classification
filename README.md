# Image Classification with CNN

A complete machine learning project that performs image classification using a Convolutional Neural Network (CNN) built with TensorFlow and Keras, deployed as a web API using Flask and an interactive interface with Streamlit.

<img width="1918" height="966" alt="image" src="https://github.com/user-attachments/assets/f1a78d08-31a4-4910-8ec7-b4ea620857ad" />


## Live Demo

**[Try the app live on Streamlit Cloud](https://imageclassification-rahul.streamlit.app/)**

*Note: The live demo shows the interface design. The full TensorFlow model runs locally due to Python 3.13 compatibility constraints on cloud platforms.*



## Project Overview

This project implements an end-to-end image classification pipeline with the following features:

- **Deep Learning Model**: CNN with multiple convolutional layers, batch normalization, and dropout
- **Dataset**: CIFAR-10 (10 object classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
- **Model Performance**: 76.35% accuracy on CIFAR-10 test set
- **Web API**: Flask-based REST API for image classification
- **Interactive Interface**: Streamlit app with professional UI/UX
- **Live Deployment**: Successfully deployed on Streamlit Cloud
- **Comprehensive Evaluation**: Training metrics, confusion matrix, and model performance analysis

## Quick Start

> **Want to get started immediately? Use the entry point scripts!**
> - `python run_training.py` - Train the model
> - `python run_flask.py` - Start Flask API
> - `python run_streamlit.py` - Start Streamlit app

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/rahulsags/Image_Classification.git
cd Image_Classification

# Create virtual environment (recommended)
python -m venv .venv
.venv\Scripts\activate  # On Windows
# source .venv/bin/activate  # On Mac/Linux

# Install dependencies
pip install -r requirements.txt
```

### 2. Train the Model

```bash
# Train the CNN model on CIFAR-10 dataset (using entry point script)
python run_training.py

# Or run directly from src directory
python src/train_cifar.py
```

The training script will:
- Download the CIFAR-10 dataset automatically
- Preprocess images with data augmentation
- Train a CNN model with early stopping
- Generate training plots and evaluation metrics
- Save the trained model as `models/model.h5` and `models/best_cifar_model.h5`
- Achieve ~76% accuracy on test set

### 3. Run the Applications

**Flask API** (using entry point script):
```bash
python run_flask.py
```
Access the web interface at: `http://localhost:5000`

**Streamlit Interface** (using entry point script):
```bash
python run_streamlit.py
```
Access the interactive interface at: `http://localhost:8501`

**Alternative - Direct execution**:
```bash
# Flask API
python apps/app.py

# Streamlit with TensorFlow
streamlit run apps/streamlit_app_with_tensorflow.py
```

### 4. Test the API

**Web Interface**: Open `http://localhost:5000` in your browser and upload an image.

**API Endpoint**: Send POST requests to `http://localhost:5000/predict`

```bash
# Example using curl
curl -X POST -F "image=@your_image.jpg" http://localhost:5000/predict
```

**API Response**:
```json
{
  "class": "cat",
  "confidence": 0.89,
  "all_predictions": {
    "airplane": 0.01,
    "automobile": 0.02,
    "bird": 0.03,
    "cat": 0.89,
    "deer": 0.01,
    "dog": 0.02,
    "frog": 0.01,
    "horse": 0.01,
    "ship": 0.00,
    "truck": 0.00
  }
}
```

## Project Structure

```
Image_Classification/
├── Entry Points
│   ├── run_training.py       # Easy training script launcher
│   ├── run_flask.py          # Easy Flask API launcher  
│   ├── run_streamlit.py      # Easy Streamlit app launcher
│   └── streamlit_app.py      # Entry point for Streamlit Cloud deployment
├── src/                      # Source code and training scripts
│   ├── train_cifar.py        # CIFAR-10 CNN training implementation
│   ├── train_model.py        # Alternative training script
│   └── utils.py              # Utility functions
├── apps/                     # Web applications
│   ├── app.py                # Flask API server
│   ├── streamlit_demo_app.py # Streamlit demo (reference version)
│   └── streamlit_app_with_tensorflow.py  # Full Streamlit app with TensorFlow
├── models/                   # Trained models and metadata
│   ├── model.h5              # Final trained CNN model
│   ├── best_cifar_model.h5   # Best model checkpoint during training
│   ├── class_names.pkl       # CIFAR-10 class labels
│   └── cifar_class_names.pkl # Alternative class names file
├── outputs/                  # Training results and visualizations
│   ├── cifar_training_history.png # Training/validation curves
│   └── cifar_confusion_matrix.png # Model performance evaluation
├── deployment/               # Deployment configurations
│   ├── Procfile              # Render.com deployment config
│   └── DEPLOYMENT.md         # Deployment instructions
├── data/                     # Dataset (auto-downloaded during training)
├── Configuration Files
│   ├── requirements.txt      # Python dependencies
│   ├── .streamlit/           # Streamlit app configuration
│   ├── .gitignore            # Git ignore rules
│   └── .python-version       # Python version specification
└── Documentation
    ├── README.md             # Main project documentation
    └── LICENSE               # Apache 2.0 License
```

## Model Architecture

The CNN model includes:

- **Input Layer**: 32×32×3 (RGB images from CIFAR-10)
- **Convolutional Blocks**: 5 blocks with Conv2D + BatchNormalization + MaxPooling
- **Feature Maps**: 32 → 32 → 64 → 64 → 128 filters
- **Regularization**: Dropout layers (0.25, 0.5) to prevent overfitting
- **Dense Layers**: 512 → 256 → 10 neurons (for 10 CIFAR-10 classes)
- **Output**: Softmax activation for multi-class classification

**Key Features**:
- Batch normalization for stable training
- Data augmentation (rotation, zoom, flip, shift)
- Early stopping and learning rate reduction
- Model checkpointing for best weights
- Achieved 76.35% accuracy on CIFAR-10 test set

## Training Details

### Dataset
- **Source**: CIFAR-10 (Canadian Institute for Advanced Research)
- **Classes**: 10 (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
- **Images**: 50,000 training + 10,000 test images
- **Size**: 32×32 pixels, RGB color
- **Split**: 80% training, 20% validation (from training set)
- **Preprocessing**: Normalize to [0,1], data augmentation applied

### Training Configuration
- **Optimizer**: Adam
- **Loss**: Sparse Categorical Crossentropy
- **Batch Size**: 32
- **Epochs**: 50 (with early stopping)
- **Callbacks**: EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
- **Final Accuracy**: 76.35% on test set

### Data Augmentation
- Rotation: ±20 degrees
- Width/Height shift: ±20%
- Horizontal flip: Yes
- Zoom: ±20%

## Deployment

### ✅ Streamlit Cloud (Currently Live)

**Live Demo**: [https://imageclassification-rahul.streamlit.app/](https://imageclassification-rahul.streamlit.app/)

The app is successfully deployed on Streamlit Cloud! Due to Python 3.13 compatibility constraints with TensorFlow on cloud platforms, the live version shows a demo interface. The full TensorFlow model works perfectly locally.

**Deployment Features**:
- Professional UI with image upload
- Interactive prediction interface
- Real-time processing simulation
- Responsive design
- Mobile-friendly interface

### 🔄 Local Development (Full TensorFlow Version)

For the complete experience with actual CNN predictions:

```bash
# Run locally with full TensorFlow model
streamlit run streamlit_app_with_tensorflow.py
```


## API Endpoints

### Main Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web interface for image upload |
| `/predict` | POST | Image classification API |
| `/health` | GET | Health check |
| `/api/info` | GET | API information |

### Prediction Endpoint Details

**URL**: `/predict`
**Method**: POST
**Content-Type**: `multipart/form-data`

**Parameters**:
- `image`: Image file (JPG, PNG, etc.)

**Response**:
```json
{
  "class": "cat",
  "confidence": 0.89,
  "all_predictions": {
    "airplane": 0.01,
    "automobile": 0.02,
    "bird": 0.03,
    "cat": 0.89,
    "deer": 0.01,
    "dog": 0.02,
    "frog": 0.01,
    "horse": 0.01,
    "ship": 0.00,
    "truck": 0.00
  }
}
```

**Error Response**:
```json
{
  "error": "Error message"
}
```

## Model Performance

After training, the model generates:

1. **Training History Plot** (`cifar_training_history.png`):
   - Training vs. validation accuracy
   - Training vs. validation loss
   - Learning curves over epochs

2. **Confusion Matrix** (`cifar_confusion_matrix.png`):
   - Detailed classification results for all 10 classes
   - True vs. predicted labels visualization
   - Per-class performance analysis

3. **Model Checkpoints**:
   - `model.h5`: Final trained model
   - `best_cifar_model.h5`: Best model during training
   - `class_names.pkl`: CIFAR-10 class labels

**Achieved Performance**:
- **Training Accuracy**: ~85%
- **Validation Accuracy**: ~76%
- **Test Accuracy**: 76.35%
- **Model Size**: 3.31 MB (866,602 parameters)

## Customization

### Using Different Datasets

1. **Modify `src/train_cifar.py`**:
   ```python
   # Change the dataset loading logic
   # CIFAR-10 is loaded automatically with:
   (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
   
   # For custom datasets, replace with:
   # Your custom dataset loading code
   ```

2. **Update preprocessing** if needed:
   ```python
   # Adjust image size, normalization, etc.
   # CIFAR-10 uses 32x32 images
   # For different sizes, modify the input shape
   ```

### Model Architecture Changes

```python
# In src/train_cifar.py, modify the SimpleCIFARClassifier class
class SimpleCIFARClassifier:
    def build_model(self):
        model = tf.keras.Sequential([
            # Modify layers here for your specific needs
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
            # ... your custom architecture
        ])
        return model
```

### Adding New Classes

1. **For CIFAR-100** (100 classes):
   ```python
   # In train_cifar.py, change:
   (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
   # Update num_classes = 100
   ```

2. **For custom datasets** organized in folders:
   ```
   data/
   ├── airplane/
   ├── automobile/
   ├── bird/
   └── ... (your classes)
   ```

## 🐛 Troubleshooting

### Common Issues

1. **Model not found error**:
   ```bash
   # Train the model first
   python train_cifar.py
   ```

2. **TensorFlow compatibility issues**:
   - Use Python 3.11 or earlier for full TensorFlow support
   - Python 3.13 has limited TensorFlow support on some platforms

3. **Memory errors during training**:
   - Reduce batch size in `train_cifar.py` (default: 32)
   - Use GPU if available for faster training

4. **Low accuracy**:
   - CIFAR-10 is more challenging than binary classification
   - 76.35% is good performance for CIFAR-10
   - Try data augmentation or different architectures for improvement

5. **Deployment issues**:
   - Check file sizes (model files: ~3.3MB)
   - Ensure all dependencies are in requirements.txt
   - For Streamlit Cloud: Python 3.13 TensorFlow compatibility is limited

### Performance Optimization

1. **Model Size Reduction**:
   ```python
   # Use model quantization
   model = tf.keras.models.load_model('model.h5')
   converter = tf.lite.TFLiteConverter.from_keras_model(model)
   tflite_model = converter.convert()
   ```

2. **Faster Inference**:
   - Use TensorFlow Serving for production
   - Implement model caching
   - Optimize image preprocessing

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.

## Acknowledgments

- **TensorFlow/Keras** for the deep learning framework
- **CIFAR-10 Dataset** by the Canadian Institute for Advanced Research
- **Flask** for the web API framework
- **Streamlit** for the interactive web interface
- **Streamlit Cloud** for hosting the live demo
- **GitHub** for version control and collaboration

## Project Stats

- **Model Accuracy**: 76.35% on CIFAR-10 test set
- **Images Processed**: 60,000 CIFAR-10 images during training
- **Model Parameters**: 866,602 trainable parameters
- **Inference Time**: ~250ms per image (local)
- **Model Size**: 3.31 MB

## Support

If you encounter any issues or have questions:

1. **Check the troubleshooting section** above
2. **Review error logs** in the terminal
3. **Open an issue** on GitHub with detailed information
4. **Ensure all dependencies** are correctly installed
5. **Check Python version compatibility** (3.11 recommended for full features)

## What's Next?

- **Improve Model**: Experiment with different architectures (ResNet, EfficientNet)
- **Add More Datasets**: Support for CIFAR-100, custom datasets
- **Mobile App**: Create a mobile interface using Flutter/React Native
- **Real-time Video**: Add webcam/video classification support
- **Model Optimization**: Implement TensorFlow Lite for mobile deployment

---

**Happy Learning and Coding!**

*Built with care using TensorFlow, Keras, Flask, and Streamlit*

**Live Demo**: [https://imageclassification-rahul.streamlit.app/](https://imageclassification-rahul.streamlit.app/)
