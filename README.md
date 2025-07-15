# 🤖 Image Classification with CNN

A complete machine learning project that performs image classification using a Convolutional Neural Network (CNN) built with TensorFlow and Keras, deployed as a web API using Flask.

## 📋 Project Overview

This project implements an end-to-end image classification pipeline with the following features:

- **Deep Learning Model**: CNN with multiple convolutional layers, batch normalization, and dropout
- **Dataset**: Dogs vs. Cats classification (can be adapted for other datasets)
- **Web API**: Flask-based REST API for image classification
- **Web Interface**: Interactive HTML interface for easy image upload and prediction
- **Multiple Deployment Options**: Support for Render, Hugging Face Spaces, and Streamlit Cloud
- **Comprehensive Evaluation**: Training metrics, confusion matrix, and model performance analysis

## 🚀 Quick Start

### 1. Installation

```bash
# Clone or download the project
git clone <your-repo-url>
cd Image_Classification

# Install dependencies
pip install -r requirements.txt
```

### 2. Train the Model

```bash
# This will download the dataset, train the model, and save it
python train_model.py
```

The training script will:
- Download the Dogs vs. Cats dataset automatically
- Preprocess images with data augmentation
- Train a CNN model with early stopping
- Generate training plots and evaluation metrics
- Save the trained model as `model.h5`

### 3. Run the Flask API

```bash
# Start the Flask development server
python app.py
```

Access the web interface at: `http://localhost:5000`

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
  "class": "Dog",
  "confidence": 0.95,
  "all_predictions": {
    "Cat": 0.05,
    "Dog": 0.95
  }
}
```

## 📁 Project Structure

```
Image_Classification/
├── train_model.py          # Model training script
├── app.py                  # Flask API server
├── gradio_app.py          # Gradio interface (alternative)
├── streamlit_app.py       # Streamlit interface (alternative)
├── requirements.txt       # Python dependencies
├── Procfile              # For Render deployment
├── README.md             # Project documentation
├── data/                 # Dataset (auto-downloaded)
├── model.h5              # Trained model (generated)
├── class_names.pkl       # Class labels (generated)
├── training_history.png  # Training plots (generated)
└── confusion_matrix.png  # Evaluation metrics (generated)
```

## 🧠 Model Architecture

The CNN model includes:

- **Input Layer**: 224×224×3 (RGB images)
- **Convolutional Blocks**: 4 blocks with Conv2D + BatchNormalization + MaxPooling
- **Feature Maps**: 32 → 64 → 128 → 256 filters
- **Regularization**: Dropout layers (0.3, 0.5) to prevent overfitting
- **Dense Layers**: 512 → 256 → num_classes neurons
- **Output**: Softmax activation for multi-class classification

**Key Features**:
- Batch normalization for stable training
- Data augmentation (rotation, zoom, flip, shift)
- Early stopping and learning rate reduction
- Model checkpointing for best weights

## 📊 Training Details

### Dataset
- **Source**: Dogs vs. Cats (Microsoft/Kaggle dataset)
- **Classes**: 2 (Cat, Dog)
- **Split**: 80% training, 20% validation
- **Preprocessing**: Resize to 224×224, normalize to [0,1]

### Training Configuration
- **Optimizer**: Adam
- **Loss**: Categorical Crossentropy
- **Batch Size**: 32
- **Epochs**: 30 (with early stopping)
- **Callbacks**: EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

### Data Augmentation
- Rotation: ±20 degrees
- Width/Height shift: ±20%
- Horizontal flip: Yes
- Zoom: ±20%

## 🌐 Deployment Options

### Option 1: Render.com (Recommended)

1. **Prepare for deployment**:
   - Ensure `requirements.txt` and `Procfile` are in your project
   - Train your model locally and include `model.h5` in your repository

2. **Deploy to Render**:
   - Create a new Web Service on [Render.com](https://render.com)
   - Connect your GitHub repository
   - Set build command: `pip install -r requirements.txt`
   - Set start command: `gunicorn app:app`
   - Deploy!

3. **Environment Variables** (if needed):
   ```
   PYTHON_VERSION=3.9.16
   ```

### Option 2: Hugging Face Spaces

1. **Create a new Space**:
   - Go to [Hugging Face Spaces](https://huggingface.co/spaces)
   - Choose "Gradio" as the SDK
   - Upload your files including `gradio_app.py`

2. **Files to upload**:
   ```
   gradio_app.py
   model.h5
   class_names.pkl
   requirements.txt
   ```

3. **Space will automatically deploy** with the Gradio interface

### Option 3: Streamlit Cloud

1. **Prepare Streamlit app**:
   - Use `streamlit_app.py` as your main file
   - Ensure all dependencies are in `requirements.txt`

2. **Deploy to Streamlit Cloud**:
   - Go to [Streamlit Cloud](https://streamlit.io/cloud)
   - Connect your GitHub repository
   - Select `streamlit_app.py` as the main file
   - Deploy!

## 🔧 API Endpoints

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
  "class": "Dog",
  "confidence": 0.95,
  "all_predictions": {
    "Cat": 0.05,
    "Dog": 0.95
  }
}
```

**Error Response**:
```json
{
  "error": "Error message"
}
```

## 📈 Model Performance

After training, the model generates:

1. **Training History Plot** (`training_history.png`):
   - Training vs. validation accuracy
   - Training vs. validation loss

2. **Confusion Matrix** (`confusion_matrix.png`):
   - Detailed classification results
   - True vs. predicted labels

3. **Classification Report**:
   - Precision, recall, F1-score
   - Per-class and overall metrics

**Expected Performance**:
- Training Accuracy: ~95%+
- Validation Accuracy: ~90%+
- Test Accuracy: ~88%+

## 🛠️ Customization

### Using Different Datasets

1. **Modify `train_model.py`**:
   ```python
   # Change the dataset download URL and extraction logic
   def download_dataset(self):
       # Your custom dataset download code
   ```

2. **Update preprocessing** if needed:
   ```python
   # Adjust image size, normalization, etc.
   def prepare_data(self, data_dir):
       # Your custom preprocessing
   ```

### Model Architecture Changes

```python
# In train_model.py, modify build_model() method
def build_model(self):
    model = models.Sequential([
        # Add/remove/modify layers here
        layers.Conv2D(64, (3, 3), activation='relu', input_shape=(...)),
        # ... your custom architecture
    ])
```

### Adding New Classes

1. **Retrain with new data** organized in folders:
   ```
   data/
   ├── class1/
   ├── class2/
   ├── class3/
   └── class4/
   ```

2. **Update `num_classes`** in the model initialization

## 🐛 Troubleshooting

### Common Issues

1. **Model not found error**:
   ```bash
   # Train the model first
   python train_model.py
   ```

2. **Memory errors during training**:
   - Reduce batch size in `train_model.py`
   - Use smaller image size (e.g., 128×128)

3. **Low accuracy**:
   - Increase training epochs
   - Add more data augmentation
   - Try different learning rates

4. **Deployment issues**:
   - Check file sizes (model files can be large)
   - Ensure all dependencies are in requirements.txt
   - Verify Python version compatibility

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

## 📝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **TensorFlow/Keras** for the deep learning framework
- **Flask** for the web API framework
- **Microsoft/Kaggle** for the Dogs vs. Cats dataset
- **Gradio & Streamlit** for easy interface creation
- **Render, Hugging Face, Streamlit Cloud** for deployment platforms

## 📞 Support

If you encounter any issues or have questions:

1. Check the troubleshooting section above
2. Review the error logs in the terminal
3. Open an issue on GitHub with detailed information
4. Ensure all dependencies are correctly installed

---

**Happy Learning! 🚀**

*Built with ❤️ using TensorFlow, Keras, and Flask*
