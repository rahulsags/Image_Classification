from flask import Flask, request, jsonify, render_template_string
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import base64
import pickle
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Global variables
model = None
class_names = None

def load_model_and_classes():
    """Load the trained model and class names"""
    global model, class_names
    
    try:
        # Load model
        if os.path.exists('model.h5'):
            model = tf.keras.models.load_model('model.h5')
            print("Model loaded successfully!")
        else:
            print("Model file not found! Please train the model first.")
            return False
        
        # Load class names
        if os.path.exists('class_names.pkl'):
            with open('class_names.pkl', 'rb') as f:
                class_names = pickle.load(f)
            print(f"Class names loaded: {class_names}")
        else:
            print("Class names file not found! Using default classes.")
            class_names = ['Cat', 'Dog']  # Default for dogs vs cats
        
        return True
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return False

def preprocess_image(image):
    """Preprocess image for prediction"""
    try:
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize to model input size (32x32 for CIFAR-10)
        image = image.resize((32, 32))
        
        # Convert to numpy array and normalize
        img_array = np.array(image)
        img_array = img_array.astype('float32') / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    except Exception as e:
        raise ValueError(f"Error preprocessing image: {str(e)}")

@app.route('/')
def home():
    """Home page with upload interface"""
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Image Classification API</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f5f5f5;
            }
            .container {
                background-color: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }
            h1 {
                color: #333;
                text-align: center;
                margin-bottom: 30px;
            }
            .upload-area {
                border: 2px dashed #ccc;
                border-radius: 10px;
                padding: 40px;
                text-align: center;
                margin: 20px 0;
                background-color: #fafafa;
            }
            .upload-area:hover {
                border-color: #007bff;
                background-color: #f0f8ff;
            }
            input[type="file"] {
                margin: 10px 0;
                padding: 10px;
                border: 1px solid #ddd;
                border-radius: 5px;
                width: 100%;
                max-width: 300px;
            }
            button {
                background-color: #007bff;
                color: white;
                padding: 12px 30px;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                font-size: 16px;
                margin: 10px;
            }
            button:hover {
                background-color: #0056b3;
            }
            .result {
                margin-top: 20px;
                padding: 15px;
                border-radius: 5px;
                display: none;
            }
            .success {
                background-color: #d4edda;
                border: 1px solid #c3e6cb;
                color: #155724;
            }
            .error {
                background-color: #f8d7da;
                border: 1px solid #f5c6cb;
                color: #721c24;
            }
            .preview-image {
                max-width: 300px;
                max-height: 300px;
                margin: 10px 0;
                border-radius: 5px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.2);
            }
            .api-info {
                background-color: #e9ecef;
                padding: 15px;
                border-radius: 5px;
                margin-top: 30px;
            }
            .api-info h3 {
                margin-top: 0;
                color: #495057;
            }
            code {
                background-color: #f8f9fa;
                padding: 2px 4px;
                border-radius: 3px;
                font-family: monospace;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🤖 Image Classification API</h1>
            <p style="text-align: center; color: #666;">
                Upload an image to classify it using our trained CNN model<br>
                <small>Trained on CIFAR-10: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck</small>
            </p>
            
            <div class="upload-area">
                <h3>📸 Upload Image</h3>
                <input type="file" id="imageInput" accept="image/*" onchange="previewImage(event)">
                <br>
                <button onclick="predictImage()">🔍 Classify Image</button>
                <div id="imagePreview"></div>
            </div>
            
            <div id="result" class="result"></div>
            
            <!-- Collapsible API documentation -->
            <details class="api-info">
                <summary style="cursor: pointer; font-weight: bold; padding: 10px; background-color: #e9ecef; border-radius: 5px; margin-top: 20px;">
                    🔧 API Documentation (Click to expand)
                </summary>
                <div style="padding: 15px; border: 1px solid #dee2e6; border-top: none; border-radius: 0 0 5px 5px;">
                    <h3>API Endpoint</h3>
                    <p><strong>POST</strong> <code>/predict</code></p>
                    <p><strong>Content-Type:</strong> <code>multipart/form-data</code></p>
                    <p><strong>Parameter:</strong> <code>image</code> (image file)</p>
                    <p><strong>Response:</strong> JSON with prediction and confidence</p>
                    <p><strong>Classes:</strong> airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck</p>
                    
                    <h4>Example using curl:</h4>
                    <code>curl -X POST -F "image=@your_image.jpg" {{ request.url_root }}predict</code>
                </div>
            </details>
        </div>
        
        <script>
            function previewImage(event) {
                const file = event.target.files[0];
                const preview = document.getElementById('imagePreview');
                
                if (file) {
                    const reader = new FileReader();
                    reader.onload = function(e) {
                        preview.innerHTML = '<img src="' + e.target.result + '" class="preview-image" alt="Preview">';
                    };
                    reader.readAsDataURL(file);
                }
            }
            
            function predictImage() {
                const fileInput = document.getElementById('imageInput');
                const resultDiv = document.getElementById('result');
                
                if (!fileInput.files[0]) {
                    showResult('Please select an image first!', 'error');
                    return;
                }
                
                const formData = new FormData();
                formData.append('image', fileInput.files[0]);
                
                // Show loading
                showResult('🔄 Analyzing image...', 'success');
                
                fetch('/predict', {
                    method: 'POST',
                    body: formData
                })
                .then(response => response.json())
                .then(data => {
                    if (data.error) {
                        showResult('❌ Error: ' + data.error, 'error');
                    } else {
                        const confidence = (data.confidence * 100).toFixed(2);
                        showResult(
                            `✅ Prediction: <strong>${data.class}</strong><br>` +
                            `📊 Confidence: <strong>${confidence}%</strong>`,
                            'success'
                        );
                    }
                })
                .catch(error => {
                    showResult('❌ Error: ' + error.message, 'error');
                });
            }
            
            function showResult(message, type) {
                const resultDiv = document.getElementById('result');
                resultDiv.innerHTML = message;
                resultDiv.className = 'result ' + type;
                resultDiv.style.display = 'block';
            }
        </script>
    </body>
    </html>
    """
    return render_template_string(html_template)

@app.route('/predict', methods=['POST'])
def predict():
    """Predict image class"""
    try:
        # Check if model is loaded
        if model is None or class_names is None:
            return jsonify({'error': 'Model not loaded. Please check server configuration.'}), 500
        
        # Check if image is provided
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image selected'}), 400
        
        # Read and preprocess image
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        # Preprocess image
        processed_image = preprocess_image(image)
        
        # Make prediction
        predictions = model.predict(processed_image)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])
        predicted_class = class_names[predicted_class_idx]
        
        # Return result
        result = {
            'class': predicted_class,
            'confidence': confidence,
            'all_predictions': {
                class_names[i]: float(predictions[0][i]) 
                for i in range(len(class_names))
            }
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'classes': class_names
    })

@app.route('/api/info')
def api_info():
    """API information endpoint"""
    return jsonify({
        'name': 'Image Classification API',
        'version': '1.0.0',
        'model': 'CNN with TensorFlow/Keras',
        'input_size': '32x32x3',
        'classes': class_names,
        'endpoints': {
            '/': 'Web interface for image upload',
            '/predict': 'POST - Image classification',
            '/health': 'GET - Health check',
            '/api/info': 'GET - API information'
        }
    })

if __name__ == '__main__':
    print("🚀 Starting Image Classification API Server")
    print("=" * 50)
    
    # Load model and class names
    if load_model_and_classes():
        print("✅ Server ready!")
        print(f"📊 Model classes: {class_names}")
        print("🌐 Access the web interface at: http://localhost:5000")
        print("🔧 API endpoint: http://localhost:5000/predict")
        app.run(host='0.0.0.0', port=5000, debug=True)
    else:
        print("❌ Failed to load model. Please train the model first by running:")
        print("python train_model.py")
