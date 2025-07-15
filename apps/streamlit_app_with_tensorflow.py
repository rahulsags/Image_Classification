import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pickle
import os
import io

# Page configuration
st.set_page_config(
    page_title="Image Classification App",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Global variables
@st.cache_resource
def load_model_and_classes():
    """Load the trained model and class names"""
    try:
        # Check for model files
        model_path = 'model.h5'
        classes_path = 'class_names.pkl'
        
        if not os.path.exists(model_path):
            st.error("⚠️ Model file not found! Please ensure 'model.h5' is in the repository.")
            return None, None
            
        # Load model
        model = tf.keras.models.load_model(model_path)
        
        # Load class names
        if os.path.exists(classes_path):
            with open(classes_path, 'rb') as f:
                class_names = pickle.load(f)
        else:
            class_names = ['Cat', 'Dog']  # Default
        
        return model, class_names
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

def preprocess_image(image):
    """Preprocess image for prediction"""
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

def predict_image(image, model, class_names):
    """Predict image class using the trained model"""
    try:
        # Preprocess image
        processed_image = preprocess_image(image)
        
        # Make prediction
        predictions = model.predict(processed_image)
        
        # Get the predicted class
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])
        predicted_class = class_names[predicted_class_idx]
        
        # Get all probabilities
        all_predictions = {class_names[i]: float(predictions[0][i]) for i in range(len(class_names))}
        
        return predicted_class, confidence, all_predictions
    
    except Exception as e:
        return None, None, str(e)

# Main app
def main():
    # Title and description
    st.title("🤖 Image Classification with CNN")
    st.markdown("---")
    
    st.markdown("""
    ### Welcome to our Image Classification App!
    
    This application uses a **Convolutional Neural Network (CNN)** trained with TensorFlow/Keras 
    to classify images. Upload an image and get instant predictions with confidence scores.
    
        """)
    
    # Load model
    model, class_names = load_model_and_classes()
    
    if model is None or class_names is None:
        st.error("❌ Model not found! Please train the model first by running `train_model.py`")
        st.stop()
    
    st.success(f"✅ Model loaded successfully! Classes: {', '.join(class_names)}")
    
    # Sidebar
    st.sidebar.title("🔧 Settings")
    st.sidebar.markdown("---")
    
    # Model info in sidebar
    st.sidebar.markdown("### Model Information")
    st.sidebar.info(f"""
    **Classes:** {len(class_names)}
    **Input Size:** 32×32×3
    **Architecture:** CNN with {len(model.layers)} layers
    **Framework:** TensorFlow/Keras
    """)
    
    # File uploader
    st.sidebar.markdown("### 📁 Upload Image")
    uploaded_file = st.sidebar.file_uploader(
        "Choose an image file",
        type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
        help="Supported formats: JPG, JPEG, PNG, BMP, TIFF"
    )
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📸 Input Image")
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            # Image info
            st.markdown(f"""
            **Image Info:**
            - **Size:** {image.size[0]} × {image.size[1]} pixels
            - **Mode:** {image.mode}
            - **Format:** {image.format}
            """)
            
        else:
            # Placeholder
            st.info("👆 Please upload an image using the sidebar")
            st.image("https://via.placeholder.com/300x300?text=Upload+Image", 
                    caption="Waiting for image...", use_column_width=True)
    
    with col2:
        st.markdown("### 🔍 Prediction Results")
        
        if uploaded_file is not None:
            with st.spinner("🔄 Analyzing image..."):
                # Make prediction
                predicted_class, confidence, all_predictions = predict_image(image, model, class_names)
                
                if predicted_class is not None:
                    # Display main prediction
                    st.success(f"**Predicted Class:** {predicted_class}")
                    st.success(f"**Confidence:** {confidence:.2%}")
                    
                    # Progress bar for confidence
                    st.progress(confidence)
                    
                    # All predictions
                    st.markdown("### 📊 All Class Probabilities")
                    
                    for class_name, prob in sorted(all_predictions.items(), key=lambda x: x[1], reverse=True):
                        st.metric(
                            label=class_name,
                            value=f"{prob:.2%}",
                            delta=None
                        )
                        st.progress(prob)
                    
                    # Interpretation
                    st.markdown("### Interpretation")
                    if confidence > 0.8:
                        st.success("**High Confidence:** The model is very confident about this prediction!")
                    elif confidence > 0.6:
                        st.warning("**Medium Confidence:** The model is reasonably confident.")
                    else:
                        st.error("**Low Confidence:** The model is uncertain. Try a clearer image.")
                
                else:
                    st.error(f"❌ **Error during prediction:** {all_predictions}")
        
        else:
            st.info("Upload an image to see prediction results here.")
    
    # Additional features
    st.markdown("---")
    
    # Expander for technical details
    with st.expander("Technical Details"):
        st.markdown("""
        ### Model Architecture
        - **Input Layer:** 32×32×3 (RGB images)
        - **Convolutional Layers:** Multiple Conv2D layers with ReLU activation
        - **Pooling Layers:** MaxPooling2D for spatial downsampling
        - **Regularization:** Batch Normalization and Dropout
        - **Output Layer:** Dense layer with Softmax activation
        
        ### Preprocessing Pipeline
        1. **Resize:** Images are resized to 32×32 pixels
        2. **Normalization:** Pixel values are scaled to [0, 1] range
        3. **Format:** Converted to RGB if needed
        4. **Batching:** Single image is expanded to batch dimension
        
        ### Training Details
        - **Optimizer:** Adam with adaptive learning rate
        - **Loss Function:** Categorical Crossentropy
        - **Callbacks:** EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
        - **Data Augmentation:** Rotation, zoom, flip, shift
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        Built with ❤️ by M Rahul Sagayaraj<br>
        <small>Image Classification CNN • Deep Learning • Computer Vision</small>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
