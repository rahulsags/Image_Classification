import streamlit as st
import numpy as np
from PIL import Image
import io

# Page configuration
st.set_page_config(
    page_title="Image Classification App - Demo",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Demo mode message
st.warning("⚠️ **Demo Mode**: TensorFlow is not compatible with Python 3.13 on Streamlit Cloud yet. This is a demo version showing the UI.")

def demo_predict(image):
    """Simulate prediction for demo purposes"""
    # Simulate processing time
    import time
    import random
    time.sleep(1)
    
    # CIFAR-10 classes
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    
    # Generate random predictions for demo
    predictions = np.random.dirichlet(np.ones(10), size=1)[0]
    
    # Get the predicted class
    predicted_class_idx = np.argmax(predictions)
    confidence = float(predictions[predicted_class_idx])
    predicted_class = class_names[predicted_class_idx]
    
    # Get all probabilities
    all_predictions = {class_names[i]: float(predictions[i]) for i in range(len(class_names))}
    
    return predicted_class, confidence, all_predictions

# Main app
def main():
    st.title("🖼️ AI Image Classifier")
    st.markdown("""
    ### 🚧 Demo Version
    This is a demonstration of the Image Classification interface. The actual CNN model requires TensorFlow, 
    which is not yet compatible with Python 3.13 on Streamlit Cloud.
    
    **Full Version Features:**
    - 🧠 Deep CNN with multiple convolutional layers
    - 📊 Trained on CIFAR-10 dataset (76.35% accuracy)
    - ⚡ Real-time image classification
    - 🎯 10 different object classes
    """)
    
    # Sidebar
    st.sidebar.title("🔧 Settings")
    st.sidebar.markdown("---")
    
    # File uploader
    st.sidebar.markdown("### 📁 Upload Image")
    uploaded_file = st.sidebar.file_uploader(
        "Choose an image file",
        type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
        help="Supported formats: JPG, JPEG, PNG, BMP, TIFF"
    )
    
    # Confidence threshold
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.1,
        help="Minimum confidence for predictions"
    )
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📸 Input Image")
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Image info
            st.markdown("### 📊 Image Information")
            col_info1, col_info2, col_info3 = st.columns(3)
            with col_info1:
                st.metric("Width", f"{image.size[0]}px")
            with col_info2:
                st.metric("Height", f"{image.size[1]}px")
            with col_info3:
                st.metric("Mode", image.mode)
        else:
            st.info("👆 Upload an image using the sidebar to get started!")
    
    with col2:
        st.markdown("### 🎯 Prediction Results")
        
        if uploaded_file is not None:
            if st.button("🔮 Predict (Demo)", type="primary", use_container_width=True):
                with st.spinner("🤖 Analyzing image... (Demo Mode)"):
                    predicted_class, confidence, all_predictions = demo_predict(image)
                
                if confidence >= confidence_threshold:
                    st.success(f"**Prediction: {predicted_class.upper()}**")
                    st.metric("Confidence", f"{confidence:.2%}")
                else:
                    st.warning(f"Low confidence prediction: {predicted_class}")
                    st.metric("Confidence", f"{confidence:.2%}")
                
                # Detailed predictions
                st.markdown("### 📈 All Predictions")
                
                # Sort predictions by confidence
                sorted_predictions = sorted(all_predictions.items(), key=lambda x: x[1], reverse=True)
                
                # Create progress bars for top 5 predictions
                for class_name, prob in sorted_predictions[:5]:
                    st.write(f"**{class_name.capitalize()}**")
                    st.progress(prob)
                    st.caption(f"{prob:.2%}")
                
                # Show prediction table
                with st.expander("📋 Detailed Results"):
                    prediction_data = []
                    for class_name, prob in sorted_predictions:
                        prediction_data.append({
                            "Class": class_name.capitalize(),
                            "Probability": f"{prob:.4f}",
                            "Percentage": f"{prob:.2%}"
                        })
                    st.dataframe(prediction_data, use_container_width=True)
        else:
            st.info("Upload an image to see predictions here!")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    ### 💡 About This Demo
    
    **Tech Stack:**
    - 🎨 **Frontend**: Streamlit
    - 🧠 **AI Model**: CNN (Convolutional Neural Network)
    - 📊 **Dataset**: CIFAR-10 (10 object classes)
    - 🔧 **Framework**: TensorFlow/Keras
    
    **GitHub Repository**: [rahulsags/Image_Classification](https://github.com/rahulsags/Image_Classification)
    
    🚧 **Note**: This is a demo version. The full version with trained CNN model will be available once TensorFlow supports Python 3.13 on Streamlit Cloud.
    """)

if __name__ == "__main__":
    main()
