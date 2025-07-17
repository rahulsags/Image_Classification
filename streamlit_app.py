import streamlit as st
import os
import sys

# Add the apps directory to the path
sys.path.append('apps')

# Simple demo app since TensorFlow isn't working on cloud
st.set_page_config(
    page_title="AI Image Classifier",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AI Image Classifier")
st.markdown("### Demo Version")

st.warning("⚠️ Demo Mode: TensorFlow is not compatible with Python 3.13 on Streamlit Cloud yet. This is a demo version showing the UI.")

st.markdown("---")

# Create two columns
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 📤 Input Image")
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['png', 'jpg', 'jpeg'],
        help="Upload an image to classify"
    )
    
    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

with col2:
    st.markdown("### 🔍 Prediction Results")
    
    if uploaded_file is not None:
        st.success("✅ Image uploaded successfully!")
        
        # Demo prediction
        st.markdown("**Predicted Class:** `airplane` (Demo)")
        st.markdown("**Confidence:** `85.2%` (Demo)")
        
        st.info("💡 This is a demo prediction. The full model runs locally with TensorFlow.")
    else:
        st.info("👆 Upload an image to see predictions here!")

st.markdown("---")
st.markdown("### 🚀 Features")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**🧠 Deep CNN**")
    st.write("5-layer CNN with batch normalization")

with col2:
    st.markdown("**🎯 CIFAR-10**")
    st.write("Trained on 60,000 images, 10 classes")

with col3:
    st.markdown("**📊 76.35% Accuracy**")
    st.write("High performance on test dataset")

st.markdown("---")
st.markdown("**GitHub:** [View Source Code](https://github.com/rahulsags/Image_Classification)")