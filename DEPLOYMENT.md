# 🚀 Image Classification App - Deployment Guide

## 📋 Prerequisites for Deployment

1. **GitHub Repository** - Your code needs to be in a public GitHub repo
2. **Model Files** - Ensure `model.h5` and `class_names.pkl` are included
3. **Streamlit Cloud Account** - Free at [share.streamlit.io](https://share.streamlit.io)

## 🌐 Deploy to Streamlit Cloud

### Step 1: Upload to GitHub
```bash
# Initialize git repository
git init
git add .
git commit -m "Initial commit - Image Classification App"

# Add your GitHub repository
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
git push -u origin main
```

### Step 2: Deploy on Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click "New app"
3. Connect your GitHub account
4. Select your repository
5. Set main file path: `streamlit_app.py`
6. Click "Deploy!"

### Step 3: Configuration
- **Requirements file**: `requirements_streamlit.txt`
- **Python version**: 3.11
- **Main file**: `streamlit_app.py`

## 📁 Required Files for Deployment
```
📦 Your Repository
├── streamlit_app.py          # Main Streamlit app
├── model.h5                  # Trained CNN model (IMPORTANT!)
├── class_names.pkl           # Class labels (IMPORTANT!)
├── requirements_streamlit.txt # Dependencies
├── .streamlit/
│   └── config.toml          # Streamlit configuration
└── README.md               # This file
```

## ⚠️ Important Notes

1. **Model Files**: Your `model.h5` and `class_names.pkl` files MUST be included in the repository
2. **File Size**: GitHub has a 100MB limit per file. If your model is larger, consider using Git LFS
3. **Dependencies**: Use the provided `requirements_streamlit.txt` for compatibility

## 🎯 Alternative Deployment Options

### Option B: Render.com (Flask API)
- Deploy your Flask API (`app.py`)
- Free tier available
- Good for API-focused deployment

### Option C: Hugging Face Spaces
- Perfect for ML models
- Free hosting
- Great for community sharing

## 🔧 Troubleshooting

**Common Issues:**
1. **ModuleNotFoundError**: Check `requirements_streamlit.txt`
2. **Model not found**: Ensure `model.h5` is in the repository
3. **Memory issues**: Consider model optimization for cloud deployment

## 📞 Support
If you encounter issues, the most common cause is missing model files in the repository.
