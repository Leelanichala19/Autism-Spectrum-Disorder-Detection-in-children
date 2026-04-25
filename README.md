# Autism Spectrum Disorder Detection in Children

## Project Overview
This project uses Deep Learning (CNN with MobileNetV2) to detect Autism Spectrum Disorder (ASD) from facial images.  
The system is deployed as a Streamlit web application for real-time prediction.

## Features
- Face detection using OpenCV
- Deep learning-based classification
- Real-time prediction using Streamlit
- Displays confidence score

## Model Details
- Model: MobileNetV2 (Transfer Learning)
- Input Size: 224 × 224 × 3
- Output: Binary classification (Autistic / Non-Autistic)
- Activation: Sigmoid

## Technologies Used
- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- Streamlit

## Performance
- Accuracy: ~90%
- High recall for autistic class
- Balanced performance

## How to Run

1. Install dependencies:
pip install -r requirements.txt

2. Run the app:
streamlit run app.py

3. Open in browser:
http://localhost:8501

##Output
- Predicts:
  - Autistic
  - Non-Autistic
- Displays confidence score

##Future Work
- Improve performance on real-world images
- Add explainable AI (SHAP)
- Deploy on cloud

## Author
Leela Nichala Chilakala  
MSc Data Science - GITAM University