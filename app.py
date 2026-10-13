import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# ----------------------------------------------------
# 1. PAGE CONFIGURATION & STYLING
# ----------------------------------------------------
st.set_page_config(
    page_title="Pneumonia Detection AI",
    page_icon="🫁",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern, animated, professional look
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #007bff 0%, #0056b3 100%);
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.6rem;
        border: none;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: 0.3s ease;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #0056b3 0%, #003d82 100%);
        box-shadow: 0 6px 8px rgba(0,0,0,0.15);
    }
    .result-card-normal {
        padding: 20px;
        border-radius: 10px;
        background-color: #d4edda;
        border-left: 6px solid #28a745;
        color: #155724;
        text-align: center;
        font-size: 20px;
        font-weight: bold;
        animation: fadeIn 0.5s ease-in-out;
    }
    .result-card-pneumonia {
        padding: 20px;
        border-radius: 10px;
        background-color: #f8d7da;
        border-left: 6px solid #dc3545;
        color: #721c24;
        text-align: center;
        font-size: 20px;
        font-weight: bold;
        animation: fadeIn 0.5s ease-in-out;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    </style>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# 2. MODEL LOADING (Cached for performance)
# ----------------------------------------------------
@st.cache_resource
def load_model():
    # Make sure 'pneumonia_densenet121.keras' is in your root directory
    model_path = "pneumonia_densenet121.keras"
    model = tf.keras.models.load_model(model_path)
    return model

with st.spinner("🔄 Loading DenseNet121 AI Engine... Please wait."):
    model = load_model()

# ----------------------------------------------------
# 3. SIDEBAR (Information & Guide)
# ----------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/color/96/lungs.png", width=80)
    st.markdown("### **About the System**")
    st.info(
        "This application uses a deep learning **DenseNet121** model pre-trained on ImageNet "
        "and fine-tuned on chest X-ray datasets to detect signs of pneumonia."
    )
    st.markdown("---")
    st.markdown("### **Instructions**")
    st.markdown("1. Upload a chest X-ray image (`.jpg`, `.jpeg`, `.png`).")
    st.markdown("2. Click **Analyze X-Ray**.")
    st.markdown("3. View prediction results and confidence score.")
    st.markdown("---")
    st.markdown("⚠️ *Disclaimer: For educational & screening purposes only. Not a definitive medical diagnosis.*")

# ----------------------------------------------------
# 4. MAIN INTERFACE CONTENT
# ----------------------------------------------------
st.title("🫁 AI-Powered Pneumonia Detection")
st.markdown("Upload a patient chest X-ray image below to instantly analyze for clinical features of pneumonia.")

# File uploader widget
uploaded_file = st.file_uploader(
    "Choose a chest X-ray image...", 
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    col1, col2 = st.columns(2, gap="medium")
    
    with col1:
        st.subheader("Uploaded X-Ray")
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, use_container_width=True, caption="Patient Scan")

    with col2:
        st.subheader("Analysis Panel")
        st.write("")
        st.write("")
        
        # Action button
        if st.button("🔬 Analyze X-Ray"):
            with st.spinner("Analyzing structural features via DenseNet121..."):
                # Preprocess image to match model input dimensions (typically 224x224)
                img_resized = image.resize((224, 224))
                img_array = np.array(img_resized) / 255.0
                img_array = np.expand_dims(img_array, axis=0)
                
                # Model Prediction
                prediction = model.predict(img_array)[0][0]
                
                # Logic: Assuming 0 = NORMAL, 1 = PNEUMONIA (adjust if reversed based on your training labels)
                # If your model outputs probability of pneumonia:
                pneumonia_prob = float(prediction)
                normal_prob = 1.0 - pneumonia_prob
                
                if pneumonia_prob > 0.5:
                    confidence = pneumonia_prob * 100
                    st.markdown(
                        f'<div class="result-card-pneumonia">⚠️ PNEUMONIA DETECTED<br><span style="font-size:14px; font-weight:normal;">Confidence: {confidence:.2f}%</span></div>', 
                        unsafe_allow_html=True
                    )
                else:
                    confidence = normal_prob * 100
                    st.markdown(
                        f'<div class="result-card-normal">✅ NORMAL LUNGS<br><span style="font-size:14px; font-weight:normal;">Confidence: {confidence:.2f}%</span></div>', 
                        unsafe_allow_html=True
                    )
                
                # Display metric breakdown
                st.markdown("### Confidence Breakdown")
                st.progress(pneumonia_prob)
                st.caption(f"Pneumonia Probability Score: {pneumonia_prob:.4f}")

else:
    # Empty state placeholder prompt
    st.markdown("""
        <div style="text-align: center; padding: 40px; border: 2px dashed #ccc; border-radius: 10px; color: #666;">
            <h4>No image uploaded yet.</h4>
            <p>Upload a chest X-ray using the uploader above to begin analysis.</p>
        </div>
    """, unsafe_allow_html=True)