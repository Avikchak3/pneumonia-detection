import os
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# ----------------------------------------------------
# 1. PAGE CONFIGURATION & STYLING
# ----------------------------------------------------
st.set_page_config(
    page_title="Pneumonia Screening Demo",
    page_icon="🫁",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #007bff 0%, #0056b3 100%);
        color: white; font-weight: bold; border-radius: 8px;
        padding: 0.6rem; border: none;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1); transition: 0.3s ease;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #0056b3 0%, #003d82 100%);
        box-shadow: 0 6px 8px rgba(0,0,0,0.15);
    }
    .result-card-normal {
        padding: 20px; border-radius: 10px; background-color: #d4edda;
        border-left: 6px solid #28a745; color: #155724; text-align: center;
        font-size: 20px; font-weight: bold; animation: fadeIn 0.5s ease-in-out;
    }
    .result-card-pneumonia {
        padding: 20px; border-radius: 10px; background-color: #f8d7da;
        border-left: 6px solid #dc3545; color: #721c24; text-align: center;
        font-size: 20px; font-weight: bold; animation: fadeIn 0.5s ease-in-out;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    </style>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# 2. MODEL LOADING (model file is stored in the same GitHub repo as app.py)
# ----------------------------------------------------
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pneumonia_densenet121.keras")
THRESHOLD = 0.5   # decision threshold used for the test-set numbers shown in the sidebar


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error("Model file 'pneumonia_densenet121.keras' was not found next to app.py. "
                 "Add it to the GitHub repo and redeploy.")
        st.caption("Files in app folder: " + ", ".join(sorted(os.listdir(os.path.dirname(MODEL_PATH)))))
        return None
    try:
        return tf.keras.models.load_model(MODEL_PATH, compile=False)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


with st.spinner("Loading DenseNet121 model..."):
    model = load_model()

if model is None:
    st.stop()

# ----------------------------------------------------
# 3. SIDEBAR
# ----------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/color/96/lungs.png", width=80)
    st.markdown("### **About the System**")
    st.info(
        "A **DenseNet121** model (ImageNet transfer learning, fine-tuned) that gives a "
        "screening score for pneumonia on chest X-rays."
    )
    st.markdown("**Test set (624 images)**")
    st.markdown("- ROC-AUC: 0.959\n- Pneumonia recall: 96.4%\n- Pneumonia precision: 85.8%\n- Accuracy: 87.8%")
    st.markdown("---")
    st.markdown("### **Instructions**")
    st.markdown("1. Upload a chest X-ray image (`.jpg`, `.jpeg`, `.png`).")
    st.markdown("2. Click **Analyze X-Ray**.")
    st.markdown("3. View the model's screening score.")
    st.markdown("---")
    st.markdown("⚠️ *Educational demo only. Not a medical diagnosis or a medical device.*")

# ----------------------------------------------------
# 4. MAIN INTERFACE
# ----------------------------------------------------
st.title("🫁 Pneumonia Screening Demo")
st.warning("Educational demo, **not a diagnostic tool**. Do not use it for medical decisions.")
st.markdown("Upload a chest X-ray image to see the model's pneumonia screening score.")

uploaded_file = st.file_uploader("Choose a chest X-ray image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col1, col2 = st.columns(2, gap="medium")

    with col1:
        st.subheader("Uploaded X-Ray")
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded image")

    with col2:
        st.subheader("Analysis Panel")
        st.write("")
        st.write("")

        if st.button("🔬 Analyze X-Ray"):
            with st.spinner("Analyzing..."):
                # IMPORTANT: the saved model already contains Rescaling(1/255) and the ImageNet
                # normalization, so feed RAW pixel values (0-255). Do NOT divide by 255 here.
                img_array = np.array(image.resize((224, 224)), dtype="float32")
                img_array = np.expand_dims(img_array, axis=0)

                pneumonia_prob = float(model.predict(img_array, verbose=0)[0][0])
                pneumonia_prob = min(max(pneumonia_prob, 0.0), 1.0)

                if pneumonia_prob >= THRESHOLD:
                    st.markdown(
                        f'<div class="result-card-pneumonia">⚠️ Possible pneumonia<br>'
                        f'<span style="font-size:14px; font-weight:normal;">Model score: {pneumonia_prob:.2f}</span></div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f'<div class="result-card-normal">✅ No pneumonia pattern found<br>'
                        f'<span style="font-size:14px; font-weight:normal;">Model score: {pneumonia_prob:.2f}</span></div>',
                        unsafe_allow_html=True,
                    )

                st.markdown("### Score")
                st.progress(pneumonia_prob)
                st.caption(f"Pneumonia score: {pneumonia_prob:.4f} (threshold {THRESHOLD}). "
                           "This is a model output, not a calibrated probability.")
else:
    st.markdown("""
        <div style="text-align: center; padding: 40px; border: 2px dashed #ccc; border-radius: 10px; color: #666;">
            <h4>No image uploaded yet.</h4>
            <p>Upload a chest X-ray using the uploader above to begin.</p>
        </div>
    """, unsafe_allow_html=True)
