import os
import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import h5py

# Load the model
with h5py.File(r'C:\Users\VICTUS\OneDrive\ドキュメント\Data Science\DL Project\prediction\driver_distraction_model.h5', 'r') as f:
    model = load_model(f)

# Class labels
class_labels = [
    "Safe", "Texting (Right)", "Phone (Right)", "Texting (Left)", "Phone (Left)",
    "Radio", "Drinking", "Reach Behind", "Hair & Makeup", "Talking"
]

# Function to preprocess the image
def preprocess_image(img):
    img = img.resize((224, 224))  # Resize to match model input size
    img_array = image.img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Configure the Streamlit page
st.set_page_config(page_title="Driver Distraction Detection", page_icon="🚗", layout="wide")

# Title and Description
st.markdown("<h1 style='text-align: center;'>🚗 Driver Distraction Detection</h1>", unsafe_allow_html=True)
st.write("Upload a driver's image to detect potential distractions using our pre-trained model.")

# Layout: Compact columns for better space usage
col1, col2 = st.columns([1, 1.2], gap="small")

# File uploader with compact spacing
with col1:
    uploaded_file = st.file_uploader("📁 **Upload an image**", type=["jpg", "jpeg", "png"])

# If an image is uploaded, display the prediction results
if uploaded_file is not None:
    # Display the uploaded image, scaled to fit
    img = image.load_img(uploaded_file)
    col1.image(img, caption='Uploaded Image', use_column_width=True)

    # Preprocess and predict
    img_array = preprocess_image(img)
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]

    # Display prediction results compactly
    with col2:
        st.markdown(f"### 🎯 **Prediction:** `{class_labels[predicted_class]}`")

        if predicted_class == 0:
            st.success("✅ **Safe Driving! No distractions detected.**")
        else:
            st.warning(f"⚠️ **Distraction Detected:** `{class_labels[predicted_class]}`")

        # Prediction probabilities with progress bars (show top 5 for compactness)
        st.subheader("🔍 **Top 5 Prediction Probabilities**")
        top_indices = np.argsort(predictions[0])[::-1][:5]  # Get top 5 predictions
        for i in top_indices:
            st.write(f"{class_labels[i]}:")
            st.progress(float(predictions[0][i]))

        # Compact visualization using a horizontal bar chart
        # fig, ax = plt.subplots(figsize=(4, 4))  # Smaller plot size
        # ax.barh(class_labels, predictions[0], color='skyblue')
        # ax.set_xlabel("Probability")
        # ax.set_xlim([0, 1])
        # ax.set_title("Probability Distribution", fontsize=12)
        # st.pyplot(fig)

else:
    # Display instructions if no file is uploaded
    st.info("👆 Upload an image to start the prediction. Ensure the image clearly shows the driver.")

# Footer with smaller font size to save space
st.markdown(
    """
    <hr>
    <p style='text-align: center; font-size: 12px;'>
    **Note:** This model detects common distractions such as texting and phone use. Use responsibly and drive safely! 🚦
    </p>
    """,
    unsafe_allow_html=True
)
