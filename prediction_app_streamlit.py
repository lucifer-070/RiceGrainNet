import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
import io

# Set page configuration
st.set_page_config(
    page_title="RiceGrainNet Classifier",
    page_icon="🌾",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stTitle {
        font-size: 2.5rem;
        color: #2c3e50;
        text-align: center;
    }
    .stSubheader {
        font-size: 1.5rem;
        color: #34495e;
    }
    </style>
    """, unsafe_allow_html=True)

def load_model():
    """Load the trained TensorFlow model"""
    try:
        path = '/RiceGrainNet/rice_classifier_model2.keras'
        model = tf.keras.models.load_model(path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def preprocess_image(image):
    """Preprocess the uploaded image for model prediction"""
    # Resize image to match model's expected input size
    target_size = (256, 256)  # Adjust based on your model's input size
    image = image.resize(target_size)
    
    # Convert to array and preprocess
    # img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = tf.expand_dims(image, 0)
    
    # # Normalize pixel values
    # img_array = img_array / 255.0
    
    return img_array

def predict(model, image):
    """Make prediction using the loaded model"""
    try:
        # Get model prediction
        predictions = model.predict(image)
        
        # Class labels (adjust these based on your model's classes)
        class_names = ['Arborio', 'Basmati', 'Ipsala', 'Jasmine', 'Karacadag']
        
        # Get the predicted class and confidence
        predicted_class = class_names[np.argmax(predictions[0])]
        confidence = float(np.max(predictions[0]))
        
        return predicted_class, confidence, predictions[0]
    
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return None, None, None

def main():
    # Header
    st.title("🌾 RiceGrainNet Classifier")
    st.markdown("### Upload an image of rice grains for classification")
    
    # Load model
    model = load_model()
    
    if model is None:
        st.error("Please ensure the model file 'rice_classification_model1.keras' is present in the same directory.")
        return
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Create columns for layout
        col1, col2 = st.columns(2)
        
        with col1:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_container_width=False)
            
        with col2:
            # Add a prediction button
            if st.button('Classify Rice'):
                with st.spinner('Processing...'):
                    # Preprocess the image
                    processed_image = preprocess_image(image)
                    
                    # Get prediction
                    predicted_class, confidence, class_probabilities = predict(model, processed_image)
                    
                    if predicted_class and confidence:
                        # Display results
                        st.success(f"Predicted Rice Type: {predicted_class}")
                        st.info(f"Confidence: {confidence:.2%}")
                        
                        # Display probability distribution
                        st.subheader("Probability Distribution")
                        class_names = ['Arborio', 'Basmati', 'Ipsala', 'Jasmine', 'Karacadag']
                        probs_df = pd.DataFrame({
                            'Rice Type': class_names,
                            'Probability': class_probabilities
                        })
                        st.bar_chart(probs_df.set_index('Rice Type'))
    
    # Add information about the model
    with st.expander("About RiceGrainNet"):
        st.markdown("""
        This application uses a Convolutional Neural Network (CNN) trained on thousands of rice grain images 
        to classify different varieties of rice. The model can identify the following rice types:
        - Arborio
        - Basmati
        - Ipsala
        - Jasmine
        - Karacadag
        
        For best results:
        - Use clear, well-lit images
        - Ensure rice grains are clearly visible
        - Avoid blurry or dark images
        - Use images with minimal background clutter
        """)

if __name__ == '__main__':
    main()
