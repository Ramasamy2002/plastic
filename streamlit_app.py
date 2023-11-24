import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image
import time

# Function to load the model with TensorFlow Hub
@st.cache(allow_output_mutation=True)
def load_model_with_hub(model_path):
    feature_extractor_model = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"
    pretrained_model = hub.KerasLayer(feature_extractor_model, input_shape=(224, 224, 3), trainable=False)

    with tf.keras.utils.custom_object_scope({'KerasLayer': hub.KerasLayer}):
        model = tf.keras.models.load_model(model_path)
    
    model.layers[0] = pretrained_model  # Replace the first layer with the TensorFlow Hub layer

    return model

# Streamlit app
st.title("Image Classification App")
st.sidebar.title("Choose an Image")

# Upload image through the sidebar
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg","png"])

if uploaded_file is not None:
    # Load the model
    model_path = 'mobile.h5'
    model = load_model_with_hub(model_path)

    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Function to make predictions
    def predict_image(image):
        start_time=time.time()
        image_array = np.array(image.resize((224, 224))) / 255.0
        result = model.predict(image_array[np.newaxis, ...])
        end_time=time.time()
        predict_time=end_time-start_time
        predicted_label_index = np.argmax(result)
        return predicted_label_index,predict_time

    # Make prediction
    predicted_label_index ,predict_time= predict_image(image)

    # Display the result
    class_labels = ['HDPE', 'PET', 'PP', 'PS', 'PVC']
    st.write("Prediction Result:")
    st.write(f"Predicted Class: {class_labels[predicted_label_index]}")
    st.write(f"Prediction time: {predict_time:.4f} seconds")
    

# Run the app with: streamlit run your_script.py
