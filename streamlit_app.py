
import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image

# Load the pre-trained model
feature_extractor_model = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"
pretrained_model = hub.KerasLayer(
    feature_extractor_model, input_shape=(224, 224, 3), trainable=False)

# Create the classification model
num_of_classes = 5
model = tf.keras.Sequential([
    pretrained_model,
    tf.keras.layers.Dense(num_of_classes, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Function to make predictions
def predict_image(image):
    image_array = np.array(image.resize((224, 224))) / 255.0
    result = model.predict(image_array[np.newaxis, ...])
    predicted_label_index = np.argmax(result)
    return predicted_label_index

# Streamlit app
st.title("Image Classification App")
st.sidebar.title("Choose an Image")

# Upload image through the sidebar
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg","png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Make prediction
    predicted_label_index = predict_image(image)

    # Display the result
    class_labels = ['HDPE', 'PET', 'PP', 'PS', 'PVC']
    st.write("Prediction Result:")
    st.write(f"Predicted Class: {class_labels[predicted_label_index]}")

# Run the app with: streamlit run your_script.py

