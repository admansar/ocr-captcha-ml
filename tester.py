from PIL import Image
import tensorflow as tf
import numpy as np
import json

def predict_captcha(image_path):
    # Load the model
    model = tf.keras.models.load_model('final_model.h5')
    
    # Load label mapping
    with open('label_mapping.json', 'r') as f:
        label_mapping = json.load(f)
    reverse_mapping = {v: k for k, v in label_mapping.items()}
    
    # Preprocess the image
    img = Image.open(image_path).convert('L')  # Convert to grayscale
    img = img.resize((200, 50))  # Use the same size as training
    img_array = np.array(img)
    img_array = img_array / 255.0  # Normalize
    img_array = img_array.reshape(1, 50, 200, 1)
    
    # Make prediction
    prediction = model.predict(img_array, verbose=0)
    predicted_class = np.argmax(prediction[0])
    confidence = prediction[0][predicted_class]
    
    # Get predicted text
    predicted_text = reverse_mapping[predicted_class]
    
    return predicted_text, confidence

# Test an image
import os
image_path = "./training/"  # Replace with your image path

for image in os.listdir(image_path):

    predicted_text, confidence = predict_captcha(image_path + image)
    print(f"Predicted text: {predicted_text}")
    print(f"Confidence: {confidence:.2%}")
