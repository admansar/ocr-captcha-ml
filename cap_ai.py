import tensorflow as tf
from tensorflow.keras import layers, models, Input
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import logging

# Disable TensorFlow logging except for errors
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel(logging.ERROR)

# Disable GPU usage completely
tf.config.set_visible_devices([], 'GPU')

def create_model(input_shape, num_classes):
    """Create a CNN model for text recognition"""
    inputs = Input(shape=input_shape)
    x = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    return model

def load_and_preprocess_data(data_dir, img_height, img_width):
    """Load and preprocess images from directory"""
    images = []
    labels = []
    
    for filename in os.listdir(data_dir):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            # Load image
            img_path = os.path.join(data_dir, filename)
            img = Image.open(img_path).convert('L')  # Convert to grayscale
            img = img.resize((img_width, img_height))
            img_array = np.array(img)
            img_array = img_array / 255.0  # Normalize
            
            # Get label from filename
            label = os.path.splitext(filename)[0]
            
            images.append(img_array)
            labels.append(label)
    
    return np.array(images), np.array(labels)

def train_model(data_dir, img_height=50, img_width=200, batch_size=32, epochs=50):
    """Train the OCR model"""
    print("Loading and preprocessing data...")
    X, y = load_and_preprocess_data(data_dir, img_height, img_width)
    
    # Convert labels to numerical format
    label_mapping = {label: idx for idx, label in enumerate(np.unique(y))}
    y_encoded = np.array([label_mapping[label] for label in y])
    num_classes = len(label_mapping)
    
    print(f"Found {len(X)} images with {num_classes} unique classes")
    
    # Split data
    indices = np.random.permutation(len(X))
    split = int(0.8 * len(X))
    train_idx, val_idx = indices[:split], indices[split:]
    
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y_encoded[train_idx], y_encoded[val_idx]
    
    # Reshape images for CNN
    X_train = X_train.reshape(X_train.shape[0], img_height, img_width, 1)
    X_val = X_val.reshape(X_val.shape[0], img_height, img_width, 1)
    
    print("Creating and compiling model...")
    model = create_model((img_height, img_width, 1), num_classes)
    model.compile(optimizer='adam',
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])
    
    # Create checkpoint directory and callbacks
    checkpoint_dir = "model_checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Save best model
    best_model_path = os.path.join(checkpoint_dir, 'best_model.weights.h5')
    best_model_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=best_model_path,
        save_weights_only=True,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True,
        verbose=1
    )
    
    # Save periodic checkpoints
    checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint_epoch_{epoch:04d}.weights.h5')
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=True,
        save_freq=5 * batch_size,  # Save every 5 epochs
        verbose=1
    )
    
    # Early stopping to prevent overfitting
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=10,
        restore_best_weights=True
    )
    
    print("Starting training...")
    history = model.fit(
        X_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_val, y_val),
        callbacks=[best_model_callback, checkpoint_callback, early_stopping]
    )
    
    # Save the final model and label mapping
    print("Saving final model and label mapping...")
    model.save('final_model.h5')
    
    import json
    with open('label_mapping.json', 'w') as f:
        json.dump(label_mapping, f)
    
    return model, history, label_mapping

def predict_text(model, image_path, label_mapping, img_height=50, img_width=200):
    """Predict text in a new image"""
    img = Image.open(image_path).convert('L')
    img = img.resize((img_width, img_height))
    img_array = np.array(img)
    img_array = img_array / 255.0
    img_array = img_array.reshape(1, img_height, img_width, 1)
    
    prediction = model.predict(img_array, verbose=0)
    predicted_class = np.argmax(prediction[0])
    
    reverse_mapping = {v: k for k, v in label_mapping.items()}
    predicted_text = reverse_mapping[predicted_class]
    
    return predicted_text, prediction[0][predicted_class]

if __name__ == "__main__":
    try:
        print("Starting OCR model training...")
        data_dir = "./captcha_images"
        
        model, history, label_mapping = train_model(data_dir)
        
        # Plot training history
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['accuracy'], label='accuracy')
        plt.plot(history.history['val_accuracy'], label='val_accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend(loc='lower right')
        plt.savefig('training_history.png')
        plt.close()
        
        print("\nTraining completed successfully!")
        print(f"Number of classes: {len(label_mapping)}")
        print("Model and checkpoints saved in current directory")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise
    # finally:
    #     test_image_path = "./captcha_images/iyt1zz.png"
    #     predicted_text, confidence = predict_text(model, test_image_path, label_mapping)
    #     print(f"Predicted text: {predicted_text}")
    #     print(f"Confidence: {confidence:.2f}")
