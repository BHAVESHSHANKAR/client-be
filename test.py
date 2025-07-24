import tensorflow as tf
import numpy as np
import os

# Path to your model
model_path = 'D:/clients/ml-be/backend/trainedmodels/brain_model.keras'

# Verify environment
print("File exists:", os.path.exists(model_path))
print("File size:", os.path.getsize(model_path), "bytes")
print("TensorFlow version:", tf.__version__)
print("Keras version:", tf.keras.__version__)
print("NumPy version:", np.__version__)

# Load the model
try:
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully")
    model.summary()
except Exception as e:
    print("Error loading model:", e)