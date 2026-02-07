import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# Load model
model = tf.keras.models.load_model("tomato_disease_model.h5")

# Class names (same order as training)
class_names = ['Early_blight', 'Healthy', 'Late_blight', 'Leaf Miner', 'Spotted Wilt Virus']

# Image path (change filename only)
img_path = "sample.jpg"

img_height = 224
img_width = 224

# Load and preprocess image
img = image.load_img(img_path, target_size=(img_height, img_width))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0

# Predict
predictions = model.predict(img_array)
predicted_class = class_names[np.argmax(predictions)]
confidence = np.max(predictions) * 100

print(f"Prediction: {predicted_class}")
print(f"Confidence: {confidence:.2f}%")
