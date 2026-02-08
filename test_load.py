import os
import tensorflow as tf

print("Exists:", os.path.exists("tomato_disease_model.keras"))

if os.path.exists("tomato_disease_model.keras"):
    print("Size:", os.path.getsize("tomato_disease_model.keras"))

print("Trying to load model...")
model = tf.keras.models.load_model("tomato_disease_model.keras")
print("MODEL LOADED SUCCESSFULLY")