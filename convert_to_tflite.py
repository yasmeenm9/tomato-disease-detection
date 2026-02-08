import tensorflow as tf

MODEL_PATH = "tomato_disease_model.keras"
TFLITE_PATH = "tomato_disease_model.tflite"

model = tf.keras.models.load_model(MODEL_PATH)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()

with open(TFLITE_PATH, "wb") as f:
    f.write(tflite_model)

print("✅ TFLite model saved:", TFLITE_PATH)
