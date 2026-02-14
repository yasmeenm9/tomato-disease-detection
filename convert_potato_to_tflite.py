import tensorflow as tf

model = tf.keras.models.load_model("potato_model.keras")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open("potato_model.tflite", "wb") as f:
    f.write(tflite_model)

print("Potato TFLite model saved successfully!")
