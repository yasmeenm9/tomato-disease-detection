from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

#model = tf.keras.models.load_model("tomato_disease_model.keras")
MODEL_PATH = "tomato_disease_model.keras"

model = None
if os.path.exists(MODEL_PATH):
    model = tf.keras.models.load_model(MODEL_PATH)
else:
    print("⚠️ Model file not found. Running app without model.")

class_names = ['Early_blight', 'Healthy', 'Late_blight', 'Leaf Miner', 'Spotted Wilt Virus']

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    confidence = None
    filename = None

    if request.method == 'POST':
        file = request.files['file']
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            img = image.load_img(filepath, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0) / 255.0

            preds = model.predict(img_array)
            prediction = class_names[np.argmax(preds)]
            confidence = round(np.max(preds) * 100, 2)
            filename = file.filename

    return render_template(
        'index.html',
        prediction=prediction,
        confidence=confidence,
        filename=filename
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)