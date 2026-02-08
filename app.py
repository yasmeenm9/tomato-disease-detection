from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

MODEL_PATH = "tomato_disease_model.keras"

print("🔁 Loading model at startup...")
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded successfully")

class_names = [
    'Early_blight',
    'Healthy',
    'Late_blight',
    'Leaf Miner',
    'Spotted Wilt Virus'
]

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    confidence = None
    filename = None

    if request.method == 'POST':
        file = request.files.get('file')
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            img = image.load_img(filepath, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0) / 255.0

            preds = model.predict(img_array)

            prediction = class_names[np.argmax(preds)]
            confidence = round(float(np.max(preds)) * 100, 2)
            filename = file.filename

    return render_template(
        'index.html',
        prediction=prediction,
        confidence=confidence,
        filename=filename
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
