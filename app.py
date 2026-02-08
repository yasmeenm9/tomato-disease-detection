from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
import os
from tensorflow.keras.preprocessing import image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Load TFLite model ONCE
INTERPRETER = tf.lite.Interpreter(model_path="tomato_disease_model.tflite")
INTERPRETER.allocate_tensors()

input_details = INTERPRETER.get_input_details()
output_details = INTERPRETER.get_output_details()

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
        file = request.files['file']
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            img = image.load_img(filepath, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0).astype(np.float32) / 255.0

            INTERPRETER.set_tensor(input_details[0]['index'], img_array)
            INTERPRETER.invoke()

            preds = INTERPRETER.get_tensor(output_details[0]['index'])

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
