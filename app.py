from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
import os
from tensorflow.keras.preprocessing import image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# ================== LOAD TOMATO MODEL ==================
tomato_interpreter = tf.lite.Interpreter(
    model_path="models/tomato_disease_model.tflite"
)
tomato_interpreter.allocate_tensors()

tomato_input = tomato_interpreter.get_input_details()
tomato_output = tomato_interpreter.get_output_details()

tomato_classes = [
    'Early_blight',
    'Healthy',
    'Late_blight',
    'Leaf Miner',
    'Spotted Wilt Virus'
]

tomato_solutions = {
    "Early_blight": {
        "solution": "Use Mancozeb fungicide. Remove infected leaves.",
        "prevention": "Avoid overhead watering."
    },
    "Late_blight": {
        "solution": "Apply Copper-based fungicides.",
        "prevention": "Ensure proper drainage."
    },
    "Leaf Miner": {
        "solution": "Spray Neem oil.",
        "prevention": "Monitor leaves regularly."
    },
    "Spotted Wilt Virus": {
        "solution": "Remove infected plants.",
        "prevention": "Control thrips."
    },
    "Healthy": {
        "solution": "No disease detected.",
        "prevention": "Maintain good crop care."
    }
}

# ================== LOAD POTATO MODEL ==================
potato_interpreter = tf.lite.Interpreter(
    model_path="models/potato_disease_model.tflite"
)
potato_interpreter.allocate_tensors()

potato_input = potato_interpreter.get_input_details()
potato_output = potato_interpreter.get_output_details()

potato_classes = [
    'Potato_Early_blight',
    'Potato_Late_blight',
    'Potato_Healthy'
]

potato_solutions = {
    "Potato_Early_blight": {
        "solution": "Apply Mancozeb or Chlorothalonil.",
        "prevention": "Avoid overhead irrigation."
    },
    "Potato_Late_blight": {
        "solution": "Use Copper fungicides.",
        "prevention": "Improve air circulation."
    },
    "Potato_Healthy": {
        "solution": "No disease detected.",
        "prevention": "Maintain balanced fertilization."
    }
}

# ================== ROUTE ==================
@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    confidence = None
    filename = None
    solution = None
    prevention = None

    if request.method == 'POST':
        file = request.files.get('file')
        crop = request.form.get('crop')  # must match dropdown name

        if file and file.filename != "":
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            # Choose model input size dynamically
            if crop == "potato":
                size = potato_input[0]['shape'][1]
            else:
                size = tomato_input[0]['shape'][1]

            img = image.load_img(filepath, target_size=(size, size))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0).astype(np.float32) / 255.0

            if crop == "potato":
                potato_interpreter.set_tensor(potato_input[0]['index'], img_array)
                potato_interpreter.invoke()
                preds = potato_interpreter.get_tensor(potato_output[0]['index'])

                prediction = potato_classes[np.argmax(preds)]
                solution_data = potato_solutions.get(prediction)

            else:
                tomato_interpreter.set_tensor(tomato_input[0]['index'], img_array)
                tomato_interpreter.invoke()
                preds = tomato_interpreter.get_tensor(tomato_output[0]['index'])

                prediction = tomato_classes[np.argmax(preds)]
                solution_data = tomato_solutions.get(prediction)

            confidence = round(float(np.max(preds)) * 100, 2)
            filename=file.filename
            if confidence < 80:
               prediction = "Unrecognized Leaf"
               solution = "Please upload a clear image of a valid tomato or potato leaf."
               prevention = "Ensure good lighting and proper crop selection."
            else:
                if solution_data:
                    solution=solution_data.get("solution")
                    prevention=solution_data.get("prevention")
    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        filename=filename,
        solution=solution,
        prevention=prevention
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
