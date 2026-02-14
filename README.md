#  Smart Crop Health Analyzer

An AI-based web application that detects crop leaf diseases from an uploaded image and provides treatment recommendations.

The system currently supports **Potato** and **Tomato** crops and predicts disease type along with confidence and prevention steps.

## Problem

Farmers often struggle to identify plant diseases early.
Manual inspection requires expertise and delay leads to crop loss.

This project provides a quick and simple solution:
Upload a leaf image → Get disease diagnosis → View treatment suggestion.

## Features

* Upload crop leaf image
* Detect disease using trained deep learning models
* Confidence score visualization
* Treatment & prevention recommendations
* Image preview before prediction
* Clean agricultural themed UI

## Tech Stack

* Python
* Flask
* TensorFlow Lite
* HTML, CSS
* NumPy

## Project Structure

app/
 ├── models/                 # .tflite models
 ├── static/uploads/         # Uploaded images
 ├── templates/              # HTML pages
 └── app.py                  # Main application

## ▶️ How to Run

```bash
# Create environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run server
python app.py
```

Open browser:

http://127.0.0.1:5000

## Future Scope

* Support more crops
* Mobile deployment
* Offline prediction
* Multi-crop detection
* Farmer advisory integration

## Note

This project is a prototype and predictions depend on model training data quality.

```