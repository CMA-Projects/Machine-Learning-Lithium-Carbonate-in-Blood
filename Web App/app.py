from flask import Flask, render_template, request, jsonify
import numpy as np
import tensorflow as tf
import joblib

app = Flask(__name__)

# Loads the homepage whicch is the index.html in this case
@app.route('/')
def home():
    return render_template('index.html')

# Try to load the model and troubleshoot if model fails to load
try:
    print("Loading Model...")
    path_to_model = "ml_models\DT_4inputs_1output.h5"
    model = joblib.load(path_to_model)
    print("Loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")

# Route to predict and its functionality
@app.route('/predict', methods=['POST'])
def predict():
    # Get the input from the user
    data = request.json

    # Extract the input values
    input_values = data.get('inputs')

    # Check if the input values have exactly 4 elements
    if not input_values or len(input_values) != 4:
        return jsonify({'error': 'Input must be a list of 4 numerical values!'})



if __name__ == '__main__':
    app.run(debug=True)