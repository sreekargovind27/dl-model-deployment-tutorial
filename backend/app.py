# backend/app.py

from flask import Flask, request, jsonify
from flask_cors import CORS
from model_handler import get_prediction

# Create Flask App 

app = Flask(__name__)

# Enable CORS 

# make requests to this backend.
CORS(app)

# API Endpoint for Prediction 
@app.route('/api/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'no file provided'}), 400
    
    file = request.files['file']
    model_name = request.form.get('model', 'resnet18')

    if not file:
        return jsonify({'error': 'no file selected'}), 400

    try:
        img_bytes = file.read()
        prediction = get_prediction(model_name=model_name, image_bytes=img_bytes)
        return jsonify({'prediction': prediction})
    except Exception as e:

        print(f"An error occurred: {e}")
        return jsonify({'error': 'Error making prediction'}), 500

# Main entry point 
if __name__ == '__main__':
    app.run(debug=True, port=5001)