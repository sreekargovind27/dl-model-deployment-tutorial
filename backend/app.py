# backend/app.py

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
from model_handler import get_prediction

# --- 1. Create Flask App ---
app = Flask(__name__, static_folder='../frontend/build', static_url_path='/')
CORS(app) # Enable CORS for all routes

# --- 2. API Endpoint for Prediction ---
@app.route('/api/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'no file provided'}), 400
    
    file = request.files['file']
    model_name = request.form.get('model', 'resnet18') # Default to resnet18 if not specified

    if not file:
        return jsonify({'error': 'no file selected'}), 400

    try:
        img_bytes = file.read()
        prediction = get_prediction(model_name=model_name, image_bytes=img_bytes)
        return jsonify({'prediction': prediction})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# --- 3. Serve React App ---
# This is for production. It serves the built React app.
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    if path != "" and os.path.exists(app.static_folder + '/' + path):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, 'index.html')

# --- 4. Main entry point ---
if __name__ == '__main__':
    # Use a production-ready server like Gunicorn in production
    # The following is for development only
    app.run(debug=True, port=5001)