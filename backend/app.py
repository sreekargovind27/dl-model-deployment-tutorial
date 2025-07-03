from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
from model_handler import get_prediction

app = Flask(__name__, static_folder='../frontend/build', static_url_path='/')
CORS(app)

@app.route('/api/predict', methods=['POST'])
def predict():
    if 'file' not in request.files: return jsonify({'error': 'no file provided'}), 400
    file, model_name = request.files['file'], request.form.get('model', 'resnet18')
    if not file: return jsonify({'error': 'no file selected'}), 400
    try:
        prediction = get_prediction(model_name=model_name, image_bytes=file.read())
        return jsonify({'prediction': prediction})
    except Exception as e:
        print(f"An error occurred: {e}")
        return jsonify({'error': 'Error making prediction'}), 500

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    if path != "" and os.path.exists(app.static_folder + '/' + path):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, 'index.html')

if __name__ == '__main__': app.run(debug=True, port=5001)