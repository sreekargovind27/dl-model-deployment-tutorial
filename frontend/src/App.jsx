// frontend/src/App.jsx

import { useState } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [selectedModel, setSelectedModel] = useState('resnet18');
  const [prediction, setPrediction] = useState('');
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (file) {
      setSelectedFile(file);
      setPreview(URL.createObjectURL(file));
      setPrediction('');
      setError('');
    }
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    if (!selectedFile) {
      setError('Please select a file first.');
      return;
    }

    setLoading(true);
    setPrediction('');
    setError('');

    const formData = new FormData();
    formData.append('file', selectedFile);
    formData.append('model', selectedModel);

    try {
      // The API endpoint is proxied by Vite during development
      const response = await axios.post('/api/predict', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      setPrediction(response.data.prediction);
    } catch (err) {
      setError('An error occurred during prediction. Please try again.');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>Deep Learning Model Deployment</h1>
        <p>Deploying PyTorch models with Flask, React & Heroku</p>
      </header>
      <main>
        <div className="controls">
          <div className="select-model">
            <label htmlFor="model-select">Choose a model:</label>
            <select
              id="model-select"
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
            >
              <option value="resnet18">ResNet18 (General)</option>
              <option value="mobilenet_v2">MobileNetV2 (Fast)</option>
              <option value="mnist_cnn">MNIST CNN (Digits 0-9)</option>
            </select>
          </div>
          <form onSubmit={handleSubmit} className="upload-form">
            <input type="file" onChange={handleFileChange} accept="image/*" />
            <button type="submit" disabled={!selectedFile || loading}>
              {loading ? 'Predicting...' : 'Get Prediction'}
            </button>
          </form>
        </div>

        {error && <p className="error">{error}</p>}

        <div className="results">
          {preview && (
            <div className="image-preview">
              <h3>Image Preview</h3>
              <img src={preview} alt="Selected" />
            </div>
          )}
          {prediction && (
            <div className="prediction-result">
              <h3>Prediction</h3>
              <p>{prediction}</p>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}

export default App;