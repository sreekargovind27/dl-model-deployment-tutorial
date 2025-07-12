# Deep Learning Model Deployment Tutorial

A full-stack application demonstrating how to deploy multiple PyTorch models (MNIST CNN, ResNet18, MobileNetV2) using a Flask backend and a React frontend, deployed on Render.

### [➡️ Live Demo ⬅️](https://dl-model-deployment-tutorial.onrender.com)

## Overview

This project serves as a comprehensive, hands-on tutorial for deploying deep learning models as a web service. It covers the entire workflow from a trained model to a publicly accessible web application. The backend is built with Python and Flask, serving a REST API for predictions. The frontend is a modern, responsive interface built with React and Vite.

## Features

-   **Multiple Model Support:** Choose between three different PyTorch models for inference.
-   **Image Upload:** Users can upload their own images for classification.
-   **Real-time Predictions:** Get instant predictions from the selected deep learning model.
-   **Image Preview:** Displays the uploaded image alongside the model's prediction.
-   **Responsive Design:** A clean and modern UI that works on both desktop and mobile.
-   **Production-Ready:** Includes a production-grade Gunicorn server and a complete deployment guide for Render.

## Tech Stack

-   **Frontend:** React, Vite, CSS
-   **Backend:** Python, Flask, PyTorch, Gunicorn
-   **Deployment:** Render

## Local Development Setup

To run this project on your local machine, follow these steps.

### Prerequisites

-   [Git](https://git-scm.com/)
-   [Python](https://www.python.org/downloads/) (3.8 or higher)
-   [Node.js](https://nodejs.org/en/) and npm

### Installation & Running

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/sreekargovind27/dl-model-deployment-tutorial.git
    cd dl-model-deployment-tutorial
    ```

2.  **Setup the Backend:**
    ```bash
    # Navigate to the backend directory
    cd backend

    # Create and activate a virtual environment
    python -m venv venv
    # On Windows:
    # venv\Scripts\activate
    # On macOS/Linux:
    # source venv/bin/activate

    # Install Python dependencies
    pip install -r requirements.txt
    ```

3.  **Setup the Frontend:**
    ```bash
    # Navigate to the frontend directory from the root
    cd ../frontend

    # Install Node.js dependencies
    npm install
    ```

4.  **Run the Application:**
    You will need two separate terminals to run both the backend and frontend servers.

    -   **Terminal 1: Start the Backend (Flask) Server**
        ```bash
        cd backend
        # (Make sure your virtual environment is activated)
        flask run --port 5001
        ```
        The backend will be running at `http://127.0.0.1:5001`.

    -   **Terminal 2: Start the Frontend (React) Server**
        ```bash
        cd frontend
        npm run dev
        ```
        The frontend development server will be running at `http://localhost:5173` (or another port if 5173 is busy). Open this URL in your browser.

## Deployment to Render

This project is configured for easy deployment on Render.

1.  **Create a new "Web Service"** on your Render dashboard and connect your GitHub repository.

2.  **Configure the settings:**
    -   **Root Directory:** Leave this **blank**. The `build.sh` script handles directory changes.
    -   **Build Command:** `bash ./build.sh`
    -   **Start Command:** `gunicorn backend.app:app`

3.  **Deploy!** Render will use the `build.sh` script to install dependencies for both the backend and frontend, build the static React files, and then use Gunicorn to start the Flask server.

## Project Structure
.

├── backend/

│ ├── data/ #mnist-dataset

│ ├── data_imagenet/ #imagenet-lables

│ ├── models/ #saved .pth file

│ ├── app.py #flask application

│ └── model_handler.py #model backend

│

├── frontend/

│ ├── public/

│ └── src/ #react components and source code

│

├── .gitignore

├── build.sh #a script for render

├── README.md

└── requirements.txt

