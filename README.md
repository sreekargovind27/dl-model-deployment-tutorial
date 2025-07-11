# Full-Stack Deep Learning Model Deployment

A comprehensive, end-to-end tutorial on deploying multiple PyTorch models as a full-stack web application. This project uses a Python/Flask backend, a React/Vite frontend, and is deployed as a monolithic service on Render.

---

## Features

-   **Multi-Model Inference**: Serves three distinct PyTorch models (ResNet18, MobileNetV2, and a custom MNIST CNN).
-   **Interactive UI**: A clean user interface built with React allows users to select a model, upload an image, and view the prediction.
-   **Memory-Efficient Backend**: Implements a **lazy loading** pattern to load models on-demand, making it suitable for resource-constrained environments like Render's free tier.
-   **Monolithic Deployment**: A simple and effective deployment strategy where the Flask server is responsible for both the API and serving the static React frontend.
-   **Modern Tooling**: Utilizes `uv` for fast Python package management and `Vite` for a modern, efficient frontend build process.

## Technology Stack

-   **Backend**: Python, Flask, Gunicorn
-   **Machine Learning**: PyTorch, Torchvision
-   **Frontend**: React.js, Vite, Axios
-   **Deployment**: Render, Git
-   **Environment**: WSL (Ubuntu), `uv`

---

## Getting Started: Local Development

Follow these instructions to get a copy of the project up and running on your local machine for development and testing.

### Prerequisites

-   [Git](https://git-scm.com/)
-   [Python 3.9+](https://www.python.org/)
-   [Node.js (LTS)](https://nodejs.org/) (we recommend using `nvm`)
-   [uv](https://github.com/astral-sh/uv) (`pip install uv`)

### Installation and Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/sreekargovind27/dl-model-deployment-tutorial.git
    cd dl-model-deployment-tutorial
    ```

2.  **Set up the Python Environment:**
    ```bash
    # Create a virtual environment
    uv venv

    # Activate the environment
    source .venv/bin/activate

    # Install all Python dependencies from pyproject.toml
    uv pip install -e .
    ```

3.  **Train the Custom MNIST Model:**
    This is a one-time step to generate the `models/mnist_cnn.pth` file.
    ```bash
    python train_mnist.py
    ```

4.  **Set up the Frontend:**
    ```bash
    cd frontend
    npm install
    cd ..
    ```

### Running the Application Locally

You will need to run the backend and frontend servers in two separate terminals.

1.  **Run the Flask Backend Server (Terminal 1):**
    ```bash
    # Make sure your virtual environment is active
    source .venv/bin/activate
    python backend/app.py
    ```
    The backend will be running at `http://127.0.0.1:5001`.

2.  **Run the React Frontend Dev Server (Terminal 2):**
    ```bash
    cd frontend
    npm run dev
    ```
    The application will open automatically in your browser at `http://localhost:5173`. The Vite proxy will handle API requests.

---

## Deployment to Render

This project is configured for a simple, single-service deployment on Render's free tier.

### Step 1: Prepare for Production

Before deploying, you must create a production build of the frontend. This compiles all your React code into a static `dist` folder that Flask can serve.

1.  **Generate `requirements.txt`:**
    Render's build system uses this file. `pipreqs` generates it, but you must manually add `gunicorn`.
    ```bash
    pipreqs backend --savepath requirements.txt --force
    # Now, open requirements.txt and add a line for gunicorn==22.0.0
    ```

2.  **Build the React App:**
    ```bash
    cd frontend
    npm run build
    cd ..
    ```

3.  **Commit All Changes to Git:**
    This is the most important step. You must commit all your code, including the newly created `frontend/dist` folder.
    ```bash
    git add .
    git commit -m "build: create final production build for deployment"
    git push origin main
    ```

### Step 2: Configure Render

1.  On the Render dashboard, create a **New Web Service** and connect it to your GitHub repository.
2.  Configure the settings with the following **exact** values. These are crucial for the deployment to work correctly.

    -   **Root Directory:** `backend`
    -   **Build Command:** `pip install -r ../requirements.txt`
    -   **Start Command:** `gunicorn --bind 0.0.0.0:$PORT app:app`

3.  Select the **Free** instance type and click **Create Web Service**.
4.  Render will automatically deploy your application on every push to the `main` branch. After the first deployment is complete, your application will be live at the provided URL.

## Important Considerations & What to Look Out For

-   **Memory Management:** The backend uses a **lazy loading** pattern for models. This is essential for the free tier, which has limited RAM. The first prediction for each model will be slow as it loads into memory. Subsequent predictions will be fast.
-   **Relative Paths are Key:** The deployment setup relies heavily on correct relative paths (`../`). The `Root Directory` is set to `backend`, so the Build Command and the model path in the code must use `../` to access files in the project root.
-   **Frontend Build is Mandatory:** For this monolithic deployment to work, you **must** run `npm run build` and commit the resulting `frontend/dist` folder to Git before deploying. If your live site is blank, it's almost certainly because the `dist` folder is missing or out of date in your repository.
-   **Input Validation:** This project assumes valid image inputs. For a real-world application, you would need to add robust server-side validation to handle incorrect file types or corrupted images.
-   **MNIST Model Specificity:** The custom MNIST model was trained on a specific format (white digits on a black background). It will perform poorly on "real-world" images of numbers. To get a correct prediction, use an image editor to create a test image in the correct format.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
