# Rice Disease Prediction

This project aims to detect various diseases in rice plants using image classification. The model is trained using logistic regression and can predict diseases from new images.

## Table of Contents
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Setup](#setup)
- [Usage](#usage)
- [Functions](#functions)
- [License](#license)

## Project Structure


## Requirements

- Python 3.x
- `numpy`
- `scikit-learn`
- `scikit-image`
- `Pillow`
- `matplotlib`
- `streamlit`
- `joblib`

## Setup

1. **Clone the repository:**
    ```sh
    git clone https://github.com/jai72/Rice-Disease-Prediction.git
    ```

2. **Install dependencies:**
    ```sh
    pip install -r requirements.txt
    ```

3. **Prepare dataset:**
    - Ensure your dataset is organized under the `data/` directory as shown in the project structure.

4. **Train the model:**
    ```sh
    python train_model.py
    ```
    This script will train the logistic regression model on the rice disease dataset and save the trained model (`rice_disease_model.pkl`) and target names (`target_names.npy`) in the project directory.

5. **Run the Streamlit app:**
    ```sh
    streamlit run app.py
    ```
    Navigate to the provided local URL (usually `http://localhost:8501`) to use the web interface for predicting diseases from uploaded images.

## Usage

1. **Train the Model:**
   - Ensure your dataset is structured under `data/` with subdirectories for each disease category.
   - Run `train_model.py` to train the logistic regression model.

2. **Predict Disease:**
   - Use `predict_disease.py` to predict diseases on new images.
   - Example usage:
     ```sh
     python predict_disease.py
     ```

3. **Streamlit Web App:**
   - Run `app.py` to launch the Streamlit web application.
   - Upload an image through the web interface to get the disease prediction.

## Functions

### `train_model.py`

- **Training Script (`train_model.py`):** Loads the dataset, trains the logistic regression model, and saves the trained model and target names.

### `predict_disease.py`

- **Prediction Script (`predict_disease.py`):** Loads a new image and predicts the disease using the trained model (`rice_disease_model.pkl`).

### `app.py`

- **Streamlit Web Application (`app.py`):** Provides a user-friendly interface for predicting diseases from uploaded images.

