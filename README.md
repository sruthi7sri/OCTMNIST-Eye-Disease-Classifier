# OCTMNIST Eye Disease Classifier – Deep Learning

## Overview
I have developed three independent deep learning applications using PyTorch, Streamlit, and classic ML libraries as part of my academic coursework for Deep Learning. Each part demonstrates practical deployment-ready models and interactive dashboards, including one for real-world medical image classification using the OCTMNIST dataset.

---
## Live Demo (Deployed App)

| Part | Description                             | Status     | Link |
|------|-----------------------------------------|------------|------|
| Part 1 | PyTorch Tutorial Demo (Notebook)         | Notebook only | – |
| Part 2 | ML on Tabular Data (Notebook)            | Notebook only | – |
| Part 3 | OCTMNIST CNN Classifier (Deployed App)   | ✅ Live      | [Launch App](https://octmnist-classifier-fhxddfydazvlesy9ycqnzg.streamlit.app/) |

---
## Methodology

### Part 1: PyTorch Tutorial Demo
- I implemented core PyTorch functionalities such as tensor operations, autograd, model definition, and training loops.
- This is delivered as a clean Jupyter notebook structured for step-by-step conceptual learning.
- Sections include tensors, backpropagation, training a small NN, and a simulated TensorBoard walkthrough.

### Part 2: Tabular Data ML Explorer
- I created a complete ML pipeline in a notebook, from EDA to modeling, using real-world tabular data (e.g., Iris dataset).
- The notebook includes data visualization, feature engineering, and model comparison (Random Forest, Logistic Regression, SVM).
- All models achieved accuracy > 65%, and the code is modular for easy extension.

### Part 3: OCTMNIST CNN Classifier
- I trained a Convolutional Neural Network (CNN) on the OCTMNIST dataset using PyTorch.
- I deployed the model as a full web app using Streamlit, allowing users to upload retina images for disease classification.
- The app displays predictions, class probabilities, and model insights — making it both educational and practical.

---

## Real-World Applications
- **Medical Diagnostics**: Automated OCT-based classification for early retinal disease detection.
- **ML Education**: Interactive tools for learning PyTorch and ML model behaviors.
- **AI-Powered Dashboards**: Turn any ML or DL model into an accessible web tool.

---

## Technology Comparison

| Component        | Chosen Tech           | Alternatives              | Why This Was Chosen |
|------------------|------------------------|----------------------------|----------------------|
| DL Framework     | PyTorch                | TensorFlow, Keras          | More control, better for academic setup |
| Web UI           | Streamlit              | Flask, Gradio              | Fastest to deploy with sliders, graphs |
| Visualization    | Matplotlib, Seaborn    | Plotly, Dash               | Easy integration and fast rendering |
| Tabular ML       | Scikit-learn           | XGBoost, LightGBM          | Simple models work well for small data |

---

## File Structure

## File Structure
```bash
OCTMNIST-Classifier/
├── analysis_notebooks/            # Part 1 & 2 notebooks
│   ├── 01_data_preprocessing.ipynb
│   └── 02_model_training.ipynb
│
├── notebooks/                     # Part 3 evaluation and bonus
│   ├── 03_model_evaluation.ipynb
│   └── 04_bonus_deep_learning.ipynb
│
├── resources/                     # Model weights and links
│   └── pretrained_weights.txt
│
├── app.py                         # Streamlit app entry point
├── best_model_final.pth           # CNN model weights
├── streamlit_octmnist.py          # backup app script
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation & Usage
### Clone the Repo
```bash
git clone https://github.com/sruthi7sri/OCTMNIST_Eye_Disease_Classifier.git
cd OCTMNIST_Eye_Disease_Classifier
```
### Install Dependencies
```bash
pip install -r requirements.txt
```

### Run Deployed App Locally
```bash
# Part 3
streamlit run deployment_part3_OCTMNIST_Classifier/app.py
```

## Project Goals
This project demonstrates a full-stack ML/DL workflow through three independent components:

- **Part 1: PyTorch Fundamentals** – Focused on tensors, autograd, model training, and educational walk-throughs in PyTorch.
- **Part 2: Tabular ML Modeling** – Explored a real-world dataset with preprocessing, ML pipelines, and model evaluation.
- **Part 3: Image Classification App (Deployed)** – Delivered a deployed deep learning application using CNN for retinal OCT images, built with PyTorch and Streamlit.

## License
© 2025 Sruthisri Venkateswaran. All rights reserved.
