# Retinal OCT Image Classifier for Eye Diseases

This project presents a complete, end-to-end deep learning pipeline for the multi-class classification of retinal eye diseases using **Optical Coherence Tomography (OCT)** images. The solution encompasses the entire machine learning lifecycle: from data exploration and preprocessing of the **OCTMNIST dataset**, to building and training a custom **Convolutional Neural Network (CNN)** in PyTorch, and finally, deploying the model in a real-time, interactive **Streamlit** web application. 👁️‍🗨️

---
## 📜 Core Features

* **Dataset**: Leverages the well-documented **OCTMNIST dataset**, a large collection of grayscale retinal OCT images, to diagnose four conditions.
* **Advanced Preprocessing**: Implements critical data preprocessing techniques, including normalization and a robust strategy to mitigate severe class imbalance using the **SMOTE (Synthetic Minority Over-sampling Technique)**.
* **Custom CNN Architecture**: A lightweight yet powerful **Convolutional Neural Network** is built from scratch using PyTorch, specifically designed for high-accuracy classification on this dataset.
* **Rigorous Evaluation**: The model's predictive power is thoroughly validated on a held-out test set using a suite of metrics, including Accuracy, Precision, Recall, F1-score, and a detailed Confusion Matrix.
* **Interactive Deployment**: The final trained model is serialized and served via a user-friendly web interface built with **Streamlit**, allowing for on-the-fly predictions from user-uploaded images.

---
## 🔬 Problem Statement & Context

Diabetic Retinopathy and other retinal pathologies are among the leading causes of preventable blindness worldwide. Optical Coherence Tomography (OCT) is a non-invasive imaging technique that provides high-resolution, cross-sectional images of the retina, essential for diagnosis. Automating the analysis of these scans can lead to faster, more accessible, and more consistent diagnoses, particularly in underserved areas. This project tackles this challenge by developing a deep learning model to accurately classify OCT scans into four clinically relevant categories: **Normal**, **Choroidal Neovascularization (CNV)**, **Diabetic Macular Edema (DME)**, and **Drusen**.

---
## 🧠 Technical Deep Dive

### 1. Data Preprocessing

The raw OCTMNIST images are first converted into PyTorch tensors and normalized using the mean and standard deviation calculated from the training set. A key challenge identified was the **severe class imbalance** in the dataset. To prevent the model from becoming biased towards the over-represented classes, **SMOTE (Synthetic Minority Over-sampling Technique)** was applied. This technique generates new, synthetic samples for the minority classes by interpolating between existing data points, creating a balanced and more representative training set.

### 2. Model Architecture: `SimpleCNN`

A custom Convolutional Neural Network was designed in PyTorch to effectively learn the hierarchical features from the OCT images. The architecture is structured as follows:

* **Convolutional Block 1**:
    * `Conv2d` layer with 32 filters, a kernel size of 3x3, and stride of 1.
    * `ReLU` activation function.
    * `MaxPool2d` layer with a 2x2 kernel to downsample feature maps.
* **Convolutional Block 2**:
    * `Conv2d` layer with 64 filters and a 3x3 kernel.
    * `ReLU` activation function.
    * `MaxPool2d` layer with a 2x2 kernel.
* **Flatten Layer**: Flattens the 2D feature maps into a 1D vector.
* **Fully Connected (Linear) Layer**: A dense layer that maps the features to the 4 output classes.

This multi-layered approach allows the model to learn simple features like edges in the first block and more complex patterns in the second, leading to robust classification.

### 3. Training & Hyperparameters

The model was trained using the following configuration:

* **Optimizer**: Adam
* **Loss Function**: Cross-Entropy Loss
* **Learning Rate**: 0.001
* **Number of Epochs**: 10
* **Batch Size**: 64

The training process, including the forward and backward passes and model saving, is fully documented in the `03_model_evaluation.ipynb` notebook.

---
## 📊 Performance & Results

The model's ability to generalize to new, unseen data was validated on the test set, yielding impressive results:

* **Overall Test Accuracy**: **91.52%**
* **Weighted Average F1-Score**: **0.92**

The detailed classification report and confusion matrix in the evaluation notebook confirm that the model performs well across all four classes, validating the effectiveness of using SMOTE to handle the initial data imbalance.

---
## 🚀 Deployment with Streamlit

The project culminates in a fully functional web application that brings the model to life. The `streamlit_octmnist.py` script loads the trained PyTorch model (`best_model_final.pth`) and provides a simple interface where users can upload an OCT image.

The application's back-end performs the necessary image transformations (resizing to 28x28, converting to grayscale, normalizing) to match the model's input requirements before making a prediction. This demonstrates a key aspect of deploying ML models: ensuring consistency between training and inference data pipelines.

---
## 📁 File Structure
```bash
OCTMNIST Eye Disease Classifier/
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

---
## 🛠️ Installation & Usage

### 1. Prerequisites
* Python 3.8+
* Pip

### 2. Setup
1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/sruthi7sri/octmnist-eye-disease-classifier.git](https://github.com/sruthi7sri/octmnist-eye-disease-classifier.git)
    cd octmnist-eye-disease-classifier
    ```
2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```
3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### 3. Running the Project
* **To run the model training and evaluation notebook:**
    ```bash
    jupyter notebook notebooks/03_model_evaluation.ipynb
    ```
* **To launch the interactive web application:**
    ```bash
    streamlit run streamlit_octmnist.py
    ```

---
## 📚 References

1.  Yang, J., Shi, R., Wei, D., Liu, Z., Zhao, L., Ke, B., Pfister, H., & Ni, B. (2022). MedMNIST v2: A Large-Scale Lightweight Benchmark for 2D and 3D Biomedical Image Classification. *Nature Scientific Data*.
2.  Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: synthetic minority over-sampling technique. *Journal of artificial intelligence research*, 16, 321-357.
3.  O'Shea, K., & Nash, R. (2015). An Introduction to Convolutional Neural Networks. *arXiv preprint arXiv:1511.08458*.

## License
© 2025 Sruthisri Venkateswaran. All rights reserved.
