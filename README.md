# ECG ARRHYTHMIA PREDICTOR PROJECT

An end-to-end deep learning project for detecting heart arrhythmias from ECG signals using the **PTB-XL dataset**.  
This project applies **Inception-style 1D convolutional blocks** and **Bidirectional LSTMs** to capture both spatial and temporal cardiac features.

---

## 📚 Overview

This model aims to classify 12-lead ECG signals into three diagnostic classes:
- **Normal**
- **Myocardial Infarction (MI)**
- **Bundle Branch Block (BBB)**

Developed as part of my **12th grade research project**, this system demonstrates how medical signal processing and machine learning can be combined for clinical insight and early arrhythmia detection.

---

## ⚙️ Workflow Summary

1. **Dataset**  
   - Uses [PTB-XL](https://physionet.org/content/ptb-xl/1.0.3/) ECG dataset.  
   - Automatically downloaded into a local `ptbxl/` folder (ignored from Git tracking).

2. **Preprocessing**
   - Signal reading with `wfdb`.
   - **No data leakage** — normalization occurs *after* train/test split.
   - Balancing strategy through oversampling minority classes.

3. **Model Architecture**
   - Inception 1D convolution blocks for multi-scale pattern extraction.
   - Bidirectional LSTM for temporal feature modeling.
   - Attention pooling for key heartbeat emphasis.
   - Focal loss to manage class imbalance.

4. **Evaluation**
   - Metrics: Accuracy, F1-score, and confusion matrix.
   - Typical performance: >85% validation accuracy (varies with hyperparameters).

---

## 🚀 Getting Started

### Requirements

To install all dependencies, run:

    pip install -r requirements.txt

---

## 📜 License

MIT License © 2025 Nathanael Wilson Bong

---

## ✨ Acknowledgements

Dataset by [PhysioNet PTB-XL](https://physionet.org/content/ptb-xl/1.0.3/)  
Developed using TensorFlow, NumPy, and WFDB.

---

## 🩺 Author

**Nathanael Wilson Bong**  
12th Grade Research Project • Canisius College, Jakarta