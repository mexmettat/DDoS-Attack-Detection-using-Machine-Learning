# 🚀 Installation & Execution Guide

This document provides step-by-step instructions to set up the environment and run the DDoS Detection System.

## 1. Environment Setup

### Prerequisites
- Python 3.9+ installed.
- (Optional but recommended) A virtual environment.

### Install Dependencies
Run the following command to install all required libraries (TensorFlow, XGBoost, Streamlit, Plotly, etc.):
```bash
pip install -r requirements.txt
```

## 2. Directory Structure
Ensure your project folder looks like this:
```text
DDos_attack/
├── data/processed/     # Place cleaned .csv files here
├── metrics/           # Performance JSONs and PNGs
├── models/            # Trained .h5, .pkl and .pkl scalers
├── src/
│   ├── train_ml.py    # Traditional ML training script
│   ├── train_cnn.py   # Deep Learning training script
│   └── app_streamlit.py # Interactive GUI
└── ddos.png           # Dashboard Sidebar Icon
```

## 3. Running the Models

### Training ML Models (XGBoost & Random Forest)
This script handles the hybrid data mixing and trains the tree-based models:
```bash
python src/train_ml.py
```

### Training Deep Learning (1D-CNN)
This script applies MinMaxScaler and trains the convolutional architecture:
```bash
python src/train_cnn.py
```

## 4. Launching the Interactive GUI (Sunum İçin)
To show the project to the professor or for testing, use the Streamlit dashboard:
```bash
streamlit run src/app_streamlit.py
```

## 5. Key File Descriptions
- **`models/scaler.pkl`**: Scaler used for ML models.
- **`models/scaler_cnn.pkl`**: Scaler used for the CNN model.
- **`models/train_columns.pkl`**: The exact feature list needed for inference (ensures column alignment).
- **`metrics/`**: Every time you train, this folder updates with the latest Accuracy/Recall scores and Confusion Matrices.

## ⚠️ Important Notes for Group Members
- **Data Alignment:** If you use a new dataset, the system will automatically try to map the 2019 column names to the 2017 schema.
- **Leakage Prevention:** Do not modify the `train_test_split` logic in Step 1 of the scripts; it is crucial for ensuring the 2019 test data remains "unseen".
- **Hardware:** CNN training is optimized for CPU on Windows. If you have a GPU, ensure you are running via WSL2 for hardware acceleration.
