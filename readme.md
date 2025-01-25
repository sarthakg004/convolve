# Project: Fraud Detection Pipeline

## Overview
This project implements a data pipeline for fraud detection using machine learning. The pipeline includes data preprocessing, feature engineering, and model training and evaluation. The project tracks experiments using MLflow and supports hyperparameter optimization for XGBoost and Multi-Layer Perceptron (MLP) models.

---

## Project Structure

```plaintext
.
├── data_preprocessing.py        # Handles data cleaning and preprocessing
├── Feature_Engineering.py       # Implements feature selection techniques
├── Model_Building.py            # Trains models and evaluates performance
├── params.yaml                  # Configuration file with parameters
├── data/
│   ├── interim/                 # Interim datasets
│   ├── processed/               # Final datasets for modeling
│   ├── raw/                     # Raw datasets
├── models/                      # Saved models and hyperparameter configurations
├── assets/                      # Visualizations and reports
└── README.md                    # Project documentation
```

---

## Requirements

To run this project, ensure you have the following installed:

- Python 3.10+
- Required Python libraries (see [requirements.txt](#))
- GPU support for training (optional, recommended for MLP)

Install dependencies using:
```bash
pip install -r requirements.txt
```

---

## Scripts Description

### 1. **Data Preprocessing (`data_preprocessing.py`)**
This script handles the following:
- Null value handling
- Feature variance thresholding
- Class balancing using downsampling
- Logistic regression and chi-squared-based feature selection
- Robust scaling and power transformations

Output:
- Cleaned datasets saved in `data/interim/`

### 2. **Feature Engineering (`Feature_Engineering.py`)**
This script performs feature selection:
- Fisher score
- Random Forest feature importance
- XGBoost feature importance
- ANOVA and backward feature elimination

Output:
- Selected features and reduced datasets in `data/processed/`

### 3. **Model Building (`Model_Building.py`)**
Trains and evaluates models:
- **XGBoost**: Hyperparameter optimization using Optuna
- **MLP**: Hyperparameter optimization and training using PyTorch

Generates:
- Saved models in `models/`
- ROC curves, confusion matrices, and classification reports in `assets/`

---

## Configuration

Edit `params.yaml` to configure:
- File paths for datasets
- Parameters for data preprocessing, feature engineering, and model training
- Experiment tracking using MLflow

Example:
```yaml
experiment:
  TRACKING_URI: "http://localhost:5000"
  EXPERIMENT_NAME: "fraud_detection_pipeline"
data_cleaning:
  NULL_PERCENTAGE: 0.2
  VARIANCE_THRESHOLD: 0.01
  IMPUTATION_TECHNIQUE: "knn"
```

---

## Usage

### 1. Preprocess Data
```bash
python data_preprocessing.py
```

### 2. Perform Feature Engineering
```bash
python Feature_Engineering.py
```

### 3. Train Models
Specify the model type (`xgb` or `mlp`) in `params.yaml` and run:
```bash
python Model_Building.py
```

---

## Results and Reports
- Final datasets: `data/processed/`
- Model artifacts: `models/`
- Visualizations and metrics: `assets/`

---

## Experiment Tracking
This project integrates with MLflow for tracking:
- Parameters
- Metrics
- Artifacts (models, plots, and reports)

Launch the MLflow UI:
```bash
mlflow ui
```
Open your browser and navigate to [http://localhost:5000](http://localhost:5000).

