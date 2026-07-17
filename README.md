# Clinical Disease Risk Assessment System

An end-to-end machine learning-powered clinical decision support system for diabetes risk prediction, built with Streamlit. The system trains and compares three models (XGBoost, Random Forest, SVM) with SMOTE oversampling and hyperparameter tuning, then provides an interactive web interface for patient risk assessment.

## Features

### Patient Assessment
- **Disease Consultation** - Enter the condition of concern (e.g., "Type 2 Diabetes Mellitus") for context-aware risk assessment
- **8 Diagnostic Inputs** - Pregnancies, Glucose, Blood Pressure, Skin Thickness, Insulin, BMI, Diabetes Pedigree Function, Age
- **Clinical Reference Ranges** - Normal/abnormal thresholds displayed below each input field
- **3 ML Models** - XGBoost (recommended), Random Forest, and SVM with RBF kernel
- **Risk Classification** - Low Risk (<30%), Moderate Risk (30-60%), High Risk (>60%) with color-coded badges
- **Probability Gauge** - Visual risk meter showing probability on a color gradient scale
- **Feature Contribution** - Horizontal bar chart showing how each measurement contributes to the prediction
- **SHAP Interpretability** - Waterfall plots explaining model predictions (for tree-based models)
- **PDF Report** - Downloadable clinical report with patient data, risk assessment, and interpretation

### Model Performance
- **Metrics Table** - Recall and ROC-AUC scores for all three models
- **ROC Curves** - Comparative visualization of model discrimination ability
- **Feature Importance** - XGBoost and Random Forest feature importance charts

### Dataset Analysis
- **Dataset Statistics** - Sample count, class distribution, imbalance ratio
- **Data Preview** - First 10 records from the cleaned dataset
- **Class Distribution** - Bar chart and interactive KDE plots by class
- **Descriptive Statistics** - Summary statistics for all features

### Assessment History
- **Persistent Storage** - All assessments saved to CSV across sessions
- **Summary Metrics** - Total assessments, high/low risk counts
- **History Table** - Sortable view of all past assessments
- **Clear History** - Option to reset assessment records

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Web Framework | [Streamlit](https://streamlit.io/) 1.58+ |
| Machine Learning | [Scikit-Learn](https://scikit-learn.org/) 1.9+, [XGBoost](https://xgboost.readthedocs.io/) 3.3+ |
| Imbalance Handling | [Imbalanced-Learn](https://imbalanced-learn.org/) (SMOTE) |
| Model Interpretability | [SHAP](https://shap.readthedocs.io/) 0.52+ |
| PDF Generation | [FPDF2](https://pyfpdf.github.io/fpdf2/) 2.8+ |
| Data Processing | [Pandas](https://pandas.pydata.org/) 3.0+, [NumPy](https://numpy.org/) 2.4+ |
| Visualization | [Matplotlib](https://matplotlib.org/), [Seaborn](https://seaborn.pydata.org/) |
| Model Persistence | [Joblib](https://joblib.readthedocs.io/) |

## Dataset

**Pima Indians Diabetes Database** (UCI Machine Learning Repository)

- **Samples:** 768 female patients of Pima Indian heritage
- **Features:** 8 diagnostic measurements
- **Target:** Diabetes outcome (268 positive, 500 negative)
- **Imbalance Ratio:** ~1.9:1 (non-diabetic : diabetic)

### Features
| Feature | Description | Clinical Range |
|---------|-------------|----------------|
| Pregnancies | Number of pregnancies | 0-20 |
| Glucose | Plasma glucose (mg/dL) | Normal: 70-100 |
| BloodPressure | Diastolic BP (mm Hg) | Normal: <80 |
| SkinThickness | Triceps skin fold (mm) | Typical: 10-40 |
| Insulin | 2-hour serum insulin (mu U/mL) | Normal: 16-166 |
| BMI | Body mass index | Normal: 18.5-24.9 |
| DiabetesPedigreeFunction | Genetic risk score | Higher = greater risk |
| Age | Patient age (years) | Risk increases >45 |

## Pipeline

The system implements a complete ML pipeline:

1. **Data Loading** - Fetch Pima Indians Diabetes dataset from UCI repository
2. **Data Cleaning** - Replace zero values with NaN, impute with median
3. **Stratified Split** - 80/20 train-test split preserving class distribution
4. **Feature Scaling** - StandardScaler normalization
5. **SMOTE Oversampling** - Synthetic minority oversampling to balance classes
6. **Hyperparameter Tuning** - GridSearchCV with 3-fold stratified CV, optimized for recall
7. **Model Training** - XGBoost, Random Forest, and SVM with best parameters
8. **Evaluation** - 5-fold CV and held-out test set with recall, precision, F1, ROC-AUC
9. **Persistence** - Models saved to disk with joblib for instant loading

## Installation

### Prerequisites
- Python 3.10+
- pip package manager

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd "ML Task 4"

# Install dependencies
pip install -r requirements.txt
```

### Requirements

Create a `requirements.txt` file with:

```
streamlit>=1.58.0
numpy>=2.4.0
pandas>=3.0.0
scikit-learn>=1.9.0
xgboost>=3.3.0
imbalanced-learn>=0.14.0
matplotlib>=3.10.0
seaborn>=0.13.0
joblib>=1.5.0
shap>=0.52.0
fpdf2>=2.8.0
```

## Usage

```bash
streamlit run app.py
```

The application will be available at **http://localhost:8501**.

### First Launch
On the first run, the system will:
1. Download the Pima Indians Diabetes dataset
2. Train all three models with hyperparameter tuning
3. Save trained models to the `saved_models/` directory
4. Launch the web interface

### Subsequent Launches
Models are loaded from disk instantly - no retraining needed.

## Project Structure

```
ML Task 4/
├── app.py                          # Main Streamlit application
├── clinical_disease_predictor.ipynb        # Original Jupyter notebook
├── clinical_disease_predictor_executed.ipynb # Executed notebook with outputs
├── README.md                       # This file
├── requirements.txt                # Python dependencies
├── saved_models/                   # Trained model artifacts (auto-generated)
│   ├── scaler.pkl
│   ├── xgb_model.pkl
│   ├── rf_model.pkl
│   ├── svm_model.pkl
│   ├── test_metrics.pkl
│   ├── df_clean.pkl
│   └── feature_names.json
└── assessment_history.csv          # Assessment records (auto-generated)
```

## Model Performance

The models are evaluated on a held-out test set (20% of data) with the following metrics:

| Model | Recall (Sensitivity) | ROC-AUC |
|-------|---------------------|---------|
| XGBoost | ~85-90% | ~0.85-0.90 |
| Random Forest | ~80-85% | ~0.83-0.88 |
| SVM (RBF) | ~75-82% | ~0.80-0.85 |

*Note: Exact metrics vary based on random state and hyperparameter search results.*

## Key Design Decisions

- **Recall as primary metric** - In medical screening, missing a positive case (false negative) is more costly than a false positive. All models are optimized for recall.
- **SMOTE + class weighting** - Dual approach to handle class imbalance: synthetic oversampling combined with balanced class weights.
- **Stratified cross-validation** - Preserves class distribution in each fold for reliable evaluation.
- **Model persistence** - Joblib serialization avoids retraining on every app restart.
- **CSV-based history** - Simple, portable storage for assessment records without requiring a database.

## Disclaimer

This system is intended for **educational and research purposes only**. It is not a substitute for professional medical diagnosis or clinical decision-making. All findings should be reviewed by a qualified healthcare provider.

## License

This project is provided for educational use. The Pima Indians Diabetes Dataset is sourced from the UCI Machine Learning Repository.