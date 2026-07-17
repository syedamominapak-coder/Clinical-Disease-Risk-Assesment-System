import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import json
import os
from datetime import datetime
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import joblib
import shap

from fpdf import FPDF
import base64
import io

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Clinical Disease Risk Assessment System",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS - Professional Medical Theme
# ============================================================
st.markdown("""
<style>
    #root > div:first-child { background: #f5f7fa; }
    .main-header {
        background: linear-gradient(135deg, #1a365d 0%, #2c5282 50%, #2b6cb0 100%);
        padding: 1.8rem 2rem;
        border-radius: 8px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.12);
    }
    .main-header h1 { font-size: 1.8rem; font-weight: 600; margin: 0; letter-spacing: 0.5px; }
    .main-header p { font-size: 0.95rem; opacity: 0.85; margin: 0.3rem 0 0 0; font-weight: 300; }
    .card {
        background: white;
        padding: 1.5rem;
        border-radius: 8px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
        margin-bottom: 1.2rem;
        border: 1px solid #e2e8f0;
    }
    .card h3 {
        color: #1a365d;
        font-size: 1.1rem;
        font-weight: 600;
        margin: 0 0 0.8rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e2e8f0;
    }
    .metric-card {
        background: #f7fafc;
        padding: 0.8rem 1rem;
        border-radius: 6px;
        text-align: center;
        border-left: 3px solid #2b6cb0;
    }
    .metric-card .value { font-size: 1.5rem; font-weight: 700; color: #1a365d; }
    .metric-card .label {
        font-size: 0.8rem;
        color: #718096;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .risk-low {
        background: #c6f6d5; color: #22543d;
        padding: 0.4rem 1.2rem; border-radius: 4px;
        font-weight: 600; font-size: 0.95rem;
        display: inline-block; border: 1px solid #9ae6b4;
    }
    .risk-moderate {
        background: #fefcbf; color: #744210;
        padding: 0.4rem 1.2rem; border-radius: 4px;
        font-weight: 600; font-size: 0.95rem;
        display: inline-block; border: 1px solid #f6e05e;
    }
    .risk-high {
        background: #fed7d7; color: #822727;
        padding: 0.4rem 1.2rem; border-radius: 4px;
        font-weight: 600; font-size: 0.95rem;
        display: inline-block; border: 1px solid #fc8181;
    }
    .sidebar-section {
        background: #f7fafc; padding: 1rem;
        border-radius: 6px; margin-bottom: 1rem;
        border: 1px solid #e2e8f0;
    }
    .sidebar-section h4 { color: #1a365d; font-size: 0.9rem; font-weight: 600; margin: 0 0 0.5rem 0; }
    .divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, #a0aec0, transparent);
        margin: 1.2rem 0;
    }
    .footer {
        text-align: center; color: #a0aec0;
        font-size: 0.8rem; padding: 1rem;
        border-top: 1px solid #e2e8f0; margin-top: 2rem;
    }
    .input-section-label {
        font-weight: 600; color: #2c5282;
        font-size: 0.85rem; text-transform: uppercase;
        letter-spacing: 0.5px; margin-bottom: 0.3rem;
    }
    .ref-range {
        font-size: 0.75rem;
        color: #718096;
        font-style: italic;
        margin-top: 0.1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 0; background: white;
        border-radius: 8px 8px 0 0;
        border: 1px solid #e2e8f0; border-bottom: none;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 0.6rem 1.5rem; font-size: 0.9rem;
        font-weight: 500; color: #4a5568;
    }
    .stTabs [aria-selected="true"] {
        color: #2b6cb0 !important; font-weight: 600;
        border-bottom: 2px solid #2b6cb0 !important;
    }
    .stButton > button {
        background: #2b6cb0; color: white;
        border: none; border-radius: 6px;
        padding: 0.5rem 1.5rem; font-weight: 500;
        font-size: 0.95rem; transition: background 0.2s;
    }
    .stButton > button:hover { background: #1a365d; color: white; border: none; }
    .stNumberInput input, .stSelectbox > div > div {
        border: 1px solid #e2e8f0 !important; border-radius: 6px !important;
    }
    .stNumberInput input:focus {
        border-color: #2b6cb0 !important; box-shadow: 0 0 0 1px #2b6cb0 !important;
    }
    div[data-testid="stExpander"] { border: 1px solid #e2e8f0; border-radius: 6px; }
    .stDataFrame { border: 1px solid #e2e8f0; border-radius: 6px; }
    .stAlert { border-radius: 6px; }
    .history-item {
        background: #f7fafc;
        border: 1px solid #e2e8f0;
        border-radius: 6px;
        padding: 0.8rem;
        margin-bottom: 0.6rem;
    }
    .history-item .date { font-size: 0.75rem; color: #a0aec0; }
    .history-item .risk { font-weight: 600; font-size: 0.9rem; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# CONSTANTS
# ============================================================
MODEL_DIR = "saved_models"
os.makedirs(MODEL_DIR, exist_ok=True)

FEATURE_NAMES = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

CLINICAL_REF_RANGES = {
    'Glucose': 'Normal: 70-100 mg/dL | Prediabetes: 100-125 | Diabetes: >126',
    'BloodPressure': 'Normal: <80 mm Hg | Elevated: 80-89 | Hypertension: >90',
    'BMI': 'Normal: 18.5-24.9 | Overweight: 25-29.9 | Obese: >30',
    'Skin Thickness': 'Typical: 10-40 mm',
    'Insulin': 'Normal: 16-166 mu U/mL (fasting)',
    'Diabetes Pedigree': 'Higher values indicate greater genetic risk',
    'Age': 'Risk increases with age, particularly >45 years',
    'Pregnancies': 'Number of pregnancies (relevant for gestational diabetes risk)'
}

HISTORY_FILE = "assessment_history.csv"

# ============================================================
# HELPER FUNCTIONS
# ============================================================
def save_models_to_disk(scaler, xgb_model, rf_model, svm_model, test_metrics, df_clean, X_test=None, y_test=None):
    """Save trained models to disk for fast loading."""
    joblib.dump(scaler, os.path.join(MODEL_DIR, 'scaler.pkl'))
    joblib.dump(xgb_model, os.path.join(MODEL_DIR, 'xgb_model.pkl'))
    joblib.dump(rf_model, os.path.join(MODEL_DIR, 'rf_model.pkl'))
    joblib.dump(svm_model, os.path.join(MODEL_DIR, 'svm_model.pkl'))
    joblib.dump(test_metrics, os.path.join(MODEL_DIR, 'test_metrics.pkl'))
    joblib.dump(df_clean, os.path.join(MODEL_DIR, 'df_clean.pkl'))
    if X_test is not None:
        joblib.dump(X_test, os.path.join(MODEL_DIR, 'X_test.pkl'))
    if y_test is not None:
        joblib.dump(y_test, os.path.join(MODEL_DIR, 'y_test.pkl'))
    # Save feature names for consistency
    with open(os.path.join(MODEL_DIR, 'feature_names.json'), 'w') as f:
        json.dump(FEATURE_NAMES, f)
    return True

def load_models_from_disk():
    """Load saved models from disk if they exist."""
    required_files = ['scaler.pkl', 'xgb_model.pkl', 'rf_model.pkl', 'svm_model.pkl',
                      'test_metrics.pkl', 'df_clean.pkl', 'feature_names.json']
    for f in required_files:
        if not os.path.exists(os.path.join(MODEL_DIR, f)):
            return None
    
    # Load feature names from JSON
    with open(os.path.join(MODEL_DIR, 'feature_names.json'), 'r') as f:
        feature_names = json.load(f)
    
    result = {
        'scaler': joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl')),
        'xgb_model': joblib.load(os.path.join(MODEL_DIR, 'xgb_model.pkl')),
        'rf_model': joblib.load(os.path.join(MODEL_DIR, 'rf_model.pkl')),
        'svm_model': joblib.load(os.path.join(MODEL_DIR, 'svm_model.pkl')),
        'test_metrics': joblib.load(os.path.join(MODEL_DIR, 'test_metrics.pkl')),
        'df_clean': joblib.load(os.path.join(MODEL_DIR, 'df_clean.pkl')),
        'feature_names': feature_names
    }
    
    # Load X_test and y_test if they exist (saved from a previous training run)
    x_test_path = os.path.join(MODEL_DIR, 'X_test.pkl')
    y_test_path = os.path.join(MODEL_DIR, 'y_test.pkl')
    if os.path.exists(x_test_path):
        result['X_test'] = joblib.load(x_test_path)
    if os.path.exists(y_test_path):
        result['y_test'] = joblib.load(y_test_path)
    
    return result

@st.cache_resource
def load_and_train_models():
    """Load data, preprocess, train models, and return artifacts. Caches to disk."""
    # Try loading from disk first
    cached = load_models_from_disk()
    if cached is not None:
        return cached
    
    # Train from scratch
    url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv'
    columns = FEATURE_NAMES + ['Outcome']
    df = pd.read_csv(url, names=columns)
    
    df_clean = df.copy()
    cols_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for col in cols_with_zeros:
        df_clean[col] = df_clean[col].replace(0, np.nan)
        df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    
    X = df_clean.drop('Outcome', axis=1).values
    y = df_clean['Outcome'].values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)
    
    # XGBoost
    neg_count = (y_train_smote == 0).sum()
    pos_count = (y_train_smote == 1).sum()
    scale_pos_weight = neg_count / pos_count
    
    xgb_param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }
    
    xgb_base = xgb.XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric='logloss',
        use_label_encoder=False
    )
    xgb_grid = GridSearchCV(
        xgb_base, xgb_param_grid,
        cv=StratifiedKFold(3, shuffle=True, random_state=42),
        scoring='recall', n_jobs=-1, verbose=0
    )
    xgb_grid.fit(X_train_smote, y_train_smote)
    xgb_best = xgb_grid.best_estimator_
    
    # Random Forest
    rf_param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [5, 10, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'class_weight': ['balanced', None]
    }
    rf_base = RandomForestClassifier(random_state=42)
    rf_grid = GridSearchCV(
        rf_base, rf_param_grid,
        cv=StratifiedKFold(3, shuffle=True, random_state=42),
        scoring='recall', n_jobs=-1, verbose=0
    )
    rf_grid.fit(X_train_smote, y_train_smote)
    rf_best = rf_grid.best_estimator_
    
    # SVM
    svm_param_grid = {
        'C': [0.1, 1, 10],
        'gamma': ['scale', 'auto'],
        'kernel': ['rbf'],
        'class_weight': ['balanced', None]
    }
    svm_base = SVC(random_state=42, probability=True)
    svm_grid = GridSearchCV(
        svm_base, svm_param_grid,
        cv=StratifiedKFold(3, shuffle=True, random_state=42),
        scoring='recall', n_jobs=-1, verbose=0
    )
    svm_grid.fit(X_train_smote, y_train_smote)
    svm_best = svm_grid.best_estimator_
    
    # Evaluate
    models = {'XGBoost': xgb_best, 'Random Forest': rf_best, 'SVM (RBF)': svm_best}
    test_metrics = {}
    for name, model in models.items():
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
        test_metrics[name] = {
            'recall': recall_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_proba)
        }
    
    artifacts = {
        'scaler': scaler,
        'xgb_model': xgb_best,
        'rf_model': rf_best,
        'svm_model': svm_best,
        'feature_names': FEATURE_NAMES,
        'test_metrics': test_metrics,
        'X_test': X_test_scaled,
        'y_test': y_test,
        'df_clean': df_clean
    }
    
    # Save to disk for future use
    save_models_to_disk(scaler, xgb_best, rf_best, svm_best, test_metrics, df_clean, X_test_scaled, y_test)
    
    return artifacts


def predict_diabetes(model, scaler, features):
    """Make prediction for a single patient."""
    features_scaled = scaler.transform([features])
    proba = model.predict_proba(features_scaled)[0]
    prediction = model.predict(features_scaled)[0]
    return prediction, proba


def get_risk_level(probability):
    """Categorize risk level based on probability."""
    if probability < 0.3:
        return "Low Risk", "risk-low"
    elif probability < 0.6:
        return "Moderate Risk", "risk-moderate"
    else:
        return "High Risk", "risk-high"


def create_gauge_chart(probability):
    """Create a gauge-like visualization for probability."""
    fig, ax = plt.subplots(figsize=(6, 1.2))
    
    gradient = np.linspace(0, 1, 100)
    for i in range(len(gradient) - 1):
        if gradient[i] < 0.3:
            c = '#48bb78'
        elif gradient[i] < 0.6:
            c = '#ecc94b'
        else:
            c = '#fc8181'
        ax.barh(0, 0.01, left=gradient[i], height=0.25, color=c, alpha=0.85)
    
    ax.axvline(x=probability, ymin=0.15, ymax=0.85, color='#1a202c', linewidth=2)
    ax.plot(probability, 0, 'v', color='#1a202c', markersize=10, zorder=5)
    
    ax.text(0.15, -0.35, 'Low', ha='center', fontsize=8, color='#48bb78', fontweight='bold')
    ax.text(0.45, -0.35, 'Moderate', ha='center', fontsize=8, color='#d69e2e', fontweight='bold')
    ax.text(0.85, -0.35, 'High', ha='center', fontsize=8, color='#fc8181', fontweight='bold')
    ax.text(probability, 0.4, f'{probability:.1%}', ha='center', fontsize=10, 
            fontweight='bold', color='#1a365d')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.4, 0.5)
    ax.axis('off')
    plt.tight_layout()
    return fig


def create_shap_force_plot(model, features_scaled, feature_names):
    """Create SHAP waterfall plot for model interpretability."""
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(features_scaled)
        
        if isinstance(shap_values, list):
            # For multi-class, use class 1 (positive)
            sv = shap_values[1]
        else:
            sv = shap_values
        
        fig, ax = plt.subplots(figsize=(8, 4))
        shap.waterfall_plot(
            shap.Explanation(values=sv[0], 
                           base_values=explainer.expected_value if not isinstance(explainer.expected_value, list) else explainer.expected_value[1],
                           data=features_scaled[0],
                           feature_names=feature_names),
            show=False,
            max_display=8
        )
        plt.tight_layout()
        return fig
    except Exception:
        return None


def create_feature_importance_chart(model, feature_names, title="Feature Importance"):
    """Create feature importance horizontal bar chart."""
    importances = model.feature_importances_
    indices = np.argsort(importances)
    
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = plt.cm.Blues(np.linspace(0.35, 0.85, len(feature_names)))
    bars = ax.barh(range(len(feature_names)), importances[indices], color=colors[::-1], 
                   edgecolor='white', height=0.7)
    
    ax.set_yticks(range(len(feature_names)))
    ax.set_yticklabels([feature_names[i] for i in indices], fontsize=9)
    ax.set_xlabel('Importance Score', fontsize=10)
    ax.set_title(title, fontsize=12, fontweight='bold', color='#1a365d')
    ax.set_xlim(0, max(importances) * 1.2)
    
    for bar, val in zip(bars, importances[indices]):
        ax.text(val + 0.005, bar.get_y() + bar.get_height()/2, f'{val:.3f}',
                va='center', fontsize=8, color='#4a5568')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(colors='#4a5568')
    plt.tight_layout()
    return fig


def create_roc_curve(models_dict, X_test, y_test):
    """Create ROC curve comparison chart."""
    fig, ax = plt.subplots(figsize=(7, 5.5))
    colors = ['#2b6cb0', '#b83280', '#dd6b20']
    for idx, (name, model) in enumerate(models_dict.items()):
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        ax.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {auc:.3f})', color=colors[idx])
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random Classifier', alpha=0.4)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=10)
    ax.set_ylabel('True Positive Rate (Recall / Sensitivity)', fontsize=10)
    ax.set_title('ROC Curves - Model Comparison', fontsize=12, fontweight='bold', color='#1a365d')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(alpha=0.2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(colors='#4a5568')
    plt.tight_layout()
    return fig


def generate_pdf_report(patient_data, risk_label, probability, prediction, model_choice, disease_concern, feature_names):
    """Generate a PDF report of the assessment."""
    pdf = FPDF()
    pdf.add_page()
    
    # Header
    pdf.set_font('Arial', 'B', 16)
    pdf.set_text_color(26, 54, 93)
    pdf.cell(0, 10, 'Clinical Disease Risk Assessment Report', 0, 1, 'C')
    pdf.ln(5)
    
    # Line
    pdf.set_draw_color(43, 108, 176)
    pdf.set_line_width(0.5)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(5)
    
    # Report info
    pdf.set_font('Arial', 'B', 11)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 7, f'Date: {datetime.now().strftime("%Y-%m-%d %H:%M")}', 0, 1)
    pdf.cell(0, 7, f'Condition Assessed: {disease_concern}', 0, 1)
    pdf.cell(0, 7, f'Analysis Model: {model_choice}', 0, 1)
    pdf.ln(5)
    
    # Risk Assessment
    pdf.set_fill_color(26, 54, 93)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 8, 'Risk Assessment Result', 0, 1, 'L', fill=True)
    pdf.ln(3)
    
    pdf.set_text_color(0, 0, 0)
    pdf.set_font('Arial', '', 11)
    pdf.cell(0, 7, f'Risk Level: {risk_label}', 0, 1)
    pdf.cell(0, 7, f'Diabetes Probability: {probability:.1%}', 0, 1)
    pdf.cell(0, 7, f'Predicted Class: {"Positive" if prediction == 1 else "Negative"}', 0, 1)
    pdf.ln(5)
    
    # Patient Measurements
    pdf.set_fill_color(26, 54, 93)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 8, 'Patient Diagnostic Measurements', 0, 1, 'L', fill=True)
    pdf.ln(3)
    
    pdf.set_text_color(0, 0, 0)
    pdf.set_font('Arial', '', 10)
    for name, value in zip(feature_names, patient_data):
        pdf.cell(0, 6, f'{name}: {value:.1f}', 0, 1)
    pdf.ln(5)
    
    # Interpretation
    pdf.set_fill_color(26, 54, 93)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 8, 'Clinical Interpretation', 0, 1, 'L', fill=True)
    pdf.ln(3)
    
    pdf.set_text_color(0, 0, 0)
    pdf.set_font('Arial', '', 10)
    interpretation = (
        f"This assessment indicates a {risk_label.lower()} for {disease_concern.lower()} "
        f"with a probability of {probability:.1%}. "
    )
    if probability < 0.3:
        interpretation += "The patient's diagnostic profile suggests low risk. Continue routine monitoring and preventive care."
    elif probability < 0.6:
        interpretation += "The patient shows moderate risk factors. Further diagnostic evaluation and lifestyle interventions are recommended."
    else:
        interpretation += "The patient demonstrates high risk. Immediate clinical follow-up, confirmatory testing, and comprehensive management are strongly recommended."
    
    pdf.multi_cell(0, 5, interpretation)
    pdf.ln(5)
    
    # Disclaimer
    pdf.set_text_color(100, 100, 100)
    pdf.set_font('Arial', 'I', 8)
    pdf.multi_cell(0, 4, 'Disclaimer: This report is generated by an automated clinical decision support system for educational and research purposes only. It is not a substitute for professional medical diagnosis or clinical decision-making. All findings should be reviewed by a qualified healthcare provider.')
    
    return pdf


def get_pdf_download_link(pdf, filename):
    """Generate a download link for the PDF."""
    pdf_output = pdf.output(dest='S')
    if isinstance(pdf_output, str):
        pdf_output = pdf_output.encode('latin-1')
    b64 = base64.b64encode(pdf_output).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}">Download PDF Report</a>'
    return href


def save_assessment_to_history(record):
    """Save assessment record to CSV history."""
    df_new = pd.DataFrame([record])
    try:
        if os.path.exists(HISTORY_FILE):
            df_hist = pd.read_csv(HISTORY_FILE)
            df_hist = pd.concat([df_hist, df_new], ignore_index=True)
        else:
            df_hist = df_new
        df_hist.to_csv(HISTORY_FILE, index=False)
    except Exception:
        pass


def load_assessment_history():
    """Load assessment history from CSV."""
    try:
        if os.path.exists(HISTORY_FILE):
            return pd.read_csv(HISTORY_FILE)
    except Exception:
        pass
    return pd.DataFrame()


# ============================================================
# LOAD MODELS (cached to disk)
# ============================================================
with st.spinner('Loading and training models. Please wait...'):
    artifacts = load_and_train_models()

scaler = artifacts['scaler']
xgb_model = artifacts['xgb_model']
rf_model = artifacts['rf_model']
svm_model = artifacts['svm_model']
feature_names = artifacts['feature_names']
test_metrics = artifacts['test_metrics']
X_test = artifacts['X_test']
y_test = artifacts['y_test']
df_clean = artifacts['df_clean']

# Initialize session state for history
if 'history_initialized' not in st.session_state:
    st.session_state.history = load_assessment_history()
    st.session_state.history_initialized = True

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 0.8rem 0;">
        <h2 style="color: #1a365d; font-size: 1.1rem; margin: 0.3rem 0; font-weight: 700; letter-spacing: 0.5px;">CLINICAL RISK ASSESSMENT</h2>
        <p style="color: #718096; font-size: 0.8rem; margin: 0;">Diabetes Prediction System v2.0</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="sidebar-section">
        <h4>Model Performance</h4>
    </div>
    """, unsafe_allow_html=True)
    
    for model_name, metrics in test_metrics.items():
        st.markdown(f"""
        <div class="metric-card" style="margin-bottom: 0.4rem;">
            <div style="font-weight: 600; color: #1a365d; font-size: 0.85rem;">{model_name}</div>
            <div style="display: flex; justify-content: space-around; margin-top: 0.2rem;">
                <div><span style="font-size: 0.7rem; color: #718096;">Recall</span><br><span style="font-weight: 700; color: #22543d;">{metrics['recall']:.2%}</span></div>
                <div><span style="font-size: 0.7rem; color: #718096;">ROC-AUC</span><br><span style="font-weight: 700; color: #1a365d;">{metrics['roc_auc']:.2%}</span></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="sidebar-section">
        <h4>System Information</h4>
        <p style="font-size: 0.8rem; color: #4a5568;">
        This clinical decision support system uses ensemble machine learning to assess diabetes risk 
        based on patient diagnostic measurements. Models are trained on the Pima Indians Diabetes 
        Database with SMOTE oversampling to handle class imbalance.
        </p>
        <p style="font-size: 0.8rem; color: #4a5568; margin-bottom: 0;">
        <strong>Models saved to disk:</strong> Instant loading on subsequent launches.
        </p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# MAIN CONTENT
# ============================================================
st.markdown("""
<div class="main-header">
    <h1>Clinical Disease Risk Assessment System</h1>
    <p>Evidence-based diabetes risk prediction using machine learning analysis of patient diagnostic data</p>
</div>
""", unsafe_allow_html=True)

# Disease concern input
st.markdown("""
<div class="card">
    <h3>Patient Consultation</h3>
    <p style="color: #718096; font-size: 0.85rem; margin: 0;">
    Please specify the condition of concern and enter the patient's diagnostic measurements to generate a risk assessment.
    </p>
</div>
""", unsafe_allow_html=True)

disease_concern = st.text_input(
    "Condition of Concern",
    value="Type 2 Diabetes Mellitus",
    placeholder="e.g., Diabetes, Prediabetes, Metabolic Syndrome",
    help="Enter the medical condition or disease you want to assess risk for"
)

if disease_concern.strip():
    st.markdown(f"""
    <div style="background: #ebf4ff; border: 1px solid #bee3f8; border-radius: 6px; padding: 0.6rem 1rem; margin-bottom: 1rem;">
        <p style="margin: 0; font-size: 0.85rem; color: #2b6cb0;">
            <strong>Consultation:</strong> Risk assessment for <strong>"{disease_concern.strip()}"</strong>. 
            Complete the diagnostic fields below to generate the report.
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["Patient Assessment", "Model Performance", "Dataset Analysis", "Assessment History"])

# ============================================================
# TAB 1: PATIENT ASSESSMENT
# ============================================================
with tab1:
    st.markdown("""
    <div class="card">
        <h3>Patient Risk Assessment</h3>
        <p style="color: #718096; font-size: 0.85rem; margin: 0;">
        Enter the patient's diagnostic measurements below. Clinical reference ranges are provided for guidance.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col_left, col_right = st.columns([1, 1.2])
    
    with col_left:
        st.markdown("""
        <div class="card">
            <h3>Patient Diagnostic Data</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Pregnancy History
        st.markdown('<p class="input-section-label">Reproductive History</p>', unsafe_allow_html=True)
        pregnancies = st.number_input(
            "Number of Pregnancies",
            min_value=0, max_value=20, value=2, step=1,
            help=CLINICAL_REF_RANGES['Pregnancies']
        )
        st.markdown(f'<p class="ref-range">{CLINICAL_REF_RANGES["Pregnancies"]}</p>', unsafe_allow_html=True)
        
        st.markdown('<div style="height: 0.3rem;"></div>', unsafe_allow_html=True)
        
        # Glucose & Blood Pressure
        st.markdown('<p class="input-section-label">Metabolic Markers</p>', unsafe_allow_html=True)
        col_gluc, col_bp = st.columns(2)
        with col_gluc:
            glucose = st.number_input(
                "Glucose (mg/dL)",
                min_value=0, max_value=300, value=120, step=1,
                help=CLINICAL_REF_RANGES['Glucose']
            )
            st.markdown(f'<p class="ref-range">{CLINICAL_REF_RANGES["Glucose"]}</p>', unsafe_allow_html=True)
        with col_bp:
            blood_pressure = st.number_input(
                "Blood Pressure (mm Hg)",
                min_value=0, max_value=150, value=70, step=1,
                help=CLINICAL_REF_RANGES['BloodPressure']
            )
            st.markdown(f'<p class="ref-range">{CLINICAL_REF_RANGES["BloodPressure"]}</p>', unsafe_allow_html=True)
        
        st.markdown('<div style="height: 0.3rem;"></div>', unsafe_allow_html=True)
        
        # Body Composition
        st.markdown('<p class="input-section-label">Body Composition</p>', unsafe_allow_html=True)
        col_skin, col_insulin = st.columns(2)
        with col_skin:
            skin_thickness = st.number_input(
                "Skin Thickness (mm)",
                min_value=0, max_value=100, value=20, step=1,
                help=CLINICAL_REF_RANGES['Skin Thickness']
            )
            st.markdown(f'<p class="ref-range">{CLINICAL_REF_RANGES["Skin Thickness"]}</p>', unsafe_allow_html=True)
        with col_insulin:
            insulin = st.number_input(
                "Insulin (mu U/mL)",
                min_value=0, max_value=900, value=80, step=1,
                help=CLINICAL_REF_RANGES['Insulin']
            )
            st.markdown(f'<p class="ref-range">{CLINICAL_REF_RANGES["Insulin"]}</p>', unsafe_allow_html=True)
        
        st.markdown('<div style="height: 0.3rem;"></div>', unsafe_allow_html=True)
        
        # BMI & Demographics
        st.markdown('<p class="input-section-label">Anthropometric Data</p>', unsafe_allow_html=True)
        col_bmi, col_dpf, col_age = st.columns(3)
        with col_bmi:
            bmi = st.number_input(
                "BMI",
                min_value=0.0, max_value=70.0, value=28.0, step=0.1, format="%.1f",
                help=CLINICAL_REF_RANGES['BMI']
            )
            st.markdown(f'<p class="ref-range">{CLINICAL_REF_RANGES["BMI"]}</p>', unsafe_allow_html=True)
        with col_dpf:
            dpf = st.number_input(
                "Diabetes Pedigree",
                min_value=0.0, max_value=3.0, value=0.5, step=0.01, format="%.2f",
                help=CLINICAL_REF_RANGES['Diabetes Pedigree']
            )
            st.markdown(f'<p class="ref-range">{CLINICAL_REF_RANGES["Diabetes Pedigree"]}</p>', unsafe_allow_html=True)
        with col_age:
            age = st.number_input(
                "Age (years)",
                min_value=18, max_value=120, value=35, step=1,
                help=CLINICAL_REF_RANGES['Age']
            )
            st.markdown(f'<p class="ref-range">{CLINICAL_REF_RANGES["Age"]}</p>', unsafe_allow_html=True)
        
        st.markdown('<div style="height: 0.8rem;"></div>', unsafe_allow_html=True)
        
        # Model selection
        model_choice = st.selectbox(
            "Analysis Model",
            options=["XGBoost (Recommended)", "Random Forest", "SVM (RBF)"],
            help="Select the machine learning model for risk assessment. XGBoost provides the best recall for disease detection."
        )
        
        model_map = {
            "XGBoost (Recommended)": xgb_model,
            "Random Forest": rf_model,
            "SVM (RBF)": svm_model
        }
        selected_model = model_map[model_choice]
        
        # Predict button
        predict_btn = st.button("Compute Risk Assessment", type="primary", use_container_width=True)
    
    with col_right:
        if predict_btn:
            features = np.array([pregnancies, glucose, blood_pressure, skin_thickness,
                                 insulin, bmi, dpf, age])
            
            # Validate
            validation_messages = []
            if glucose == 0:
                validation_messages.append("Glucose value is 0. Please enter a valid measurement.")
            if blood_pressure == 0:
                validation_messages.append("Blood Pressure value is 0. Please enter a valid measurement.")
            if bmi == 0:
                validation_messages.append("BMI value is 0. Please enter a valid measurement.")
            
            if validation_messages:
                for msg in validation_messages:
                    st.warning(msg)
            else:
                prediction, proba = predict_diabetes(selected_model, scaler, features)
                risk_label, risk_class = get_risk_level(proba[1])
                
                # Results
                st.markdown("""
                <div class="card">
                    <h3>Assessment Results</h3>
                </div>
                """, unsafe_allow_html=True)
                
                # Risk badge and probability
                col_badge, col_prob = st.columns([1, 1])
                with col_badge:
                    st.markdown(f'<div style="text-align: center; padding: 0.8rem;"><span class="{risk_class}">{risk_label}</span></div>', unsafe_allow_html=True)
                with col_prob:
                    st.markdown(f"""
                    <div style="text-align: center; padding: 0.8rem;">
                        <div style="font-size: 2rem; font-weight: 700; color: #1a365d;">{proba[1]:.1%}</div>
                        <div style="color: #718096; font-size: 0.8rem;">Diabetes Probability</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Gauge
                gauge_fig = create_gauge_chart(proba[1])
                st.pyplot(gauge_fig)
                
                st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
                
                # Details
                col_d1, col_d2, col_d3 = st.columns(3)
                with col_d1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="label">Model</div>
                        <div class="value" style="font-size: 1rem;">{model_choice}</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col_d2:
                    pred_class = "Positive" if prediction == 1 else "Negative"
                    pred_color = '#c53030' if prediction == 1 else '#22543d'
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="label">Predicted Class</div>
                        <div class="value" style="font-size: 1rem; color: {pred_color};">{pred_class}</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col_d3:
                    confidence = proba[prediction]
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="label">Confidence</div>
                        <div class="value" style="font-size: 1rem;">{confidence:.1%}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Save to history
                record = {
                    'Date': datetime.now().strftime('%Y-%m-%d %H:%M'),
                    'Disease': disease_concern.strip(),
                    'Model': model_choice,
                    'Risk Level': risk_label,
                    'Probability': f"{proba[1]:.1%}",
                    'Prediction': pred_class,
                    'Glucose': glucose,
                    'BloodPressure': blood_pressure,
                    'BMI': bmi,
                    'Age': age
                }
                save_assessment_to_history(record)
                st.session_state.history = load_assessment_history()
                
                st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
                
                # PDF Download
                pdf = generate_pdf_report(
                    features, risk_label, proba[1], prediction,
                    model_choice, disease_concern.strip(), feature_names
                )
                pdf_filename = f"risk_assessment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                pdf_link = get_pdf_download_link(pdf, pdf_filename)
                st.markdown(pdf_link, unsafe_allow_html=True)
                
                st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
                
                # Feature contribution
                st.markdown("""
                <div class="card">
                    <h3>Feature Contribution Analysis</h3>
                    <p style="color: #718096; font-size: 0.85rem; margin: 0;">
                    Relative contribution of each patient measurement to the computed risk score.
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                if hasattr(selected_model, 'feature_importances_'):
                    importances = selected_model.feature_importances_
                    feat_df = pd.DataFrame({
                        'Feature': feature_names,
                        'Value': features,
                        'Importance': importances
                    }).sort_values('Importance', ascending=False)
                    
                    col_f1, col_f2 = st.columns([1, 1])
                    with col_f1:
                        for _, row in feat_df.iterrows():
                            bar_width = row['Importance'] / feat_df['Importance'].max()
                            st.markdown(f"""
                            <div style="margin-bottom: 0.4rem;">
                                <div style="display: flex; justify-content: space-between; font-size: 0.8rem;">
                                    <span style="font-weight: 500; color: #2d3748;">{row['Feature']}</span>
                                    <span style="color: #718096;">{row['Value']:.1f}</span>
                                </div>
                                <div style="background: #edf2f7; border-radius: 4px; height: 16px; width: 100%;">
                                    <div style="background: linear-gradient(90deg, #2b6cb0, #1a365d); border-radius: 4px; height: 100%; width: {bar_width * 100}%;"></div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    with col_f2:
                        imp_fig = create_feature_importance_chart(selected_model, feature_names)
                        st.pyplot(imp_fig)
                    
                    # SHAP interpretability (for tree-based models)
                    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
                    st.markdown("""
                    <div class="card">
                        <h3>SHAP Model Interpretability</h3>
                        <p style="color: #718096; font-size: 0.85rem; margin: 0;">
                        SHAP (SHapley Additive exPlanations) values show how each feature contributes to pushing 
                        the prediction away from the baseline. Red features push toward higher risk, blue toward lower risk.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if model_choice != "SVM (RBF)":
                        features_scaled = scaler.transform([features])
                        shap_fig = create_shap_force_plot(selected_model, features_scaled, feature_names)
                        if shap_fig:
                            st.pyplot(shap_fig)
                    else:
                        st.info("SHAP analysis is not available for SVM models. Please select XGBoost or Random Forest for interpretability.")
                else:
                    st.info("Feature importance analysis is not available for the SVM model. Please select XGBoost or Random Forest for this visualization.")
        else:
            st.markdown("""
            <div class="card" style="text-align: center; padding: 2.5rem;">
                <div style="font-size: 3rem; font-weight: 100; color: #2b6cb0; line-height: 1;">+</div>
                <h3 style="color: #1a365d; margin-top: 0.8rem; font-size: 1.1rem;">Ready for Patient Assessment</h3>
                <p style="color: #718096; font-size: 0.85rem;">
                Enter patient diagnostic measurements in the left panel and click 
                <strong>"Compute Risk Assessment"</strong> to generate a risk analysis.
                </p>
                <div style="background: #f7fafc; border-radius: 6px; padding: 0.8rem; margin-top: 0.8rem; text-align: left; border: 1px solid #e2e8f0;">
                    <p style="font-size: 0.8rem; margin: 0; color: #4a5568;">
                    <strong>New Features:</strong> 
                    Clinical reference ranges shown below each input field. 
                    PDF report download available after assessment.
                    SHAP model interpretability visualization included.
                    Assessment history tracked across sessions.
                    </p>
                </div>
            </div>
            """, unsafe_allow_html=True)

# ============================================================
# TAB 2: MODEL PERFORMANCE
# ============================================================
with tab2:
    st.markdown("""
    <div class="card">
        <h3>Model Performance Evaluation</h3>
        <p style="color: #718096; font-size: 0.85rem; margin: 0;">
        Comparative analysis of all three machine learning models evaluated on a held-out test set 
        (20% of the original dataset, stratified to preserve class distribution). Models are trained 
        with SMOTE oversampling and hyperparameter tuning.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Metrics table
    metrics_data = []
    for model_name, metrics in test_metrics.items():
        metrics_data.append({
            'Model': model_name,
            'Recall (Sensitivity)': f"{metrics['recall']:.2%}",
            'ROC-AUC Score': f"{metrics['roc_auc']:.2%}"
        })
    
    metrics_df = pd.DataFrame(metrics_data)
    st.dataframe(metrics_df, width='stretch', hide_index=True)
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    # ROC Curve
    st.markdown("""
    <div class="card">
        <h3>Receiver Operating Characteristic (ROC) Curves</h3>
        <p style="color: #718096; font-size: 0.85rem; margin: 0;">
        ROC curves illustrate the diagnostic ability of each model across various threshold settings. 
        A higher Area Under the Curve (AUC) indicates better discrimination between positive and negative cases.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    models_dict = {'XGBoost': xgb_model, 'Random Forest': rf_model, 'SVM (RBF)': svm_model}
    roc_fig = create_roc_curve(models_dict, X_test, y_test)
    st.pyplot(roc_fig)
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    # Feature Importance
    st.markdown("""
    <div class="card">
        <h3>Predictive Feature Importance</h3>
        <p style="color: #718096; font-size: 0.85rem; margin: 0;">
        Identification of the most influential diagnostic features for diabetes prediction. 
        Glucose concentration, BMI, and age consistently emerge as the strongest predictors.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col_imp1, col_imp2 = st.columns(2)
    
    with col_imp1:
        imp_fig_xgb = create_feature_importance_chart(xgb_model, feature_names, "Feature Importance - XGBoost")
        st.pyplot(imp_fig_xgb)
    
    with col_imp2:
        rf_importances = rf_model.feature_importances_
        rf_indices = np.argsort(rf_importances)
        
        fig_rf, ax_rf = plt.subplots(figsize=(7, 4))
        colors_rf = plt.cm.Oranges(np.linspace(0.3, 0.75, len(feature_names)))
        bars_rf = ax_rf.barh(range(len(feature_names)), rf_importances[rf_indices], 
                            color=colors_rf[::-1], edgecolor='white', height=0.7)
        ax_rf.set_yticks(range(len(feature_names)))
        ax_rf.set_yticklabels([feature_names[i] for i in rf_indices], fontsize=9)
        ax_rf.set_xlabel('Importance Score', fontsize=10)
        ax_rf.set_title('Feature Importance - Random Forest', fontsize=12, fontweight='bold', color='#1a365d')
        ax_rf.set_xlim(0, max(rf_importances) * 1.2)
        for bar, val in zip(bars_rf, rf_importances[rf_indices]):
            ax_rf.text(val + 0.005, bar.get_y() + bar.get_height()/2, f'{val:.3f}',
                    va='center', fontsize=8, color='#4a5568')
        ax_rf.spines['top'].set_visible(False)
        ax_rf.spines['right'].set_visible(False)
        ax_rf.tick_params(colors='#4a5568')
        plt.tight_layout()
        st.pyplot(fig_rf)

# ============================================================
# TAB 3: DATASET ANALYSIS
# ============================================================
with tab3:
    st.markdown("""
    <div class="card">
        <h3>Dataset Overview</h3>
        <p style="color: #718096; font-size: 0.85rem; margin: 0;">
        Pima Indians Diabetes Database sourced from the UCI Machine Learning Repository. 
        The dataset contains diagnostic measurements from female patients of Pima Indian heritage.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col_ds1, col_ds2, col_ds3, col_ds4 = st.columns(4)
    
    with col_ds1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="value">{len(df_clean)}</div>
            <div class="label">Total Samples</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_ds2:
        diabetic_count = (df_clean['Outcome'] == 1).sum()
        non_diabetic_count = (df_clean['Outcome'] == 0).sum()
        st.markdown(f"""
        <div class="metric-card">
            <div class="value">{diabetic_count}</div>
            <div class="label">Positive Cases</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_ds3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="value">{non_diabetic_count}</div>
            <div class="label">Negative Cases</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_ds4:
        ratio = non_diabetic_count / diabetic_count
        st.markdown(f"""
        <div class="metric-card">
            <div class="value">{ratio:.1f}:1</div>
            <div class="label">Imbalance Ratio</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    # Data preview
    st.markdown("""
    <div class="card">
        <h3>Data Preview</h3>
        <p style="color: #718096; font-size: 0.85rem; margin: 0;">
        First 10 records from the cleaned dataset. Zero values in medical measurements have been 
        imputed using the median value of the respective feature.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.dataframe(df_clean.head(10), width='stretch', hide_index=True)
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    # Class distribution
    st.markdown("""
    <div class="card">
        <h3>Class Distribution Analysis</h3>
    </div>
    """, unsafe_allow_html=True)
    
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        fig_dist, ax_dist = plt.subplots(figsize=(5.5, 3.5))
        colors_dist = ['#2b6cb0', '#c53030']
        labels_dist = ['Negative (Non-Diabetic)', 'Positive (Diabetic)']
        values_dist = [non_diabetic_count, diabetic_count]
        bars_dist = ax_dist.bar(labels_dist, values_dist, color=colors_dist, width=0.5, edgecolor='white')
        for bar, val in zip(bars_dist, values_dist):
            ax_dist.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 8, 
                        f'{val} ({val/len(df_clean)*100:.1f}%)', ha='center', fontsize=9, fontweight='bold')
        ax_dist.set_ylabel('Count', fontsize=10)
        ax_dist.set_title('Class Distribution', fontsize=11, fontweight='bold', color='#1a365d')
        ax_dist.spines['top'].set_visible(False)
        ax_dist.spines['right'].set_visible(False)
        ax_dist.set_ylim(0, max(values_dist) * 1.25)
        ax_dist.tick_params(colors='#4a5568')
        plt.tight_layout()
        st.pyplot(fig_dist)
    
    with col_chart2:
        selected_feat = st.selectbox(
            "Select Feature for Distribution Analysis:",
            options=['Glucose', 'BMI', 'Age', 'BloodPressure', 'Insulin', 'DiabetesPedigreeFunction'],
            index=0
        )
        
        fig_kde, ax_kde = plt.subplots(figsize=(5.5, 3.5))
        for cls, color, label in zip([0, 1], ['#2b6cb0', '#c53030'], ['Negative', 'Positive']):
            sns.kdeplot(df_clean[df_clean['Outcome']==cls][selected_feat], 
                       ax=ax_kde, label=label, color=color, fill=True, alpha=0.25, linewidth=2)
        ax_kde.set_title(f'{selected_feat} Distribution by Class', fontsize=11, fontweight='bold', color='#1a365d')
        ax_kde.set_xlabel(selected_feat, fontsize=10)
        ax_kde.set_ylabel('Density', fontsize=10)
        ax_kde.legend(fontsize=9)
        ax_kde.spines['top'].set_visible(False)
        ax_kde.spines['right'].set_visible(False)
        ax_kde.tick_params(colors='#4a5568')
        plt.tight_layout()
        st.pyplot(fig_kde)
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    # Summary statistics
    st.markdown("""
    <div class="card">
        <h3>Descriptive Statistics</h3>
        <p style="color: #718096; font-size: 0.85rem; margin: 0;">
        Summary statistics for all diagnostic features in the dataset.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    desc_df = df_clean.drop('Outcome', axis=1).describe().round(2)
    st.dataframe(desc_df, width='stretch')

# ============================================================
# TAB 4: ASSESSMENT HISTORY
# ============================================================
with tab4:
    st.markdown("""
    <div class="card">
        <h3>Assessment History</h3>
        <p style="color: #718096; font-size: 0.85rem; margin: 0;">
        Previously computed patient risk assessments are stored locally and displayed below.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    if len(st.session_state.history) > 0:
        # Display summary metrics
        hist = st.session_state.history
        total = len(hist)
        high_risk_count = len(hist[hist['Risk Level'] == 'High Risk']) if 'Risk Level' in hist.columns else 0
        low_risk_count = len(hist[hist['Risk Level'] == 'Low Risk']) if 'Risk Level' in hist.columns else 0
        
        col_h1, col_h2, col_h3 = st.columns(3)
        with col_h1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="value">{total}</div>
                <div class="label">Total Assessments</div>
            </div>
            """, unsafe_allow_html=True)
        with col_h2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="value">{high_risk_count}</div>
                <div class="label">High Risk Cases</div>
            </div>
            """, unsafe_allow_html=True)
        with col_h3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="value">{low_risk_count}</div>
                <div class="label">Low Risk Cases</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        
        # Display history table
        display_cols = ['Date', 'Disease', 'Risk Level', 'Probability', 'Prediction', 'Model', 'Glucose', 'BMI']
        display_cols = [c for c in display_cols if c in hist.columns]
        
        st.dataframe(
            hist[display_cols].sort_values('Date', ascending=False),
            width='stretch',
            hide_index=True
        )
        
        # Option to clear history
        if st.button("Clear Assessment History"):
            if os.path.exists(HISTORY_FILE):
                os.remove(HISTORY_FILE)
            st.session_state.history = pd.DataFrame()
            st.rerun()
    else:
        st.markdown("""
        <div class="card" style="text-align: center; padding: 2rem;">
            <h3 style="color: #718096;">No Assessment History</h3>
            <p style="color: #a0aec0; font-size: 0.9rem;">
            Completed patient assessments will appear here. Use the Patient Assessment tab to generate a new risk analysis.
            </p>
        </div>
        """, unsafe_allow_html=True)

# ============================================================
# FOOTER
# ============================================================
st.markdown("""
<div class="footer">
    <p><strong>Clinical Disease Risk Assessment System v2.0</strong> | Built with Streamlit, Scikit-Learn, SHAP, and FPDF2 | Pima Indians Diabetes Database</p>
    <p style="font-size: 0.75rem;">This system is intended for educational and research purposes only. It is not a substitute for professional medical diagnosis or clinical decision-making.</p>
</div>
""", unsafe_allow_html=True)