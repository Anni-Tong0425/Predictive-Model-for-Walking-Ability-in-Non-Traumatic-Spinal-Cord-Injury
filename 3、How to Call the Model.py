# -*- coding: utf-8 -*-
"""
Documentation Generator for Bernoulli Naive Bayes Model
Generates two documents:
1. Model Usage Guide
2. Model Parameter Documentation
"""

import os
import json
from datetime import datetime


def generate_model_usage_guide():
    """Generate model usage guide document"""

    content = f"""Bernoulli Naive Bayes Model - Usage Guide
==============================================

Document Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

1. OVERVIEW
-----------
The Bernoulli Naive Bayes model is a machine learning classifier designed for 
binary classification tasks. This specific implementation is trained to predict 
whether spinal cord injury patients can walk based on clinical features.

The model uses Bernoulli Naive Bayes algorithm, which is particularly suited 
for binary/boolean features. All continuous features are binarized using an 
optimal threshold determined during training.

2. PREREQUISITES
----------------
- Python 3.7 or higher
- Required Python packages:
  * scikit-learn >= 1.0.0
  * pandas >= 1.3.0
  * numpy >= 1.21.0
  * matplotlib >= 3.5.0
  * seaborn >= 0.11.0
  * shap >= 0.41.0 (for model interpretability)

3. MODEL FILES STRUCTURE
------------------------
After training, the following files and folders are generated:

3.1 Saved Models:
----------------
- `Saved_Models/Bernoulli_Naive_Bayes_Model.pkl` - Main model file (joblib format)
- `Saved_Models/Bernoulli_Naive_Bayes_Model_full.pkl` - Complete model data (pickle format)

3.2 Visualization Results:
-------------------------
- `Model_Visualization_Results/`
  * `ROC_Curve.png` - Receiver Operating Characteristic curve
  * `Precision_Recall_Curve.png` - Precision-Recall curve
  * `Confusion_Matrix.png` - Confusion matrix
  * `Feature_Importance_EN.png` - Feature importance plot (English)
  * `Feature_Importance.csv` - Feature importance data in CSV format

3.3 SHAP Analysis Results:
-------------------------
- `SHAP_Analysis_Results/`
  * `SHAP_Summary_Plot.png` - SHAP summary plot
  * `SHAP_Bar_Plot.png` - SHAP feature importance with numerical values
  * `SHAP_Bar_Plot_Standard.png` - Standard SHAP bar plot
  * `SHAP_Dependence_*.png` - SHAP dependence plots for top features
  * `SHAP_Force_Plot_*.png/.html` - SHAP force plots for 7 representative cases
  * `SHAP_Waterfall_Plot_*.png/.html` - SHAP waterfall plots for 7 representative cases
  * `SHAP_Beeswarm_Plot.png` - SHAP beeswarm plot
  * `SHAP_Decision_Plot.png` - SHAP decision plot
  * `*.npy` - SHAP values data files
  * `SHAP_Analysis_Report.txt` - Detailed SHAP analysis report

4. LOADING THE MODEL
--------------------
There are two ways to load the saved model:

Method 1: Using joblib (recommended)
"""
import joblib

# Load the model
model_data = joblib.load('Saved_Models/Bernoulli_Naive_Bayes_Model.pkl')

# Extract components
model = model_data['model']
scaler = model_data['scaler']
binarizer = model_data['binarizer']
features = model_data['features']

"""
Method 2: Using pickle
import pickle

with open('Saved_Models/Bernoulli_Naive_Bayes_Model_full.pkl', 'rb') as f:
    model_data = pickle.load(f)
"""
