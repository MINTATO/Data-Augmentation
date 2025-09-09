# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 09:08:41 2024

@author: yousf
"""

# Import necessary libraries
import pandas as pd
import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

# Load the dataset
file_path = 'C:/Users/yousf/Macquarie University/Jincheol Kim - 2024_PACE/00_Project without image data/Data/Perovsite database query - 3D.xlsx'
data = pd.ExcelFile(file_path)
df = data.parse('Perovsite database query - 3D')

# Select features based on the heatmap
heatmap_features = [
    "JV_default_Voc",
    "JV_default_Jsc",
    "JV_default_FF",
    "JV_default_PCE",  # Target variable
    "Perovskite_composition_a_ions",
    "Perovskite_composition_b_ions",
    "Perovskite_composition_c_ions",
    "Perovskite_deposition_procedure",
    "Perovskite_deposition_solvents",
    "Perovskite_thickness",
    "ETL_thickness",
    "HTL_thickness_list",
    "Perovskite_additives_compounds",
    "Perovskite_additives_concentrations",
    "Perovskite_deposition_solvents_mixing_ratios",
    "JV_test_atmosphere",
]

# Filter and clean the dataset
df_heatmap = df[heatmap_features].dropna()

# Ensure consistent data types
for col in df_heatmap.columns:
    if df_heatmap[col].dtype == 'object':
        df_heatmap[col] = df_heatmap[col].astype(str)

# Encode categorical variables
for col in df_heatmap.columns:
    if df_heatmap[col].dtype == 'object':
        le = LabelEncoder()
        df_heatmap[col] = le.fit_transform(df_heatmap[col])

# Split data into features and target
X_heatmap = df_heatmap.drop("JV_default_PCE", axis=1)
y_heatmap = df_heatmap["JV_default_PCE"]

# Train-test split
X_train_hm, X_test_hm, y_train_hm, y_test_hm = train_test_split(
    X_heatmap, y_heatmap, test_size=0.2, random_state=42
)

# Prepare data for XGBoost
dtrain_hm = xgb.DMatrix(X_train_hm, label=y_train_hm)
dtest_hm = xgb.DMatrix(X_test_hm, label=y_test_hm)

# Define XGBoost model parameters
params_hm = {
    "objective": "reg:squarederror",
    "max_depth": 6,
    "learning_rate": 0.1,
}

# Train the XGBoost model with reduced boosting rounds
model_hm = xgb.train(params_hm, dtrain_hm, num_boost_round=20)

# Perform SHAP analysis for feature importance
explainer_hm = shap.TreeExplainer(model_hm)
shap_values_hm = explainer_hm.shap_values(X_test_hm)

# Generate SHAP summary plot with adjusted x-axis limits
import matplotlib.pyplot as plt

# Modify SHAP summary plot
shap.summary_plot(shap_values_hm, X_test_hm, show=False)
plt.xlim(-1.5, 1)  # Adjust x-axis limits
plt.show()

