# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 03:13:31 2024

@author: yousf
"""

# -*- coding: utf-8 -*-
"""

 5-fold cross-validations of (a) XGBoost, (c) Neural Net, and  Averaged MAE (left axis) and R2  axis) values as a function of a number of data used in the training set.
(right 
"""


import pandas as pd
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import numpy as np


import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, mean_absolute_error, r2_score
from xgboost import XGBRegressor
import matplotlib.pyplot as plt

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import numpy as np
# Load the provided Excel file to examine its contents
file_path = 'C:/Users/yousf/OneDrive/Desktop/new_dataset/Book1.xlsx'
excel_data = pd.ExcelFile(file_path)

# Load the specific sheet into a DataFrame
df = excel_data.parse('Sheet1')

# Select relevant features and target
features = df[['JV_default_Voc', 'JV_default_Jsc', 'JV_default_FF']].dropna()
target = df['JV_default_PCE'].dropna()

# Ensure features and target have matching indices
common_index = features.index.intersection(target.index)
X = features.loc[common_index]
y = target.loc[common_index]

# Initialize XGBoost Regressor
xgb_model = XGBRegressor(random_state=42, n_estimators=100)

# Define scoring metrics
mae_scorer = make_scorer(mean_absolute_error)
r2_scorer = make_scorer(r2_score)

# Lists to store results
mae_results = []
r2_results = []
dataset_sizes = np.linspace(0.2, 1.0, 5)  # 20%, 40%, 60%, 80%, 100% of the dataset

# Perform cross-validation on different dataset sizes
for size in dataset_sizes:
    # Select a subset of the data
    subset_X = X.sample(frac=size, random_state=42)
    subset_y = y[subset_X.index]
    
    # Perform 5-fold cross-validation for MAE and R2
    mae_cv = cross_val_score(xgb_model, subset_X, subset_y, cv=5, scoring=mae_scorer)
    r2_cv = cross_val_score(xgb_model, subset_X, subset_y, cv=5, scoring=r2_scorer)
    
    # Store the mean of the MAE and R2 scores
    mae_results.append(np.mean(mae_cv))
    r2_results.append(np.mean(r2_cv))

# Plotting MAE and R^2 scores as a function of dataset size
fig, ax1 = plt.subplots(figsize=(8, 6))

# Plot MAE on the left y-axis
ax1.plot(dataset_sizes, mae_results, 'bo-', label="MAE")
ax1.set_xlabel("Number of dataset")
ax1.set_ylabel("MAE", color="blue")
ax1.tick_params(axis='y', labelcolor="blue")

# Plot R^2 on the right y-axis
ax2 = ax1.twinx()
ax2.plot(dataset_sizes, r2_results, 'ro-', label="R^2")
ax2.set_ylabel("R^2", color="red")
ax2.tick_params(axis='y', labelcolor="red")

# Show plot
plt.title("MAE and R^2 vs. Dataset Size")
plt.show()

# import pandas as pd
# from sklearn.model_selection import cross_val_predict
# from sklearn.metrics import mean_absolute_error, r2_score
# from sklearn.neural_network import MLPRegressor
# import matplotlib.pyplot as plt
# import numpy as np

# # Load the provided Excel file to examine its contents
# file_path = 'D:/PACE Project/01_Material data project/Data/All_data.xlsx'
# excel_data = pd.ExcelFile(file_path)

# # Checking the sheet names to understand the structure
# sheet_names = excel_data.sheet_names
# sheet_names
# # Load the specific sheet into a DataFrame to explore its content
# df = excel_data.parse('Perovsite database query - 3D')


# # Select relevant features and target
# features = df[['JV_default_Voc', 'JV_default_Jsc', 'JV_default_FF']].dropna()
# target = df['JV_default_PCE'].dropna()

# # Ensure features and target have matching indices
# common_index = features.index.intersection(target.index)
# X = features.loc[common_index]
# y = target.loc[common_index]

# # Initialize the Neural Network Regressor
# nn_model = MLPRegressor(hidden_layer_sizes=(50, 50), max_iter=1000, random_state=42)

# # Perform 5-fold cross-validation and get predictions
# y_pred = cross_val_predict(nn_model, X, y, cv=5)

# # Calculate MAE and R^2 scores
# mae = mean_absolute_error(y, y_pred)
# r2 = r2_score(y, y_pred)

# # Plotting predicted vs actual values
# plt.figure(figsize=(8, 6))
# plt.scatter(y, y_pred, color="blue", alpha=0.5, edgecolor='k')
# plt.plot([min(y), max(y)], [min(y), max(y)], color="green", linestyle="--")
# plt.xlabel("Given (actual)")
# plt.ylabel("Predicted")
# plt.title("5-Fold Cross-Validation for Neural Network Model")
# plt.text(min(y)+2, max(y)-3, f"MAE = {mae:.3f}\nR2 = {r2:.3f}", fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
# plt.show()

# import pandas as pd
# from sklearn.model_selection import cross_val_predict
# from sklearn.metrics import mean_absolute_error, r2_score
# from sklearn.gaussian_process import GaussianProcessRegressor
# from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
# import matplotlib.pyplot as plt
# import numpy as np

# # Load the provided Excel file to examine its contents
# file_path = 'D:/PACE Project/01_Material data project/Data/All_data.xlsx'
# excel_data = pd.ExcelFile(file_path)

# # Checking the sheet names to understand the structure
# sheet_names = excel_data.sheet_names
# sheet_names
# # Load the specific sheet into a DataFrame to explore its content
# df = excel_data.parse('Perovsite database query - 3D')

# # Select relevant features and target
# features = df[['JV_default_Voc', 'JV_default_Jsc', 'JV_default_FF']].dropna()
# target = df['JV_default_PCE'].dropna()

# # Ensure features and target have matching indices
# common_index = features.index.intersection(target.index)
# X = features.loc[common_index]
# y = target.loc[common_index]

# # Define the kernel for the Gaussian Process Regressor
# kernel = C(1.0) * RBF(length_scale=1.0)

# # Initialize the Gaussian Process Regressor
# gpr_model = GaussianProcessRegressor(kernel=kernel, random_state=42)

# # Perform 5-fold cross-validation and get predictions
# y_pred = cross_val_predict(gpr_model, X, y, cv=5)

# # Calculate MAE and R^2 scores
# mae = mean_absolute_error(y, y_pred)
# r2 = r2_score(y, y_pred)

# # Plotting predicted vs actual values
# plt.figure(figsize=(8, 6))
# plt.scatter(y, y_pred, color="blue", alpha=0.5, edgecolor='k')
# plt.plot([min(y), max(y)], [min(y), max(y)], color="green", linestyle="--")
# plt.xlabel("Given (actual)")
# plt.ylabel("Predicted")
# plt.title("5-Fold Cross-Validation for Gaussian Process Regressor (GPR)")
# plt.text(min(y)+2, max(y)-3, f"MAE = {mae:.3f}\nR^2 = {r2:.3f}", fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
# plt.show()