#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lightgbm import LGBMRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, mean_absolute_error, r2_score


# ---------------------------
file_path = 'C:/Users/yousf/OneDrive/Desktop/new_dataset/Book1.xlsx'
excel_data = pd.ExcelFile(file_path)

df = excel_data.parse('Sheet1')

features = df[['JV_default_Voc', 'JV_default_Jsc', 'JV_default_FF']].dropna()
target = df['JV_default_PCE'].dropna()

common_index = features.index.intersection(target.index)
X = features.loc[common_index]
y = target.loc[common_index]

# ---------------------------
lgbm_model = LGBMRegressor(
    objective='regression',
    n_estimators=300,       
    learning_rate=0.05,
    num_leaves=31,
    random_state=42
)

# ---------------------------
mae_scorer = make_scorer(mean_absolute_error)
r2_scorer = make_scorer(r2_score)

mae_results = []
r2_results = []
dataset_sizes = np.linspace(0.2, 1.0, 5)  # 20%, 40%, 60%, 80%, 100%

# ---------------------------
for size in dataset_sizes:
    subset_X = X.sample(frac=size, random_state=42)
    subset_y = y[subset_X.index]
    
    mae_cv = cross_val_score(lgbm_model, subset_X, subset_y, cv=5, scoring=mae_scorer)
    r2_cv = cross_val_score(lgbm_model, subset_X, subset_y, cv=5, scoring=r2_scorer)
    
    mae_results.append(np.mean(mae_cv))
    r2_results.append(np.mean(r2_cv))
    
    print(f"[LightGBM][frac={size:.1f}]  "
          f"MAE: {np.mean(mae_cv):.4f} ± {np.std(mae_cv):.4f} | "
          f"R^2: {np.mean(r2_cv):.4f} ± {np.std(r2_cv):.4f}")

# ---------------------------
fig, ax1 = plt.subplots(figsize=(8, 6))

ax1.plot(dataset_sizes, mae_results, 'bo-', label="MAE")
ax1.set_xlabel("Number of dataset")
ax1.set_ylabel("MAE", color="blue")
ax1.tick_params(axis='y', labelcolor="blue")

ax2 = ax1.twinx()
ax2.plot(dataset_sizes, r2_results, 'ro-', label="R^2")
ax2.set_ylabel("R^2", color="red")
ax2.tick_params(axis='y', labelcolor="red")

plt.title("LightGBM – MAE and R^2 vs. Dataset Size")
plt.show()

