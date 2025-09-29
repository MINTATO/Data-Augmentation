#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import StandardScaler


# ---------------------------
df = pd.read_excel('C:/Users/yousf/OneDrive/Desktop/Preprocessed_Perovskite_Dataset.xlsx')
data = df.to_numpy()
data_dim = data.shape[1]  


# ---------------------------
def build_kde_model(bandwidth=0.2, kernel='gaussian', random_state=42):
    scaler = StandardScaler()
    X = scaler.fit_transform(data)
    kde = KernelDensity(bandwidth=bandwidth, kernel=kernel)
    kde.fit(X)
    return scaler, kde

scaler, kde = build_kde_model(bandwidth=0.2, kernel='gaussian', random_state=42)

# ---------------------------
def train_kde(data, epochs=1000, batch_size=32, density_quantile=0.05):
    X = scaler.transform(data)
    train_log_density = kde.score_samples(X)
    min_log_density = float(np.quantile(train_log_density, density_quantile))

    for epoch in range(epochs):
        if epoch % 100 == 0:
            neg_ll = float(-np.mean(train_log_density))  
            print(f"Epoch {epoch} / {epochs}, KDE NegLL: {neg_ll:.4f}, Density threshold: {min_log_density:.4f}")

    return min_log_density

min_log_density = train_kde(data, epochs=1000, batch_size=32, density_quantile=0.05)

# ---------------------------
num_samples = 100
rng = np.random.RandomState(42)

def kde_sample(n, bandwidth=0.2):
    try:
        Xs = kde.sample(n_samples=n, random_state=rng)
    except Exception:
        X = scaler.transform(data)
        idx = rng.randint(0, X.shape[0], size=n)
        Xs = X[idx] + rng.normal(scale=bandwidth, size=(n, X.shape[1]))
    return Xs


needed = num_samples
synthetic_list = []
max_batches = 20
bandwidth_used = 0.2  

while needed > 0 and max_batches > 0:
    draw_n = int(np.ceil(needed * 1.8))
    Xs = kde_sample(draw_n, bandwidth=bandwidth_used)

    
    logd = kde.score_samples(Xs)
    Xs_kept = Xs[logd >= min_log_density]

   
    kept = scaler.inverse_transform(Xs_kept)

    
    for j in range(data.shape[1]):
        mn, mx = data[:, j].min(), data[:, j].max()
        kept[:, j] = np.clip(kept[:, j], mn, mx)

    take = min(needed, kept.shape[0])
    if take > 0:
        synthetic_list.append(kept[:take])
        needed -= take

    max_batches -= 1


if needed > 0:
    Xs = kde_sample(needed, bandwidth=bandwidth_used)
    kept = scaler.inverse_transform(Xs)
    for j in range(data.shape[1]):
        mn, mx = data[:, j].min(), data[:, j].max()
        kept[:, j] = np.clip(kept[:, j], mn, mx)
    synthetic_list.append(kept)

synthetic_data = np.vstack(synthetic_list) if synthetic_list else np.empty((0, data.shape[1]))

# ---------------------------
synthetic_df = pd.DataFrame(synthetic_data, columns=df.columns)
output_synthetic_path = 'C:/Users/yousf/Macquarie University/Jincheol Kim - 2024_PACE/00_Project without image data/Data/P_KDE.xlsx'
synthetic_df.to_excel(output_synthetic_path, index=False)

print(f"KDE synthetic data saved to: {output_synthetic_path}")

