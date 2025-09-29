#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np

# ---------------------------
df = pd.read_excel('C:/Users/yousf/OneDrive/Desktop/Preprocessed_Perovskite_Dataset.xlsx')
data = df.to_numpy()
data_dim = data.shape[1]  


# ---------------------------
def build_noise_model(noise_scale=0.01, random_state=42):
    rng = np.random.RandomState(random_state)
    feature_std = data.std(axis=0, ddof=1)           
    feature_min = data.min(axis=0)                   
    feature_max = data.max(axis=0)                   
    return {
        "rng": rng,
        "noise_scale": float(noise_scale),
        "feature_std": feature_std,
        "feature_min": feature_min,
        "feature_max": feature_max
    }

noise_cfg = build_noise_model(noise_scale=0.01, random_state=42)


# ---------------------------
def train_noise(data, epochs=1000, batch_size=32):
    proxy = float(np.mean(noise_cfg["feature_std"]))
    for epoch in range(epochs):
        if epoch % 100 == 0:
            print(f"Epoch {epoch} / {epochs}, Noise proxy (mean σ): {proxy:.6f}, noise_scale: {noise_cfg['noise_scale']:.4f}")
    return proxy  

_ = train_noise(data, epochs=1000, batch_size=32)


# ---------------------------
num_samples = 100
rng = noise_cfg["rng"]
sigma = noise_cfg["feature_std"] * noise_cfg["noise_scale"]

synthetic_rows = []
for _ in range(num_samples):
    base = data[rng.randint(0, data.shape[0])]          
    noise = rng.normal(loc=0.0, scale=sigma, size=data_dim)
    synth = base + noise
    # clip to original range (physical sanity)
    synth = np.clip(synth, noise_cfg["feature_min"], noise_cfg["feature_max"])
    synthetic_rows.append(synth)

synthetic_data = np.vstack(synthetic_rows)


# ---------------------------
synthetic_df = pd.DataFrame(synthetic_data, columns=df.columns)
output_synthetic_path = 'C:/Users/yousf/Macquarie University/Jincheol Kim - 2024_PACE/00_Project without image data/Data/P_Noise.xlsx'
synthetic_df.to_excel(output_synthetic_path, index=False)

print(f"Noise-injection synthetic data saved to: {output_synthetic_path}")

