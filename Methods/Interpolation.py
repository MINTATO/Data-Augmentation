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
def infer_groups(df: pd.DataFrame):
    cols_lower = [c.lower() for c in df.columns]
    # direct architecture-like column
    for i, c in enumerate(cols_lower):
        if ('architecture' in c) or (c == 'arch') or c.endswith('_arch'):
            return df.iloc[:, i].values  # categorical/numeric ok
    # one-hot style detection (pin/nip)
    pin_idx = [i for i, c in enumerate(cols_lower) if ('pin' in c.replace('-', '')) and ('spin' not in c)]
    nip_idx = [i for i, c in enumerate(cols_lower) if ('nip' in c.replace('-', ''))]
    if pin_idx and nip_idx:
        pin_v = df.iloc[:, pin_idx[0]].values.astype(float)
        nip_v = df.iloc[:, nip_idx[0]].values.astype(float)
        return np.where(pin_v >= nip_v, 'PIN', 'NIP')
    return None  # no grouping available

group_ids = infer_groups(df)


# ---------------------------
def build_interp_model(mode='linear', random_state=42):
    rng = np.random.RandomState(random_state)
    return {"rng": rng, "mode": mode}

interp_cfg = build_interp_model(mode='linear', random_state=42)


# ---------------------------
def train_interpolation(data, epochs=1000, batch_size=32):
    rng = interp_cfg["rng"]
    if data.shape[0] > batch_size:
        idx = rng.choice(data.shape[0], size=batch_size, replace=False)
        batch = data[idx]
    else:
        batch = data

    if batch.shape[0] > 1:
        diffs = batch[:, None, :] - batch[None, :, :]
        dists = np.sqrt(np.sum(diffs * diffs, axis=-1))
        proxy = float(np.mean(dists[np.triu_indices_from(dists, k=1)]))
    else:
        proxy = 0.0

    for epoch in range(epochs):
        if epoch % 100 == 0:
            print(f"Epoch {epoch} / {epochs}, Interp proxy (mean pairwise dist): {proxy:.6f}, mode: {interp_cfg['mode']}")
    return proxy

_ = train_interpolation(data, epochs=1000, batch_size=32)


# ---------------------------
num_samples = 100
rng = interp_cfg["rng"]

feature_min = data.min(axis=0)
feature_max = data.max(axis=0)

def sample_pair_indices(n_rows, group_ids, rng):
    if group_ids is None:
        i = rng.randint(0, n_rows)
        j = rng.randint(0, n_rows)
        while j == i:
            j = rng.randint(0, n_rows)
        return i, j
    # group-aware sampling
    # pick an anchor index, then pick j from same group if any
    i = rng.randint(0, n_rows)
    gi = group_ids[i]
    same = np.where(group_ids == gi)[0]
    if same.size > 1:
        j = rng.choice(same)
        while j == i:
            j = rng.choice(same)
        return i, j
    # fallback: global different index
    j = rng.randint(0, n_rows)
    while j == i:
        j = rng.randint(0, n_rows)
    return i, j

synthetic_rows = []
for _ in range(num_samples):
    i, j = sample_pair_indices(data.shape[0], group_ids, rng)
    alpha = rng.rand()  # U[0,1]
    synth = alpha * data[i] + (1.0 - alpha) * data[j]
    # clip per feature (physical sanity, keeps inside observed bounds)
    synth = np.clip(synth, feature_min, feature_max)
    synthetic_rows.append(synth)

synthetic_data = np.vstack(synthetic_rows)


# ---------------------------
synthetic_df = pd.DataFrame(synthetic_data, columns=df.columns)
output_synthetic_path = 'C:/Users/yousf/Macquarie University/Jincheol Kim - 2024_PACE/00_Project without image data/Data/P_Interp.xlsx'
synthetic_df.to_excel(output_synthetic_path, index=False)

print(f"Interpolation synthetic data saved to: {output_synthetic_path}")

