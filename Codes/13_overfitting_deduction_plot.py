# -*- coding: utf-8 -*-
"""
Created on Tue Jun 24 17:27:20 2025

@author: H.A.R
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import math

def safe_load_histories(file_path, model_name=""):
    data = np.load(file_path, allow_pickle=True)

    if isinstance(data, np.ndarray) and data.shape == ():  
        data = data.item()

    if isinstance(data, dict):
        if all(isinstance(v, list) and len(v) == 10 for v in data.values()):
            histories = []
            for i in range(10):
                histories.append({
                    'loss':         data['loss'][i],
                    'val_loss':     data['val_loss'][i],
                    'accuracy':     data['accuracy'][i],
                    'val_accuracy': data['val_accuracy'][i]
                })
            return histories
        else:
            return [data]

    while isinstance(data, (list, np.ndarray)) and len(data) == 1:
        data = data[0]

    if isinstance(data, (list, np.ndarray)) and isinstance(data[0], dict):
        return list(data)

    raise ValueError(f"Unknown format in history file for model: {model_name}")

# === Overfitting plot section ===

base_dir = r"E:\Bras_article\models"
model_names = os.listdir(base_dir)
model_names = [m for m in model_names if os.path.isdir(os.path.join(base_dir, m))]

cols = 2
rows = math.ceil(len(model_names) / cols)

fig, axes = plt.subplots(rows, cols, figsize=(14, 4 * rows))
axes = axes.flatten()
plot_idx = 0

for model_name in model_names:
    model_path = os.path.join(base_dir, model_name)

    # Find history file
    history_file = None
    for fname in os.listdir(model_path):
        if fname.lower() in ['histories.npy', 'training_history.npy']:
            history_file = os.path.join(model_path, fname)
            break

    if not history_file:
        print(f"No history file for {model_name}. Skipping.")
        continue

    try:
        histories = safe_load_histories(history_file, model_name)
        min_len = min(len(h['accuracy']) for h in histories)
        all_train_acc = np.array([h['accuracy'][:min_len] for h in histories])
        all_val_acc   = np.array([h['val_accuracy'][:min_len] for h in histories])
    except Exception as e:
        print(f"Failed to load {model_name}: {e}")
        continue

    # Compute average overfitting gap
    acc_gap = np.mean(all_train_acc - all_val_acc, axis=0)

    # Plot
    ax = axes[plot_idx]
    plot_idx += 1
    epochs = np.arange(1, len(acc_gap) + 1)
    ax.plot(epochs, acc_gap, label="Train - Val Accuracy Gap", color='red')
    ax.axhline(0, linestyle='--', color='gray', linewidth=1)
    ax.set_title(f"{model_name} - Overfitting Detection")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy Gap")
    ax.grid(True)
    ax.legend()

# Hide unused axes
for i in range(plot_idx, len(axes)):
    fig.delaxes(axes[i])

plt.tight_layout()
plt.show()
