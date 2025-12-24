# -*- coding: utf-8 -*-
"""
Created on Sun Mar 16 15:48:04 2025

@author: H.A.R
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import umap

# Set style for publication-ready plots
sns.set(style="ticks", context="paper", font_scale=1.2)
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
plt.rcParams['pdf.fonttype'] = 42  # Ensure editable text in PDF

# Load data
df = pd.read_csv("E:/Bras/4_Species.csv")

# Extract metadata and codon frequencies
species_names = df.iloc[:, 1]  # Species column
codon_columns = df.columns[3:] 

# Normalize codon frequencies (relative frequencies)
df_codons = df[codon_columns].div(df[codon_columns].sum(axis=1), axis=0)

# UMAP model with optimized parameters
umap_model = umap.UMAP(
    n_neighbors=50,
    min_dist=0.1,
    n_components=2,
    random_state=42,
    metric='euclidean',
    n_jobs=-1
)
umap_results = umap_model.fit_transform(df_codons)

# Create figure with adjustable dimensions
fig = plt.figure(figsize=(7, 5))  # Width, height in inches (adjust for Overleaf)
ax = fig.add_subplot(111)

# Hexbin plot for density visualization
hexbin = ax.hexbin(
    umap_results[:, 0],
    umap_results[:, 1],
    gridsize=150,
    cmap='viridis',
    mincnt=1,
    bins='log',
    alpha=0.8
)

# Add colorbar
cbar = fig.colorbar(hexbin, ax=ax, pad=0.02)
cbar.set_label('Gene density', rotation=270, labelpad=15)

# Axis labels
ax.set_xlabel("UMAP 1", labelpad=10)
ax.set_ylabel("UMAP 2", labelpad=10)

# Remove redundant borders
sns.despine(trim=True)

# Adjust layout and save
plt.tight_layout(pad=2.0)

# Save as vector graphic (for Overleaf)
plt.savefig(
    "umap_plot.pdf",
    format="pdf",
    dpi=300,
    bbox_inches="tight",
    transparent=True
)

plt.show()