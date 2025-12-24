# -*- coding: utf-8 -*-
"""
Created on Sun Mar 16 18:10:49 2025

@author: H.A.R
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

# Configure plot settings for publications
sns.set(style="ticks", context="paper", font_scale=1.2)
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
plt.rcParams['pdf.fonttype'] = 42  # Embed fonts in PDF

# Load data
file_path = "E:/Bras/4_Species.csv"
df = pd.read_csv(file_path)

# Extract metadata and codon frequencies
species_names = df.iloc[:, 1]  # Species labels column
codon_columns = df.columns[3:]
df_codons = df[codon_columns].div(df[codon_columns].sum(axis=1), axis=0)  # Normalize

# Run t-SNE on full data (Barnes-Hut approximation for scalability)
tsne = TSNE(
    n_components=2,
    perplexity=30,
    n_iter=1000,
    init='pca',
    random_state=42,
    method='barnes_hut',  # Required for large datasets
    n_jobs=-1,
    verbose=1  # Monitor progress
)
tsne_results = tsne.fit_transform(df_codons)

# Create figure
fig = plt.figure(figsize=(7, 5))
ax = fig.add_subplot(111)

# Hexbin plot for density visualization
hexbin = ax.hexbin(
    tsne_results[:, 0],
    tsne_results[:, 1],
    gridsize=150,
    cmap='viridis',
    mincnt=1,
    bins='log',
    alpha=0.8
)

# Formatting
cbar = fig.colorbar(hexbin, ax=ax, pad=0.02)
cbar.set_label('Gene density', rotation=270, labelpad=15)
ax.set_xlabel("t-SNE 1", labelpad=10)
ax.set_ylabel("t-SNE 1", labelpad=10)
sns.despine(trim=True)
plt.tight_layout(pad=2.0)

# Save vector/raster formats for Overleaf
plt.savefig("E:/Bras/visual/tsne_plot.pdf", format="pdf", dpi=300, bbox_inches="tight")
plt.savefig("E:/Bras/visual/tsne_plot.png", format="png", dpi=600, bbox_inches="tight")
plt.close()