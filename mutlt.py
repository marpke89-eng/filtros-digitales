import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import MDS

# 1. Generate Synthetic Matrix & MDS Coordinates
np.random.seed(42)
drinks = [
    "Agua", "Refresco de cola", "Jugo natural", "Bebida energética",
    "Té embotellado", "Café frío", "Agua con gas", "Leche saborizada"
]
n = len(drinks)

# Dissimilarity matrix simulation (5 = very similar, 1 = very different -> Dissimilarity = 6 - Similarity)
# Let's define realistic perceptual similarities
# Group 1: Agua, Agua con gas
# Group 2: Refresco de cola, Bebida energética
# Group 3: Jugo natural, Té embotellado
# Group 4: Leche saborizada, Café frío (or unique)

# Let's use sklearn MDS on a predefined dissimilarity matrix
sim_matrix = np.array([
    # Ag,  Ref, Jug, Ene, Té,  Caf, AgG, Lec
    [5.0, 1.2, 3.1, 1.1, 3.5, 1.8, 4.7, 1.5], # Agua
    [1.2, 5.0, 1.5, 4.2, 2.1, 3.0, 1.8, 2.0], # Refresco de cola
    [3.1, 1.5, 5.0, 1.8, 3.8, 2.2, 2.8, 2.5], # Jugo natural
    [1.1, 4.2, 1.8, 5.0, 2.0, 3.2, 1.5, 1.6], # Bebida energética
    [3.5, 2.1, 3.8, 2.0, 5.0, 2.9, 3.1, 2.2], # Té embotellado
    [1.8, 3.0, 2.2, 3.2, 2.9, 5.0, 1.9, 3.8], # Café frío
    [4.7, 1.8, 2.8, 1.5, 3.1, 1.9, 5.0, 1.4], # Agua con gas
    [1.5, 2.0, 2.5, 1.6, 2.2, 3.8, 1.4, 5.0]  # Leche saborizada
])

dissimilarity = 6.0 - sim_matrix
mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
coords = mds.fit_transform(dissimilarity)

# Plotting MDS result
plt.figure(figsize=(8, 6), dpi=300)
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#17becf', '#e377c2']
categories = ['Hidratación Básica', 'Estimulantes/Carbonatadas', 'Saludables/Naturales', 'Estimulantes/Carbonatadas', 
              'Saludables/Naturales', 'Lácteas/Café', 'Hidratación Básica', 'Lácteas/Café']

plt.scatter(coords[:, 0], coords[:, 1], c=['#005A9C', '#D9534F', '#5CB85C', '#D9534F', '#5CB85C', '#8E44AD', '#005A9C', '#8E44AD'], s=150, zorder=3)

for i, txt in enumerate(drinks):
    plt.annotate(txt, (coords[i, 0] + 0.08, coords[i, 1] + 0.05), fontsize=11, fontweight='bold', color='#333333')

plt.axhline(0, color='grey', linestyle='--', linewidth=0.8, alpha=0.7)
plt.axvline(0, color='grey', linestyle='--', linewidth=0.8, alpha=0.7)
plt.title('Mapa Perceptual 2D de Productos de Bebidas (MDS)', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Dimensión 1: Funcionalidad vs. Indulgence / Estimulación', fontsize=11, labelpad=10)
plt.ylabel('Dimensión 2: Origen Natural / Salud vs. Procesado', fontsize=11, labelpad=10)
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.savefig('mds_map.png', dpi=300)
plt.close()

print("MDS Map generated successfully.")
