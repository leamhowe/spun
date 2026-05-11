import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

# Set the style for scientific presentation
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("talk") 

# ==========================================
# 1. DATA ENTRY
# ==========================================

# Data from Table 1 (Focusing on Snow-Only RMSE)
data_rmse = {
    'Scenario': ['Tier 1: Intrinsic\n(10m)', 'Tier 1: Intrinsic\n(10m)', 'Tier 1: Intrinsic\n(10m)',
                 'Tier 2a: Control\n(20m Clear)', 'Tier 2a: Control\n(20m Clear)', 'Tier 2a: Control\n(20m Clear)', 'Tier 2a: Control\n(20m Clear)',
                 'Tier 2b: Reality\n(20m All-Sky)', 'Tier 2b: Reality\n(20m All-Sky)', 'Tier 2b: Reality\n(20m All-Sky)', 'Tier 2b: Reality\n(20m All-Sky)'],
    'Model': ['SPUN', 'NDSI (>0.4)', 'Wang et al.',
              'SPUN', 'LIS (Ops)', 'NDSI (>0.4)', 'Wang et al.',
              'SPUN', 'LIS (Ops)', 'NDSI (>0.4)', 'Wang et al.'],
    'RMSE': [42.16, 74.24, 67.00,
             28.98, 33.30, 75.64, 41.17,
             26.58, 50.09, 59.32, 51.07]
}
df_rmse = pd.DataFrame(data_rmse)

# Data from Table 2 (Binary Performance)
data_pr = {
    'Scenario': ['Tier 1', 'Tier 1', 'Tier 1',
                 'Tier 2a', 'Tier 2a', 'Tier 2a', 'Tier 2a',
                 'Tier 2b', 'Tier 2b', 'Tier 2b', 'Tier 2b'],
    'Model': ['SPUN', 'NDSI', 'Wang',
              'SPUN', 'LIS', 'NDSI', 'Wang',
              'SPUN', 'LIS', 'NDSI', 'Wang'],
    'Recall': [0.821, 0.432, 0.132,
               0.789, 0.439, 0.333, 0.035,
               0.874, 0.013, 0.396, 0.134],
    'Precision': [0.668, 0.246, 0.347,
                  0.492, 0.562, 0.028, 0.093,
                  0.652, 0.562, 0.232, 0.354]
}
df_pr = pd.DataFrame(data_pr)

# Data from Table 3 (Hydrological Deviation)
data_dev = {
    'Scenario': ['Control (Clear)', 'Control (Clear)', 'Control (Clear)',
                 'Reality (All-Sky)', 'Reality (All-Sky)', 'Reality (All-Sky)'],
    'Model': ['SPUN', 'LIS', 'Wang et al.',
              'SPUN', 'LIS', 'Wang et al.'],
    'Deviation': [12.3, -57.7, -70.0,
                  18.2, -98.4, -63.6]
}
df_dev = pd.DataFrame(data_dev)

# ==========================================
# 2. PLOTTING
# ==========================================

# Define colors
colors = {'SPUN': '#1f77b4', 'NDSI': '#ff7f0e', 'NDSI (>0.4)': '#ff7f0e', 
          'Wang et al.': '#2ca02c', 'Wang': '#2ca02c', 
          'LIS': '#d62728', 'LIS (Ops)': '#d62728'}

# --- Figure 1: RMSE Bar Chart ---
plt.figure(figsize=(10, 6))
sns.barplot(data=df_rmse, x='Scenario', y='RMSE', hue='Model', palette=colors)
plt.title('Snow-Only RMSE (Lower is Better)', fontweight='bold')
plt.ylabel('RMSE (Snow Fraction %)')
plt.xlabel('')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# --- Figure 2: Precision-Recall Scatter ---
plt.figure(figsize=(8, 8))
sns.scatterplot(data=df_pr, x='Recall', y='Precision', hue='Model', style='Scenario', s=200, palette=colors)
plt.xlim(0, 1.0)
plt.ylim(0, 1.0)
plt.title('Trade-off: Precision vs. Recall', fontweight='bold')
plt.xlabel('Recall (finding snow)')
plt.ylabel('Precision (trusting snow)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, linestyle='--')
plt.tight_layout()
plt.show()

# --- Figure 3: Deviation Diverging Bar ---
plt.figure(figsize=(10, 5))
# Add a column for color based on sign
df_dev['Color'] = df_dev['Deviation'].apply(lambda x: '#d62728' if x < 0 else '#1f77b4')

ax = sns.barplot(data=df_dev, x='Deviation', y='Model', hue='Scenario', palette='vlag')
plt.axvline(0, color='black', linewidth=2, linestyle='--')
plt.title('Total Snow Area Deviation', fontweight='bold')
plt.xlabel('% Deviation from Ground Truth')
plt.xlim(-100, 50) # Cut off the extreme positive outliers
plt.text(40, 2, "NDSI > +700%", color='#ff7f0e', fontweight='bold', va='center')
plt.tight_layout()
plt.show()