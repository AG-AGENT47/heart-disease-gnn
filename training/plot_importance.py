import matplotlib.pyplot as plt
import numpy as np

# Data from your output
factors = ['fbs: < 120mg/dl (Normal)', 'oldpeak: 1.0-1.5 (ST Depr.)',
           'age: 63-67', 'ca: 3 Blocked Vessels', 'sex: Male']
scores = [0.9111, 0.9107, 0.8915, 0.8908, 0.8664]

# Reverse for horizontal bar chart (Top at top)
factors = factors[::-1]
scores = scores[::-1]

plt.figure(figsize=(10, 6))
bars = plt.barh(range(len(factors)), scores, color='#4c72b0', align='center')

plt.yticks(range(len(factors)), factors, fontsize=11)
plt.xlabel('GNNExplainer Importance Score', fontsize=12)
plt.title('Top 5 Factors contributing to High Risk (Patient 0)', fontsize=14, pad=20)
plt.xlim(0.8, 0.95)  # Zoom in to show differences since scores are close
plt.grid(axis='x', linestyle='--', alpha=0.5)

# Add value labels
for i, v in enumerate(scores):
    plt.text(v + 0.002, i, f'{v:.4f}', va='center', fontweight='bold')

plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300)
print("Saved feature_importance.png")