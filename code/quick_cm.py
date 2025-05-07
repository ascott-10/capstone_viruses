import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ✅ Manually defined confusion matrix
cm = np.array([
    [77, 19],  # True Wild Type
    [16, 62]   # True Mutant
])

# ✅ Properly spaced class labels
labels = ['Wild Type', 'Mutant']

fig, ax = plt.subplots(figsize=(7, 7))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap="Greens",
    xticklabels=labels,
    yticklabels=labels,
    cbar=False,
    square=True,
    linewidths=0.7,
    linecolor='gray',
    ax=ax,
    annot_kws={"size": 20}
)

ax.set_xlabel('Predicted Label', fontsize=18, labelpad=10)
ax.set_ylabel('True Label', fontsize=18, labelpad=10)
ax.set_title('Confusion Matrix', fontsize=22, pad=15)
ax.tick_params(labelsize=16)

plt.tight_layout()
plt.show()