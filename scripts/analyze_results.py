"""Analysis and Visualization Script"""
import os
import sys
os.environ['PYTHONIOENCODING'] = 'utf-8'

from pathlib import Path
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
sns.set_style('darkgrid')

from config import (
    DATASET_PATH, CHECKPOINT_DIR, FIGURES_DIR, MODEL_PATH,
    TEST_SIZE, RANDOM_STATE, DEVICE,
)
from model import HypotensionModelV2

print("=" * 70)
print("[ANALYSIS] Hypotension Prediction Model - Performance Analysis")
print("=" * 70)

# 1. Load Data
print("\n[1/4] Loading data...")
df = pd.read_csv(DATASET_PATH)
target = "label"
feature_cols = [c for c in df.columns if c not in ("caseid", target)]

print(f"- Dataset: {len(df):,} rows, {len(feature_cols)} features")
print(f"- Label 0 (No Hypotension): {(df[target]==0).sum():,} ({(df[target]==0).sum()/len(df)*100:.1f}%)")
print(f"- Label 1 (Hypotension): {(df[target]==1).sum():,} ({(df[target]==1).sum()/len(df)*100:.1f}%)")
print(f"- Unique Cases: {df['caseid'].nunique()}")

# 2. Feature Distribution
print("\n[2/4] Creating visualizations...")

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle('Feature Distribution (5-min lookback window)', fontsize=14, fontweight='bold')
axes = axes.flatten()
for idx, col in enumerate(feature_cols):
    axes[idx].hist(df[col], bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    axes[idx].set_xlabel(col, fontsize=10)
    axes[idx].set_ylabel('Frequency', fontsize=10)
    axes[idx].grid(True, alpha=0.3)
    axes[idx].set_yscale('log')

plt.tight_layout()
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
plt.savefig(FIGURES_DIR / 'analysis_feature_distribution.png', dpi=100, bbox_inches='tight')
plt.close()
print("OK: Feature distribution -> figures/analysis_feature_distribution.png")

# 3. Label Distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
label_counts = df[target].value_counts()
colors = ['steelblue', 'salmon']
all_labels = ['No Hypotension', 'Hypotension']
available_labels = [all_labels[i] for i in label_counts.index]
axes[0].pie(label_counts, labels=available_labels, autopct='%1.1f%%', 
            colors=colors[:len(label_counts)], startangle=90)
axes[0].set_title('Label Distribution')
label_counts.plot(kind='bar', ax=axes[1], color=colors[:len(label_counts)])
axes[1].set_title('Label Count')
axes[1].set_ylabel('Count')
axes[1].set_xticklabels(available_labels, rotation=0)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'analysis_label_distribution.png', dpi=100, bbox_inches='tight')
plt.close()
print("OK: Label distribution -> figures/analysis_label_distribution.png")

# 4. Feature Statistics
print("\n[3/4] Feature Statistics...")
feature_stats = df[feature_cols].describe()
print(feature_stats.to_string())

# 5. Case Analysis
case_label_dist = df.groupby('caseid')[target].agg(['sum', 'count', 'mean']).rename(
    columns={'sum': 'Hypo_Count', 'count': 'Total_Samples', 'mean': 'Hypo_Rate'}
).sort_values('Hypo_Rate', ascending=False)

print(f"\n[CASE ANALYSIS]")
print(f"- Total cases: {len(case_label_dist)}")
print(f"- Cases with hypotension: {(case_label_dist['Hypo_Count'] > 0).sum()}")
print(f"- Cases without hypotension: {(case_label_dist['Hypo_Count'] == 0).sum()}")
print(f"- Mean hypotension rate: {case_label_dist['Hypo_Rate'].mean()*100:.2f}%")
print("\nTop 10 cases by hypotension rate:")
print(case_label_dist.head(10).to_string())

# 6. Model Evaluation
print("\n[4/4] Model Evaluation...")

if not MODEL_PATH.exists():
    print("ERROR: Model not found. Run train.py first.")
else:
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    X = df[feature_cols].fillna(0).values.astype(np.float32)
    y = df[target].values.astype(np.int64)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    in_dim = checkpoint.get("in_dim", len(feature_cols))
    model = HypotensionModelV2(in_dim=in_dim).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    
    # Predictions
    with torch.no_grad():
        X_t = torch.from_numpy(X_test).to(device)
        logits = model(X_t).cpu().numpy()
    
    y_prob = 1 / (1 + np.exp(-logits))
    y_pred = (y_prob >= 0.5).astype(int).flatten()
    
    print(f"\n[MODEL PERFORMANCE]")
    print(f"- Test Size: {len(X_test):,}")
    print(f"- Accuracy: {(y_pred == y_test).sum() / len(y_test):.4f}")
    try:
        auc = roc_auc_score(y_test, y_prob)
        print(f"- AUC-ROC: {auc:.4f}")
    except ValueError:
        auc = None
    # ROC curve 저장
    try:
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC (AUC = {auc:.3f})" if auc is not None else "ROC")
        plt.plot([0, 1], [0, 1], "k--", lw=1)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve — Hypotension Prediction")
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.savefig(FIGURES_DIR / "roc_curve.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("OK: ROC curve -> figures/roc_curve.png")
    except Exception as e:
        print(f"ROC save skip: {e}")
    # Confusion Matrix
    if len(np.unique(y_test)) > 1:
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        print(f"\n[CONFUSION MATRIX]")
        print(f"TN: {cm[0,0]}, FP: {cm[0,1]}")
        print(f"FN: {cm[1,0]}, TP: {cm[1,1]}")
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['No Hypo', 'Hypo'], 
                    yticklabels=['No Hypo', 'Hypo'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / 'analysis_confusion_matrix.png', dpi=100, bbox_inches='tight')
        plt.close()
        print("OK: Confusion matrix -> figures/analysis_confusion_matrix.png")
    else:
        print(f"\n[WARNING] Only one class in test set")
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        print(f"Confusion Matrix:\n{cm}")

print("\n" + "=" * 70)
print("[COMPLETE] Analysis and Visualization Done")
print("=" * 70)
print("\nGenerated files (in figures/):")
print(f"  - analysis_feature_distribution.png")
print(f"  - analysis_label_distribution.png")
print(f"  - analysis_confusion_matrix.png (if model exists)")
