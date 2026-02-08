"""
프로젝트 내 모든 그래프를 figures/ 폴더에 이미지로 저장.
- ROC curve (AUC)
- Confusion matrix
- Feature distribution
- Label distribution

사용법: python scripts/export_all_figures.py  (프로젝트 루트에서)
"""
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix

from config import DATASET_PATH, FIGURES_DIR, MODEL_PATH, DEVICE
from data_loader import load_csv_and_preprocess
from model import HypotensionModelV2

plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
sns.set_style("darkgrid")


def main():
    if not DATASET_PATH.exists():
        print(f"[오류] 데이터셋 없음: {DATASET_PATH}")
        return
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("[1/4] 데이터 로드 및 전처리...")
    data = load_csv_and_preprocess(DATASET_PATH)
    X_test = data["X_test"]
    y_test = data["y_test"]
    feature_cols = data["feature_cols"]
    df = pd.read_csv(DATASET_PATH)
    target = "label"

    # 1) ROC curve (모델 필요)
    if MODEL_PATH.exists():
        print("[2/4] 모델 로드 및 예측...")
        device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=False)
        in_dim = ckpt.get("in_dim", len(feature_cols))
        model = HypotensionModelV2(in_dim=in_dim).to(device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        with torch.no_grad():
            X_t = torch.from_numpy(X_test).float().to(device)
            logits = model(X_t).cpu().numpy()
        y_prob = 1 / (1 + np.exp(-logits))
        y_pred = (y_prob >= 0.5).astype(int)
        try:
            auc = roc_auc_score(y_test, y_prob)
        except ValueError:
            auc = None

        # ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {auc:.3f})" if auc is not None else "ROC curve")
        plt.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve — Intraoperative Hypotension Prediction")
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.savefig(FIGURES_DIR / "roc_curve.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  저장: figures/roc_curve.png (AUC = {auc:.3f})" if auc is not None else "  저장: figures/roc_curve.png")

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No Hypo", "Hypo"], yticklabels=["No Hypo", "Hypo"])
        plt.title("Confusion Matrix")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "confusion_matrix.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("  저장: figures/confusion_matrix.png")
    else:
        print("[2/4] 모델 없음, ROC/Confusion matrix 생략")

    # 2) Feature distribution
    print("[3/4] Feature distribution...")
    feats = [c for c in feature_cols if c in df.columns]
    n = len(feats)
    if n > 0:
        ncols = min(3, n)
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
        if n == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        for idx, col in enumerate(feats):
            axes[idx].hist(df[col], bins=50, alpha=0.7, color="steelblue", edgecolor="black")
            axes[idx].set_xlabel(col, fontsize=10)
            axes[idx].set_ylabel("Frequency", fontsize=10)
            axes[idx].grid(True, alpha=0.3)
            axes[idx].set_yscale("log")
        for idx in range(n, len(axes)):
            axes[idx].set_visible(False)
        plt.suptitle("Feature Distribution (5-min lookback)", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "feature_distribution.png", dpi=100, bbox_inches="tight")
        plt.close()
        print("  저장: figures/feature_distribution.png")

    # 3) Label distribution
    print("[4/4] Label distribution...")
    label_counts = df[target].value_counts()
    colors = ["steelblue", "salmon"]
    all_labels = ["No Hypotension", "Hypotension"]
    available = [all_labels[i] for i in label_counts.index]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].pie(label_counts, labels=available, autopct="%1.1f%%", colors=colors[: len(label_counts)], startangle=90)
    axes[0].set_title("Label Distribution")
    label_counts.plot(kind="bar", ax=axes[1], color=colors[: len(label_counts)])
    axes[1].set_title("Label Count")
    axes[1].set_ylabel("Count")
    axes[1].set_xticklabels(available, rotation=0)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "label_distribution.png", dpi=100, bbox_inches="tight")
    plt.close()
    print("  저장: figures/label_distribution.png")

    print(f"\n[완료] 모든 그래프 저장 위치: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
