"""Generate charts for the Critic Agent evaluation blog post."""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.size"] = 12

OUT_DIR = "/Users/choeyunbeom/Desktop/github.io/choeyunbeom.github.io/assets/images/critic-eval"

# ── Colors ──
C_70B = "#4F46E5"   # indigo
C_8B = "#F59E0B"    # amber
C_BG = "#F8F8FC"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Chart 1: Model Comparison Bar Chart
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def chart_model_comparison():
    metrics = ["Sensitivity", "Specificity", "Accuracy", "Borderline\nDetection"]
    scores_70b = [100, 67, 83, 100]
    scores_8b = [67, 67, 67, 0]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5.5))
    fig.patch.set_facecolor(C_BG)
    ax.set_facecolor(C_BG)

    bars1 = ax.bar(x - width/2, scores_70b, width, label="70B (llama-3.3-70b)", color=C_70B, edgecolor="white", linewidth=0.5)
    bars2 = ax.bar(x + width/2, scores_8b, width, label="8B (llama-3.1-8b)", color=C_8B, edgecolor="white", linewidth=0.5)

    # Value labels
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                f"{int(bar.get_height())}%", ha="center", va="bottom", fontweight="bold", fontsize=11, color=C_70B)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                f"{int(bar.get_height())}%", ha="center", va="bottom", fontweight="bold", fontsize=11, color=C_8B)

    ax.set_ylabel("Score (%)", fontsize=13)
    ax.set_title("Critic Agent Performance: 70B vs 8B", fontsize=15, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=12)
    ax.set_ylim(0, 115)
    ax.legend(fontsize=11, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.axhline(y=30, color="#E5E7EB", linestyle="--", linewidth=0.8)

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/model_comparison.png", dpi=150, bbox_inches="tight", facecolor=C_BG)
    plt.close()
    print("  ✓ model_comparison.png")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Chart 2: Confusion Matrix Heatmaps (side by side)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def chart_confusion_matrices():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.patch.set_facecolor(C_BG)

    # 70B
    cm_70b = np.array([[2, 1], [0, 3]])
    # 8B
    cm_8b = np.array([[2, 1], [1, 2]])

    labels_pred = ["sufficient", "insufficient"]
    labels_actual = ["Clean", "Hallucinated"]

    for ax, cm, title, cmap in [
        (ax1, cm_70b, "70B (llama-3.3-70b)", "Blues"),
        (ax2, cm_8b, "8B (llama-3.1-8b)", "Oranges"),
    ]:
        ax.set_facecolor(C_BG)
        im = ax.imshow(cm, interpolation="nearest", cmap=cmap, vmin=0, vmax=3)

        for i in range(2):
            for j in range(2):
                color = "white" if cm[i, j] >= 2 else "black"
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        fontsize=20, fontweight="bold", color=color)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(labels_pred, fontsize=10)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(labels_actual, fontsize=10)
        ax.set_xlabel("Predicted", fontsize=11)
        ax.set_ylabel("Actual", fontsize=11)
        ax.set_title(title, fontsize=13, fontweight="bold", pad=10)

    plt.suptitle("Confusion Matrices", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/confusion_matrices.png", dpi=150, bbox_inches="tight", facecolor=C_BG)
    plt.close()
    print("  ✓ confusion_matrices.png")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Chart 3: Per-Case Cited vs Uncited (70B)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def chart_citation_breakdown():
    cases = [
        ("clean_1", 7, 3, "clean"),
        ("clean_2", 8, 2, "clean"),
        ("clean_3", 7, 4, "clean"),
        ("halluc_1", 0, 9, "hallucinated"),
        ("halluc_2", 2, 5, "hallucinated"),
        ("halluc_3", 0, 9, "hallucinated"),
        ("border_1", 6, 4, "borderline"),
        ("border_2", 7, 3, "borderline"),
        ("border_3", 7, 3, "borderline"),
    ]

    labels = [c[0] for c in cases]
    cited = [c[1] for c in cases]
    uncited = [c[2] for c in cases]
    types = [c[3] for c in cases]

    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(12, 5.5))
    fig.patch.set_facecolor(C_BG)
    ax.set_facecolor(C_BG)

    bars_c = ax.bar(x, cited, 0.6, label="Cited", color="#10B981", edgecolor="white", linewidth=0.5)
    bars_u = ax.bar(x, uncited, 0.6, bottom=cited, label="Uncited", color="#EF4444", edgecolor="white", linewidth=0.5)

    # 30% threshold line per case
    for i, (c, u) in enumerate(zip(cited, uncited)):
        total = c + u
        if total > 0:
            threshold_y = total * 0.7  # 30% uncited means 70% cited
            ax.plot([i - 0.35, i + 0.35], [threshold_y, threshold_y],
                    color="#6B7280", linestyle="--", linewidth=1)

    # Type background shading
    ax.axvspan(-0.5, 2.5, alpha=0.06, color="#10B981")  # clean
    ax.axvspan(2.5, 5.5, alpha=0.06, color="#EF4444")    # hallucinated
    ax.axvspan(5.5, 8.5, alpha=0.06, color="#F59E0B")    # borderline

    ax.text(1, -1.8, "Clean", ha="center", fontsize=10, color="#059669", fontweight="bold")
    ax.text(4, -1.8, "Hallucinated", ha="center", fontsize=10, color="#DC2626", fontweight="bold")
    ax.text(7, -1.8, "Borderline", ha="center", fontsize=10, color="#D97706", fontweight="bold")

    ax.set_ylabel("Number of Claims", fontsize=13)
    ax.set_title("Citation Breakdown per Case (70B Judge)", fontsize=15, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10, rotation=30, ha="right")
    ax.legend(fontsize=11, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add "30% threshold" annotation
    ax.annotate("30% threshold", xy=(8.3, 7.7), fontsize=9, color="#6B7280", style="italic")

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/citation_breakdown.png", dpi=150, bbox_inches="tight", facecolor=C_BG)
    plt.close()
    print("  ✓ citation_breakdown.png")


if __name__ == "__main__":
    print("Generating charts...")
    chart_model_comparison()
    chart_confusion_matrices()
    chart_citation_breakdown()
    print(f"\nAll charts saved to {OUT_DIR}/")
