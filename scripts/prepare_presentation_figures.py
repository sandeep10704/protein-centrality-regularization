"""
Script: prepare_presentation_figures.py
Curates and generates the 9 designated figures for the college project presentation:
1. Overall project architecture.
2. Protein graph construction.
3. GATv2 architecture.
4. AGA methodology.
5. Bio-AGA methodology.
6. Training and validation curves.
7. Model performance comparison.
8. Ablation comparison.
9. Graph edge-retention visualization.
"""

import os
import shutil
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

SRC_DIR = "results/figures"
DEST_DIR = "results/presentation_figures"
os.makedirs(DEST_DIR, exist_ok=True)


def create_gatv2_architecture_figure(save_path):
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    ax.axis("off")

    # Schematic of dynamic GATv2 attention mechanism
    # Node i and Node j
    c1 = plt.Circle((0.2, 0.7), 0.08, color="#1f78b4", ec="black", lw=1.5)
    c2 = plt.Circle((0.2, 0.3), 0.08, color="#33a02c", ec="black", lw=1.5)
    ax.add_patch(c1)
    ax.add_patch(c2)
    ax.text(0.2, 0.7, "Node $u$\n($h_u$)", color="white", weight="bold", ha="center", va="center", fontsize=10)
    ax.text(0.2, 0.3, "Node $v$\n($h_v$)", color="white", weight="bold", ha="center", va="center", fontsize=10)

    # Linear transformation block
    r1 = patches.FancyBboxPatch((0.36, 0.35), 0.16, 0.3, boxstyle="round,pad=0.02", fc="#f0f0f0", ec="#333333", lw=1.5)
    ax.add_patch(r1)
    ax.text(0.44, 0.5, "Concatenate &\nLinear Proj:\n$W_L [h_u || h_v || e_{uv}]$", ha="center", va="center", weight="bold", fontsize=9)

    # LeakyReLU block
    r2 = patches.FancyBboxPatch((0.58, 0.4), 0.12, 0.2, boxstyle="round,pad=0.02", fc="#fee0d2", ec="#de2d26", lw=1.5)
    ax.add_patch(r2)
    ax.text(0.64, 0.5, "LeakyReLU\nActivation", ha="center", va="center", weight="bold", fontsize=9)

    # Attention vector a^T
    r3 = patches.FancyBboxPatch((0.76, 0.4), 0.14, 0.2, boxstyle="round,pad=0.02", fc="#deebf7", ec="#3182bd", lw=1.5)
    ax.add_patch(r3)
    ax.text(0.83, 0.5, "Dynamic Vector\n$a^T \\cdot (\\cdot)$\n$softmax(\\cdot)$", ha="center", va="center", weight="bold", fontsize=9)

    # Output attention alpha_uv
    c3 = plt.Circle((0.95, 0.5), 0.04, color="#e41a1c", ec="black", lw=1.5)
    ax.add_patch(c3)
    ax.text(0.95, 0.5, r"$\alpha_{uv}$", color="white", weight="bold", ha="center", va="center", fontsize=11)

    # Arrows
    ax.annotate("", xy=(0.36, 0.55), xytext=(0.28, 0.68), arrowprops=dict(arrowstyle="->", lw=2, color="#1f78b4"))
    ax.annotate("", xy=(0.36, 0.45), xytext=(0.28, 0.32), arrowprops=dict(arrowstyle="->", lw=2, color="#33a02c"))
    ax.annotate("", xy=(0.58, 0.5), xytext=(0.52, 0.5), arrowprops=dict(arrowstyle="->", lw=2, color="#333333"))
    ax.annotate("", xy=(0.76, 0.5), xytext=(0.70, 0.5), arrowprops=dict(arrowstyle="->", lw=2, color="#333333"))
    ax.annotate("", xy=(0.91, 0.5), xytext=(0.90, 0.5), arrowprops=dict(arrowstyle="->", lw=2, color="#333333"))

    ax.set_title("GATv2 Universal Dynamic Attention Mechanism: $\\alpha_{uv} = \\mathrm{Softmax}_v (a^T \\mathrm{LeakyReLU}(W [h_u || h_v || e_{uv}]))$", pad=15, weight="bold")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Generated {save_path}")


def prepare_figures():
    print("Curating 9 presentation figures in results/presentation_figures/...")

    # 1. Overall project architecture
    shutil.copy(f"{SRC_DIR}/fig01_overall_architecture.png", f"{DEST_DIR}/01_overall_architecture.png")

    # 2. Protein graph construction
    shutil.copy(f"{SRC_DIR}/fig02_protein_graph_construction.png", f"{DEST_DIR}/02_protein_graph_construction.png")

    # 3. GATv2 architecture
    create_gatv2_architecture_figure(f"{DEST_DIR}/03_gatv2_architecture.png")

    # 4. AGA methodology
    shutil.copy(f"{SRC_DIR}/fig03_original_aga_mechanism.png", f"{DEST_DIR}/04_aga_methodology.png")

    # 5. Bio-AGA methodology
    shutil.copy(f"{SRC_DIR}/fig04_bio_aga_mechanism.png", f"{DEST_DIR}/05_bio_aga_methodology.png")

    # 6. Training and validation curves
    shutil.copy(f"{SRC_DIR}/fig08_training_validation_curves.png", f"{DEST_DIR}/06_training_validation_curves.png")

    # 7. Model performance comparison
    shutil.copy(f"{SRC_DIR}/fig09_model_accuracy_comparison.png", f"{DEST_DIR}/07_model_performance_comparison.png")

    # 8. Ablation comparison
    shutil.copy(f"{SRC_DIR}/fig11_ablation_study.png", f"{DEST_DIR}/08_ablation_comparison.png")

    # 9. Graph edge-retention visualization
    shutil.copy(f"{SRC_DIR}/fig14_sequence_distance_retention.png", f"{DEST_DIR}/09_edge_retention_visualization.png")

    print(f"All 9 presentation figures prepared in {DEST_DIR}!")


if __name__ == "__main__":
    prepare_figures()
