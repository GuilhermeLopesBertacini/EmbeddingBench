#!/usr/bin/env python3
"""
Gera análise visual comparativa (t-SNE 2D) do espaço de embeddings:
1. MobileNetV4 Base Pretrained (ImageNet) vs MobileNetV4 Fine-Tuned (Balança).
2. Fine-Tuned Baseline Puro vs ISP Otimizado.
"""

import sys
from pathlib import Path

# Garante que a raiz do repositório está no sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from src.emulation.transforms import EmulationParams, apply_emulation
from src.extractors.onnx import ONNXFeatureExtractor
from src.search.optimizer import EmulationSearchEngine
from src.core.resource_guard import configure_system_resources

sns.set_theme(style="whitegrid", palette="tab20")
plt.rcParams.update({"font.size": 10, "figure.autolayout": True})


def extract_normalized_embeddings(engine: EmulationSearchEngine, params: EmulationParams) -> np.ndarray:
    """Extrai embeddings para todos os frames cacheados sob os parâmetros fornecidos."""
    all_embs = []
    batch_size = 128
    for start_idx in range(0, len(engine.cached_images), batch_size):
        chunk = engine.cached_images[start_idx : start_idx + batch_size]
        batch_images = [apply_emulation(img, params) for img in chunk]
        batch_array = np.stack(batch_images, axis=0).astype(np.float32)
        emb = engine.extractor.extract(batch_array)
        all_embs.append(emb)

    full_embs = np.concatenate(all_embs, axis=0)
    return full_embs


def plot_side_by_side_projections(
    embs_left: np.ndarray,
    embs_right: np.ndarray,
    labels: np.ndarray,
    title_main: str,
    title_left: str,
    title_right: str,
    sub_left: str,
    sub_right: str,
    output_path: Path,
    method: str = "tsne",
):
    """Gera projeção 2D lado a lado com cores por SKU."""
    if method == "tsne":
        reducer_left = TSNE(n_components=2, perplexity=30, random_state=42, metric="cosine")
        reducer_right = TSNE(n_components=2, perplexity=30, random_state=42, metric="cosine")
        title_prefix = "t-SNE (Métrica Cosseno)"
    else:
        reducer_left = PCA(n_components=2, random_state=42)
        reducer_right = PCA(n_components=2, random_state=42)
        title_prefix = "PCA"

    coords_left = reducer_left.fit_transform(embs_left)
    coords_right = reducer_right.fit_transform(embs_right)

    unique_labels = sorted(list(set(labels)))
    palette = sns.color_palette("tab20", n_colors=len(unique_labels))
    label_to_color = {lbl: palette[i % len(palette)] for i, lbl in enumerate(unique_labels)}

    fig, axes = plt.subplots(1, 2, figsize=(18, 8), sharex=False, sharey=False)

    for ax, coords, title, sub_label in [
        (axes[0], coords_left, title_left, sub_left),
        (axes[1], coords_right, title_right, sub_right),
    ]:
        for lbl in unique_labels:
            mask = (labels == lbl)
            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                label=lbl,
                color=label_to_color[lbl],
                alpha=0.75,
                s=28,
                edgecolors="none",
            )
        ax.set_title(f"{title}\n({sub_label})", fontsize=13, fontweight="bold", pad=10)
        ax.set_xlabel("Dimensão 1", fontsize=11)
        ax.set_ylabel("Dimensão 2", fontsize=11)

    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="lower center",
        ncol=min(11, len(unique_labels)),
        title="SKUs de Produtos",
        bbox_to_anchor=(0.5, -0.05),
        frameon=True,
    )

    fig.suptitle(
        f"{title_main} - {title_prefix}",
        fontsize=16,
        fontweight="bold",
        y=1.02,
    )
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Gráfico salvo em: {output_path}")


def main():
    output_dir = Path("outputs/plots")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Configura tetos dinâmicos de CPU e GPU
    plan = configure_system_resources()

    params_baseline = EmulationParams(gamma_delta=1.0, contrast_delta=1.0, saturation_delta=1.0)
    params_optimal = EmulationParams(gamma_delta=0.85, contrast_delta=1.20, saturation_delta=1.30)

    base_onnx_path = Path("data/models/mobilenetv4_conv_medium_pretrained.onnx")
    finetuned_onnx_path = Path("data/models/fine-tuned.onnx")

    if not base_onnx_path.exists():
        print("Modelo base não encontrado. Executando scripts/download_and_export_base_mobilenetv4.py...")
        import subprocess
        subprocess.run(["python3", "scripts/download_and_export_base_mobilenetv4.py"], check=True)

    print("Carregando extrator MobileNetV4 Base (ImageNet Pretrained)...")
    base_extractor = ONNXFeatureExtractor(
        base_onnx_path,
        num_threads=plan.usable_cpu_threads,
        gpu_mem_limit_bytes=plan.usable_gpu_vram_bytes,
    )
    engine_base = EmulationSearchEngine(extractor=base_extractor)
    labels = engine_base.labels_arr

    print("Extraindo embeddings do Modelo Base...")
    embs_base = extract_normalized_embeddings(engine_base, params_baseline)

    print("\nCarregando extrator MobileNetV4 Fine-Tuned...")
    finetuned_extractor = ONNXFeatureExtractor(
        finetuned_onnx_path,
        num_threads=plan.usable_cpu_threads,
        gpu_mem_limit_bytes=plan.usable_gpu_vram_bytes,
    )
    engine_finetuned = EmulationSearchEngine(extractor=finetuned_extractor)

    print("Extraindo embeddings do Modelo Fine-Tuned (Baseline e ISP Otimizado)...")
    embs_ft_base = extract_normalized_embeddings(engine_finetuned, params_baseline)
    embs_ft_opt = extract_normalized_embeddings(engine_finetuned, params_optimal)

    # Gráfico 1: Base Pretrained vs Fine-Tuned (ambos no baseline puro)
    print("\nGerando Gráfico 1: MobileNetV4 Base Pretrained vs Fine-Tuned...")
    plot_side_by_side_projections(
        embs_base,
        embs_ft_base,
        labels,
        title_main="Efeito do Fine-Tuning de Métrica (MobileNetV4 Medium)",
        title_left="Base Pretrained (ImageNet)",
        title_right="Fine-Tuned (Balança / ArcFace)",
        sub_left="Pesos genéricos ImageNet (sem especialização)",
        sub_right="Especializado para SKUs de Balança",
        output_path=output_dir / "mobilenetv4_base_vs_finetuned_tsne.png",
        method="tsne",
    )

    # Gráfico 2: Fine-Tuned Baseline vs Emulação ISP Otimizada
    print("\nGerando Gráfico 2: Fine-Tuned Baseline vs Emulação ISP...")
    plot_side_by_side_projections(
        embs_ft_base,
        embs_ft_opt,
        labels,
        title_main="Efeito da Otimização de ISP no Modelo Fine-Tuned",
        title_left="Baseline Puro de Câmera",
        title_right="ISP Otimizado (Emulação)",
        sub_left="γ=1.00 | Contraste=1.00 | Saturação=1.00",
        sub_right="γ=0.85 | Contraste=1.20 | Saturação=1.30",
        output_path=output_dir / "mobilenetv4_finetuned_isp_comparison.png",
        method="tsne",
    )

    print("\nTodas as visualizações foram geradas com sucesso em outputs/plots/!")


if __name__ == "__main__":
    main()
