#!/usr/bin/env python3
"""
Script de execução da busca em grade para otimização de parâmetros de ISP/emulação.
Suporta modelos fine-tuned em ONNX, modelo base MobileNetV4 ou DINOv3 ConvNeXt em PyTorch.
"""

import argparse
import sys
from pathlib import Path

# Garante que a raiz do repositório está no sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from src.extractors.onnx import ONNXFeatureExtractor
from src.extractors.pytorch_dino import PyTorchDINOv3Extractor
from src.search.optimizer import EmulationSearchEngine
from src.core.resource_guard import configure_system_resources


def parse_args():
    parser = argparse.ArgumentParser(description="Otimização de parâmetros de câmera via emulação e clusters")
    parser.add_argument(
        "--model-type",
        choices=["onnx", "mobilenetv4_base", "dinov3_tiny", "dinov3_small", "dinov3_base"],
        default="onnx",
        help="Tipo do extrator de features a ser utilizado",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Caminho customizado para o modelo (.onnx ou .pth)",
    )
    parser.add_argument(
        "--frames-dir",
        type=str,
        default="data/scale_frames",
        help="Diretório contendo os frames por SKU",
    )
    parser.add_argument(
        "--config-tag",
        type=str,
        default="exp167_wb4100_gain25",
        help="Configuração de câmera a ser filtrada",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Nome do arquivo CSV de saída em outputs/",
    )
    parser.add_argument(
        "--max-threads",
        type=int,
        default=None,
        help="Limite manual de CPU threads (padrão: detecta cores ociosos automaticamente)",
    )
    parser.add_argument(
        "--gpu-mem-limit-mb",
        type=int,
        default=None,
        help="Limite manual de VRAM em MB (padrão: detecta VRAM livre com margem de segurança)",
    )
    parser.add_argument(
        "--gpu-safety-factor",
        type=float,
        default=0.8,
        help="Fração da VRAM livre a ser utilizada (padrão: 0.8 = 20%% de margem livre)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 0. Configura tetos de recursos de hardware (CPU / GPU)
    gpu_mem_bytes = int(args.gpu_mem_limit_mb * 1024 * 1024) if args.gpu_mem_limit_mb else None
    plan = configure_system_resources(
        max_threads=args.max_threads,
        gpu_mem_limit_bytes=gpu_mem_bytes,
        gpu_safety_factor=args.gpu_safety_factor,
    )

    # 1. Instanciação do Extrator de Features adequado
    if args.model_type == "onnx":
        path = Path(args.model_path or "data/models/fine-tuned.onnx")
        print(f"Usando Extrator ONNX (Fine-Tuned): {path}")
        extractor = ONNXFeatureExtractor(
            path,
            num_threads=plan.usable_cpu_threads,
            gpu_mem_limit_bytes=plan.usable_gpu_vram_bytes,
        )
        default_csv = "emulation_search_finetuned_onnx.csv"
    elif args.model_type == "mobilenetv4_base":
        path = Path(args.model_path or "data/models/mobilenetv4_conv_medium_pretrained.onnx")
        if not path.exists():
            import subprocess
            subprocess.run(["python3", "scripts/download_and_export_base_mobilenetv4.py"], check=True)
        print(f"Usando Extrator ONNX (MobileNetV4 Base Pretrained): {path}")
        extractor = ONNXFeatureExtractor(
            path,
            num_threads=plan.usable_cpu_threads,
            gpu_mem_limit_bytes=plan.usable_gpu_vram_bytes,
        )
        default_csv = "emulation_search_mobilenetv4_base_pretrained.csv"
    else:
        # Família DINOv3 ConvNeXt em PyTorch
        arch_map = {
            "dinov3_tiny": ("convnext_tiny", "data/models/dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth"),
            "dinov3_small": ("convnext_small", "data/models/dinov3_convnext_small_pretrain_lvd1689m-296db49d.pth"),
            "dinov3_base": ("convnext_base", "data/models/dinov3_convnext_base_pretrain_lvd1689m-801f2ba9.pth"),
        }
        arch, default_path = arch_map[args.model_type]
        ckpt_path = Path(args.model_path or default_path)
        print(f"Usando Extrator Foundation Model: {arch} de {ckpt_path}")
        extractor = PyTorchDINOv3Extractor(
            ckpt_path,
            architecture=arch,
            num_threads=plan.usable_cpu_threads,
            gpu_mem_limit_bytes=plan.usable_gpu_vram_bytes,
        )
        default_csv = f"emulation_search_{args.model_type}.csv"

    # 2. Configuração do Engine
    engine = EmulationSearchEngine(
        extractor=extractor,
        scale_frames_dir=args.frames_dir,
        target_config=args.config_tag,
    )
    print(f"Dataset carregado: {len(engine.cached_images)} frames em {len(set(engine.labels))} SKUs.")

    # 3. Grade de busca
    gammas = [0.85, 0.92, 1.0, 1.08, 1.15]
    contrasts = [0.90, 1.0, 1.10, 1.20]
    saturations = [0.85, 1.0, 1.15, 1.30]

    results = engine.run_grid_search(gammas, contrasts, saturations)

    df_results = pd.DataFrame(results)
    csv_name = args.output_csv or default_csv
    csv_out = output_dir / csv_name
    df_results.to_csv(csv_out, index=False)
    print(f"\nResultados completos salvos em: {csv_out}")

    print("\n--- TOP 5 CONFIGURAÇÕES (Por Silhouette Score) ---")
    print(df_results.head(5)[["gamma_delta", "contrast_delta", "saturation_delta", "silhouette_cosine", "davies_bouldin", "fisher_ratio"]])

    baseline_row = df_results[
        (df_results["gamma_delta"] == 1.0)
        & (df_results["contrast_delta"] == 1.0)
        & (df_results["saturation_delta"] == 1.0)
    ]
    if not baseline_row.empty:
        print("\n--- BASELINE PURO (Sem modificação de câmera) ---")
        print(baseline_row[["gamma_delta", "contrast_delta", "saturation_delta", "silhouette_cosine", "davies_bouldin", "fisher_ratio"]])


if __name__ == "__main__":
    main()
