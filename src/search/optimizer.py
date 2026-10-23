from __future__ import annotations

import itertools
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from PIL import Image

from src.dataset.frame_loader import load_dataset_by_config, resize_center_crop
from src.emulation.transforms import EmulationParams, apply_emulation
from src.extractors.base import BaseFeatureExtractor
from src.metrics.cluster_quality import evaluate_cluster_separability


class EmulationSearchEngine:
    """
    Motor que carrega o dataset filtrado, aplica a emulação e avalia
    as métricas de clusters contra qualquer extrator aderente ao BaseFeatureExtractor.
    """

    def __init__(
        self,
        extractor: BaseFeatureExtractor,
        scale_frames_dir: Path | str = "data/scale_frames",
        target_config: str = "exp167_wb4100_gain25",
        target_size: Tuple[int, int] = (224, 224),
    ):
        self.extractor = extractor
        self.target_size = target_size

        raw_dataset = load_dataset_by_config(scale_frames_dir, target_config=target_config)
        if not raw_dataset:
            raise ValueError(
                f"Nenhuma imagem encontrada em '{scale_frames_dir}' com a tag de câmera '{target_config}'. "
                f"Verifique se o caminho existe e se o padrão de nomes das imagens corresponde à configuração."
            )

        self.cached_images: List[np.ndarray] = []
        self.labels: List[str] = []

        image_size = target_size[0]
        for sku, frames in raw_dataset.items():
            for f in frames:
                with Image.open(f.path) as img:
                    img_cropped = resize_center_crop(img, image_size=image_size)
                    # Mantém no formato canônico [0.0, 1.0] RGB (H, W, 3)
                    arr = np.asarray(img_cropped, dtype=np.float32) / 255.0
                    self.cached_images.append(arr)
                    self.labels.append(sku)

        self.labels_arr = np.array(self.labels)

    def evaluate_candidate(self, params: EmulationParams, batch_size: int = 128) -> Dict[str, float]:
        """Aplica a emulação no dataset em lotes, extrai embeddings e calcula métricas de cluster."""
        all_embeddings = []

        for start_idx in range(0, len(self.cached_images), batch_size):
            chunk = self.cached_images[start_idx : start_idx + batch_size]
            batch_images = [apply_emulation(img, params) for img in chunk]
            batch_array = np.stack(batch_images, axis=0).astype(np.float32)

            # O extrator recebe (B, H, W, 3) em [0.0, 1.0] e devolve embeddings L2-normalizados
            emb = self.extractor.extract(batch_array)
            all_embeddings.append(emb)

        full_embeddings = np.concatenate(all_embeddings, axis=0)
        metrics = evaluate_cluster_separability(full_embeddings, self.labels_arr)
        return metrics

    def run_grid_search(
        self,
        gammas: List[float],
        contrasts: List[float],
        saturations: List[float],
    ) -> List[Dict]:
        """Varre o produto cartesiano dos parâmetros relativos."""
        results = []
        total = len(gammas) * len(contrasts) * len(saturations)
        print(f"Iniciando busca em grade ({total} combinações)...")

        for idx, (g, c, s) in enumerate(itertools.product(gammas, contrasts, saturations), 1):
            params = EmulationParams(gamma_delta=g, contrast_delta=c, saturation_delta=s)
            metrics = self.evaluate_candidate(params)
            record = {
                "gamma_delta": g,
                "contrast_delta": c,
                "saturation_delta": s,
                **metrics,
            }
            results.append(record)
            if idx % 5 == 0 or idx == total:
                print(
                    f"[{idx:2d}/{total}] gamma={g:.2f}, contrast={c:.2f}, sat={s:.2f} "
                    f"-> Silhouette={metrics['silhouette_cosine']:.4f}, Fisher={metrics['fisher_ratio']:.2f}"
                )

        results.sort(key=lambda r: r["silhouette_cosine"], reverse=True)
        return results
