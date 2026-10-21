from __future__ import annotations

from typing import Dict
import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score


def evaluate_cluster_separability(
    embeddings: np.ndarray,
    labels: np.ndarray,
) -> Dict[str, float]:
    """
    Avalia a geometria dos clusters no espaço latente.
    Assume que os embeddings já estão ou serão normalizados na hiperesfera L2.
    
    Métricas computadas:
    - silhouette_cosine: Coesão intra-classe vs separação inter-classe [-1, 1] (maior é melhor)
    - davies_bouldin: Razão de similaridade entre clusters (menor é melhor)
    - calinski_harabasz: Razão de dispersão entre vs dentro de clusters (maior é melhor)
    - mean_inter_distance: Distância angular média entre centróides de classes diferentes
    - mean_intra_variance: Dispersão angular média das amostras ao redor de seu centróide
    - fisher_ratio: Separação inter / Dispersão intra (maior é melhor)
    """
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return {
            "silhouette_cosine": 0.0,
            "davies_bouldin": float("inf"),
            "calinski_harabasz": 0.0,
            "mean_inter_distance": 0.0,
            "mean_intra_variance": 1.0,
            "fisher_ratio": 0.0,
        }

    # Normalização L2 de salvaguarda
    norms = np.linalg.norm(embeddings, axis=-1, keepdims=True)
    norm_embeddings = embeddings / np.maximum(norms, 1e-12)

    sil = float(silhouette_score(norm_embeddings, labels, metric="cosine"))
    db = float(davies_bouldin_score(norm_embeddings, labels))
    ch = float(calinski_harabasz_score(norm_embeddings, labels))

    # Cálculo dos centróides e dispersões para Fisher Ratio
    centroids = []
    intra_vars = []

    for lbl in unique_labels:
        mask = (labels == lbl)
        pts = norm_embeddings[mask]
        
        # Centróide na hiperesfera unitária
        c = np.mean(pts, axis=0)
        c = c / np.maximum(np.linalg.norm(c), 1e-12)
        centroids.append(c)

        # Dispersão intra-classe (1.0 - similaridade por cosseno com o centróide)
        cosine_sims = np.dot(pts, c)
        intra_vars.append(float(np.mean(1.0 - cosine_sims)))

    centroids = np.array(centroids)

    # Distância inter-classes (1.0 - cosseno entre pares de centróides)
    inter_dists = []
    num_classes = len(centroids)
    for i in range(num_classes):
        for j in range(i + 1, num_classes):
            inter_dists.append(float(1.0 - np.dot(centroids[i], centroids[j])))

    mean_inter = float(np.mean(inter_dists)) if inter_dists else 0.0
    mean_intra = float(np.mean(intra_vars)) if intra_vars else 1.0
    fisher_ratio = float(mean_inter / (mean_intra + 1e-6))

    return {
        "silhouette_cosine": sil,
        "davies_bouldin": db,
        "calinski_harabasz": ch,
        "mean_inter_distance": mean_inter,
        "mean_intra_variance": mean_intra,
        "fisher_ratio": fisher_ratio,
    }
