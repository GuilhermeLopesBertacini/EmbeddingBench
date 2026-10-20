from abc import ABC, abstractmethod
from pathlib import Path
import numpy as np

class BaseFeatureExtractor(ABC):
  def __init__(self, model_path: Path | str) -> None:
    self.model_path = Path(model_path)
    if not self.model_path.exists():
       raise FileNotFoundError(f"Path {self.model_path.absolute()} does not exists.")

  def extract(self, images: np.ndarray) -> np.ndarray:
    """
    Parameters:
        images (np.ndarray): shape (B, H, W, C), dtype float32, valores em [0.0, 1.0], espaço de cor RGB (C = 3)

    Returns:
        np.ndarray: shape (B, D), dtype float32, vetores unitários
    """

    if len(images.shape) != 4 or images.shape[-1] != 3 or images.dtype != np.float32:
        raise ValueError(f"Images must have shape (B, H, W, 3) of type float32. \nActual shape: {images.shape}. \nActual type: {images.dtype}")

    emb = self._forward(images)

    norms = np.linalg.norm(
       emb,
       axis=-1, # select the same feature (by its index) for every row
       keepdims=True
    )

    return emb / np.maximum(norms, 1e-12)

  @abstractmethod
  def _forward(self, images: np.ndarray) -> np.ndarray:
     pass