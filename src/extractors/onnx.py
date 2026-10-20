from typing import Any
import onnxruntime as ort
import numpy as np
from pathlib import Path
from src.extractors.base import BaseFeatureExtractor

class ONNXFeatureExtractor(BaseFeatureExtractor):
    def __init__(self, model_path: Path | str) -> None:
        super().__init__(model_path)
        self.session = ort.InferenceSession(
            path_or_bytes=self.model_path,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        self.input_name = self.session.get_inputs()[0].name

    def _forward(self, images: np.ndarray) -> np.ndarray:
        """
        Parameters:
            images (np.ndarray): shape (B, H, W, C), dtype float32, valores em [0.0, 1.0], espaço de cor RGB (C = 3)
        
        Returns:
            np.ndarray: shape (B, D), dtype float32, vetores unitários 
        """
        images = (images * 255).transpose(0, 3, 1, 2).astype(np.float32) # (B, C, H, W)

        outputs = self.session.run(
            output_names=None,
            input_feed={self.input_name: images}
        )

        return outputs[0] # type: ignore