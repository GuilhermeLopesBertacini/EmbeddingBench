from typing import Any, Optional
import onnxruntime as ort
import numpy as np
from pathlib import Path
from src.extractors.base import BaseFeatureExtractor
from src.core.resource_guard import get_onnx_session_options, get_onnx_providers


class ONNXFeatureExtractor(BaseFeatureExtractor):
    def __init__(
        self,
        model_path: Path | str,
        num_threads: Optional[int] = None,
        gpu_mem_limit_bytes: Optional[int] = None,
    ) -> None:
        super().__init__(model_path)
        sess_options = get_onnx_session_options(num_threads=num_threads)
        providers = get_onnx_providers(gpu_mem_limit_bytes=gpu_mem_limit_bytes)
        self.session = ort.InferenceSession(
            path_or_bytes=str(self.model_path),
            sess_options=sess_options,
            providers=providers,
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