from __future__ import annotations

from pathlib import Path
from typing import Optional, Union
import numpy as np
import timm
import torch

from src.extractors.base import BaseFeatureExtractor
from src.core.resource_guard import configure_system_resources

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)


class PyTorchDINOv3Extractor(BaseFeatureExtractor):
    """
    Extrator nativo em PyTorch para checkpoints DINOv3 ConvNeXt (.pth).
    Executa em GPU CUDA (se suportado pelo hardware) ou CPU com limites dinâmicos de recursos.
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        architecture: str = "convnext_tiny",
        device: str | None = None,
        num_threads: Optional[int] = None,
        gpu_mem_limit_bytes: Optional[int] = None,
    ):
        super().__init__(model_path)
        self.architecture = architecture

        # Configura tetos de CPU threads e VRAM dinamicamente
        configure_system_resources(
            max_threads=num_threads,
            gpu_mem_limit_bytes=gpu_mem_limit_bytes,
        )

        if device is None:
            if torch.cuda.is_available():
                capability = torch.cuda.get_device_capability()
                if capability[0] < 7:
                    # GPU com compute capability antiga (ex: Pascal sm_61) falha com cuDNN moderno
                    print(f"GPU detectada com CC {capability[0]}.{capability[1]} (< 7.0). Usando CPU para estabilidade.")
                    self.device = torch.device("cpu")
                else:
                    self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        print(f"Inicializando {architecture} DINOv3 em dispositivo: {self.device}")
        self.model = timm.create_model(architecture, pretrained=False, num_classes=0)

        # Mapeamento dos pesos do checkpoint original para o schema de nomes do timm
        ckpt = torch.load(self.model_path, map_location="cpu")
        new_state = {}
        for k, v in ckpt.items():
            new_k = k
            if new_k.startswith("downsample_layers.0."):
                new_k = new_k.replace("downsample_layers.0.", "stem.")
            elif new_k.startswith("downsample_layers."):
                parts = new_k.split(".")
                stage_idx = parts[1]
                rest = ".".join(parts[2:])
                new_k = f"stages.{stage_idx}.downsample.{rest}"
            elif new_k.startswith("stages."):
                parts = new_k.split(".")
                stage_idx = parts[1]
                block_idx = parts[2]
                param = ".".join(parts[3:])
                param = param.replace("dwconv.", "conv_dw.")
                param = param.replace("pwconv1.", "mlp.fc1.")
                param = param.replace("pwconv2.", "mlp.fc2.")
                new_k = f"stages.{stage_idx}.blocks.{block_idx}.{param}"
            elif new_k.startswith("norm.") or new_k.startswith("norms.3."):
                new_k = "head.norm." + ".".join(new_k.split(".")[1:])
            new_state[new_k] = v

        res = self.model.load_state_dict(new_state, strict=False)
        if res.missing_keys or res.unexpected_keys:
            print(f"Carga de pesos DINOv3: Missing={len(res.missing_keys)}, Unexpected={len(res.unexpected_keys)}")

        self.model.to(self.device)
        self.model.eval()

    def _forward(self, images: np.ndarray) -> np.ndarray:
        # 1. Transposição de (B, H, W, 3) para (B, 3, H, W)
        chw = images.transpose(0, 3, 1, 2)
        # 2. Normalização ImageNet padrão (já que o PyTorch não tem isso no grafo)
        norm = (chw - IMAGENET_MEAN) / IMAGENET_STD
        tensor = torch.from_numpy(norm.astype(np.float32)).to(self.device)

        with torch.no_grad():
            emb = self.model(tensor)

        return emb.cpu().numpy().astype(np.float32)
