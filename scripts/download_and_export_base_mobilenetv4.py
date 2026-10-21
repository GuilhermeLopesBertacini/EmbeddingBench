#!/usr/bin/env python3
"""
Faz o download dos pesos pré-treinados do modelo base (ImageNet)
definido em AirisML/airis_deeplearning/training/configs/train_scale_embedding.yaml:
  backbone_name: mobilenetv4_conv_medium
  pretrained: true

Exporta para ONNX em data/models/mobilenetv4_conv_medium_pretrained.onnx
com a mesma assinatura do modelo fine-tuned:
  Input:  'pixel_values' [batch, 3, 224, 224] (RGB raw 0-255)
  Normalização interna: (pixel_values / 255.0 - mean) / std
  Output: 'embeddings' [batch, feature_dim] (L2-normalizado)
"""

from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class BaseMobileNetV4Wrapper(nn.Module):
    """
    Empacota o backbone MobileNetV4 pré-treinado no ImageNet.
    Aplica internamente a normalização do ImageNet sobre pixels 0-255
    e realiza normalização L2 nos embeddings de saída, exatamente como
    o callback de exportação do AirisML.
    """

    def __init__(
        self,
        backbone_name: str = "mobilenetv4_conv_medium",
        pretrained: bool = True,
        input_mean: list[float] = [0.485, 0.456, 0.406],
        input_std: list[float] = [0.229, 0.224, 0.225],
        input_scale: float = 255.0,
    ):
        super().__init__()
        print(f"Baixando/carregando backbone '{backbone_name}' pré-treinado no ImageNet via timm...")
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )
        self.register_buffer("mean", torch.tensor(input_mean, dtype=torch.float32).view(1, -1, 1, 1))
        self.register_buffer("std", torch.tensor(input_std, dtype=torch.float32).view(1, -1, 1, 1))
        self.input_scale = input_scale

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # Normalização idêntica à do AirisML
        x = (pixel_values / self.input_scale - self.mean) / self.std
        features = self.backbone(x)
        # Normalização L2
        embeddings = F.normalize(features, p=2, dim=1)
        return embeddings


def export_base_model():
    output_dir = Path("data/models")
    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = output_dir / "mobilenetv4_conv_medium_pretrained.onnx"

    print("Inicializando modelo base pré-treinado...")
    wrapper = BaseMobileNetV4Wrapper(
        backbone_name="mobilenetv4_conv_medium",
        pretrained=True,
    )
    wrapper.eval()

    dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float32)

    print(f"Exportando grafo ONNX para {onnx_path}...")
    torch.onnx.export(
        wrapper,
        dummy_input,
        str(onnx_path),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["pixel_values"],
        output_names=["embeddings"],
        dynamic_axes={
            "pixel_values": {0: "batch"},
            "embeddings": {0: "batch"},
        },
        dynamo=False,
    )

    import onnx
    onnx_model = onnx.load(str(onnx_path))
    meta = onnx_model.metadata_props.add()
    meta.key = "task"
    meta.value = "embedding_base"
    meta_bb = onnx_model.metadata_props.add()
    meta_bb.key = "backbone"
    meta_bb.value = "mobilenetv4_conv_medium"
    onnx.save(onnx_model, str(onnx_path))

    print(f"✅ Modelo base pré-treinado pronto em: {onnx_path}")
    print(f"Tamanho do arquivo: {onnx_path.stat().st_size / (1024 * 1024):.2f} MB")


if __name__ == "__main__":
    export_base_model()
