from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
from PIL import Image
import numpy as np

# Padrão: exp<E>_wb<W>_gain<G>_<timestamp>_<weight>.jpg
FILENAME_PATTERN = re.compile(
    r"^(?P<prefix>exp(?P<exp>\d+)_wb(?P<wb>\d+)_gain(?P<gain>\d+))_"
    r"(?P<timestamp>\d{8}T\d{9})"
    r"(?:_(?P<weight>w_\w+))?\.jpg$"
)


@dataclass(frozen=True)
class FrameMetadata:
    path: Path
    sku: str
    config_tag: str  # ex: "exp167_wb4100_gain25"
    exposure: int
    white_balance: int
    gain: int
    timestamp: str
    weight: Optional[str]


def parse_frame_filename(file_path: Path, sku: str) -> Optional[FrameMetadata]:
    """Extrai parâmetros de hardware codificados no nome da imagem."""
    match = FILENAME_PATTERN.match(file_path.name)
    if not match:
        return None

    data = match.groupdict()
    return FrameMetadata(
        path=file_path,
        sku=sku,
        config_tag=data["prefix"],
        exposure=int(data["exp"]),
        white_balance=int(data["wb"]),
        gain=int(data["gain"]),
        timestamp=data["timestamp"],
        weight=data.get("weight"),
    )


def resize_center_crop(image: Image.Image, image_size: int = 224) -> Image.Image:
    """
    Eval geometry: Resize(round(1.14 * size)), then CenterCrop(size).
    Garante o mesmo aspect ratio e escala usados no treino de avaliação.
    """
    image = image.convert("RGB")
    width, height = image.size
    short = int(round(image_size * 1.14))
    if width <= height:
        new_size = (short, int(short * height / width))
    else:
        new_size = (int(short * width / height), short)
    image = image.resize(new_size, resample=Image.Resampling.BILINEAR)
    left = int(round((new_size[0] - image_size) / 2.0))
    top = int(round((new_size[1] - image_size) / 2.0))
    return image.crop((left, top, left + image_size, top + image_size))


def load_dataset_by_config(
    scale_frames_dir: Path | str,
    target_config: str = "exp167_wb4100_gain25",
    min_samples_per_sku: int = 3,
) -> Dict[str, List[FrameMetadata]]:
    """
    Varre data/scale_frames/{sku}/ e filtra apenas imagens com a configuração alvo,
    descartando classes com amostras insuficientes.
    """
    base_dir = Path(scale_frames_dir)
    if not base_dir.is_dir():
        raise FileNotFoundError(f"Diretório não encontrado: {base_dir}")

    dataset: Dict[str, List[FrameMetadata]] = {}

    for sku_dir in sorted(base_dir.iterdir()):
        if not sku_dir.is_dir() or sku_dir.name.startswith("."):
            continue

        sku = sku_dir.name
        sku_frames: List[FrameMetadata] = []

        for img_path in sorted(sku_dir.glob("*.jpg")):
            meta = parse_frame_filename(img_path, sku)
            if meta and meta.config_tag == target_config:
                sku_frames.append(meta)

        if len(sku_frames) >= min_samples_per_sku:
            dataset[sku] = sku_frames

    return dataset
