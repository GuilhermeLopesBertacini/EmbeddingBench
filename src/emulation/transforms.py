from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass(frozen=True)
class EmulationParams:
    """
    Parâmetros relativos de emulação física/ISP:
    - gamma_delta: ajuste exponencial de gamma (ex: 1.0 = inalterado, <1.0 = abre sombras, >1.0 = aprofunda)
    - contrast_delta: fator de escala de contraste linear (1.0 = inalterado, >1.0 = mais contraste)
    - saturation_delta: fator de saturação cromática (1.0 = inalterado, >1.0 = cores mais vivas)
    - ccm: matriz opcional 3x3 para rotação/balanceamento de canais de cor
    """
    gamma_delta: float = 1.0
    contrast_delta: float = 1.0
    saturation_delta: float = 1.0
    ccm: Optional[np.ndarray] = None

    def is_identity(self) -> bool:
        return (
            abs(self.gamma_delta - 1.0) < 1e-4
            and abs(self.contrast_delta - 1.0) < 1e-4
            and abs(self.saturation_delta - 1.0) < 1e-4
            and self.ccm is None
        )


def apply_emulation(
    image_rgb: np.ndarray,
    params: EmulationParams,
) -> np.ndarray:
    """
    Aplica emulação de ISP sobre imagem RGB normalizada em [0.0, 1.0].
    
    Ordem das transformações físicas:
    1. Ajuste de Gamma (em escala de intensidade)
    2. Ajuste de Contraste (estiramento afim em torno de 0.5)
    3. Ajuste de Saturação (interpolação entre luminância Rec.709 e cores)
    4. Color Correction Matrix 3x3 (se fornecida)
    5. Clipping de segurança em [0.0, 1.0]
    """
    if params.is_identity():
        return image_rgb

    out = image_rgb.astype(np.float32).copy()

    # 1. Gamma
    if abs(params.gamma_delta - 1.0) >= 1e-4:
        out = np.power(np.maximum(out, 0.0), 1.0 / params.gamma_delta)

    # 2. Contraste (centrado em 0.5)
    if abs(params.contrast_delta - 1.0) >= 1e-4:
        out = (out - 0.5) * params.contrast_delta + 0.5

    # 3. Saturação (Rec.709: Y = 0.2126 R + 0.7152 G + 0.0722 B)
    if abs(params.saturation_delta - 1.0) >= 1e-4:
        gray = 0.2126 * out[..., 0] + 0.7152 * out[..., 1] + 0.0722 * out[..., 2]
        gray = np.expand_dims(gray, axis=-1)
        out = gray + params.saturation_delta * (out - gray)

    # 4. Color Correction Matrix 3x3
    if params.ccm is not None:
        out = np.matmul(out, params.ccm.T)

    return np.clip(out, 0.0, 1.0).astype(np.float32)
