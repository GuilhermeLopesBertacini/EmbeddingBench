from pathlib import Path
import numpy as np
import pytest

from src.extractors.base import BaseFeatureExtractor
from src.extractors.onnx import ONNXFeatureExtractor


class DummyExtractor(BaseFeatureExtractor):
    """Implementação concreta de teste para validar o contrato da classe base."""

    def __init__(self):
        # Passa um arquivo que existe para passar pela checagem de existência
        super().__init__(__file__)

    def _forward(self, images: np.ndarray) -> np.ndarray:
        # Retorna intencionalmente vetores com norma != 1.0 (magnitude constante = 5.0)
        batch_size = images.shape[0]
        return np.ones((batch_size, 64), dtype=np.float32) * 5.0


# ==============================================================================
# Testes Unitários de Contrato (BaseFeatureExtractor)
# ==============================================================================

def test_base_extractor_enforces_unit_norm():
    """Valida se o Template Method garante norma unitária no retorno."""
    extractor = DummyExtractor()
    dummy_images = np.random.rand(4, 224, 224, 3).astype(np.float32)

    embeddings = extractor.extract(dummy_images)

    assert embeddings.shape == (4, 64)
    assert embeddings.dtype == np.float32

    # A norma L2 de cada linha deve ser estritamente 1.0
    norms = np.linalg.norm(embeddings, axis=-1)
    np.testing.assert_allclose(norms, 1.0, rtol=1e-5, atol=1e-6)


def test_base_extractor_raises_on_non_existent_path():
    """Valida se caminhos inexistentes disparam FileNotFoundError imediatamente."""
    class ConcreteTestExtractor(BaseFeatureExtractor):
        def _forward(self, images: np.ndarray) -> np.ndarray:
            return np.zeros((images.shape[0], 10), dtype=np.float32)

    with pytest.raises(FileNotFoundError):
        ConcreteTestExtractor("/caminho/completamente/invalido/modelo.onnx")


@pytest.mark.parametrize(
    "invalid_input, error_reason",
    [
        (np.zeros((224, 224, 3), dtype=np.float32), "Shape 3D em vez de 4D (faltando batch)"),
        (np.zeros((2, 224, 224, 1), dtype=np.float32), "Canais != 3"),
        (np.zeros((2, 224, 224, 4), dtype=np.float32), "Canais != 3 (RGBA)"),
        (np.zeros((2, 224, 224, 3), dtype=np.uint8), "Dtype inteiro em vez de float32"),
        (np.zeros((2, 224, 224, 3), dtype=np.float64), "Dtype float64 em vez de float32"),
    ],
)
def test_base_extractor_rejects_invalid_inputs(invalid_input, error_reason):
    """Valida a blindagem dos invariantes de entrada contra shapes e dtypes incorretos."""
    extractor = DummyExtractor()
    with pytest.raises(ValueError, match="Images must have shape"):
        extractor.extract(invalid_input)


# ==============================================================================
# Testes de Integração (ONNXFeatureExtractor)
# ==============================================================================

def test_onnx_feature_extractor_with_real_model():
    """
    Testa a inferência real de ponta a ponta com o modelo fine-tuned.
    Pula o teste caso os pesos locais não estejam presentes.
    """
    model_path = Path("data/models/fine-tuned.onnx")
    if not model_path.exists():
        pytest.skip(f"Modelo não encontrado em: {model_path}")

    extractor = ONNXFeatureExtractor(model_path)

    # Simula um batch de 2 imagens no padrão do contrato: (B, H, W, 3), float32, [0, 1]
    batch_size = 2
    dummy_batch = np.random.rand(batch_size, 224, 224, 3).astype(np.float32)

    embeddings = extractor.extract(dummy_batch)

    # MobileNetV4 Medium fine-tuned exportado tem saída de dimensão 256
    assert embeddings.shape == (batch_size, 256)
    assert embeddings.dtype == np.float32

    # Valida se a saída é normalizada no espaço de cosseno
    norms = np.linalg.norm(embeddings, axis=-1)
    np.testing.assert_allclose(norms, 1.0, rtol=1e-5, atol=1e-6)
