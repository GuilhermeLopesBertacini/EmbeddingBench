import os
import pytest
import torch

from src.core.resource_guard import (
    get_available_cpu_cores,
    get_available_gpu_memory,
    configure_system_resources,
    get_onnx_session_options,
    get_onnx_providers,
    SystemResourcePlan,
)


def test_get_available_cpu_cores():
    """Valida se o cálculo de cores disponíveis retorna um valor plausível."""
    cores = get_available_cpu_cores(safety_margin=1)
    try:
        total_cores = len(os.sched_getaffinity(0))
    except Exception:
        total_cores = os.cpu_count() or 1

    assert isinstance(cores, int)
    assert 1 <= cores <= total_cores


def test_get_available_gpu_memory():
    """Valida se a consulta de VRAM retorna um inteiro não-negativo."""
    vram = get_available_gpu_memory(device_idx=0, safety_factor=0.8)
    assert isinstance(vram, int)
    assert vram >= 0


def test_configure_system_resources():
    """Valida a geração do plano de recursos e a configuração de tetos."""
    plan = configure_system_resources(max_threads=2, gpu_safety_factor=0.7)

    assert isinstance(plan, SystemResourcePlan)
    assert plan.usable_cpu_threads <= 2
    assert plan.total_cpu_cores >= 1
    assert len(plan.summary()) > 0
    assert "[ResourceGuard]" in plan.summary()

    # Verifica se os limites foram repassados ao PyTorch
    assert torch.get_num_threads() == plan.usable_cpu_threads


def test_onnx_session_options():
    """Valida se as opções de sessão do ONNX Runtime respeitam o número de threads."""
    import onnxruntime as ort

    opts = get_onnx_session_options(num_threads=3)
    assert isinstance(opts, ort.SessionOptions)
    assert opts.intra_op_num_threads == 3
    assert opts.inter_op_num_threads == 1


def test_onnx_providers_always_includes_cpu():
    """Valida se a lista de providers inclui CPU como fallback seguro."""
    providers = get_onnx_providers()
    assert isinstance(providers, list)
    assert "CPUExecutionProvider" in providers
