from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any, Dict, List, Optional
import psutil
import torch


@dataclass(frozen=True)
class SystemResourcePlan:
    """Plano de alocação de recursos calculado dinamicamente."""
    total_cpu_cores: int
    usable_cpu_threads: int
    current_cpu_load: float
    cuda_available: bool
    gpu_name: Optional[str]
    total_gpu_vram_bytes: int
    free_gpu_vram_bytes: int
    usable_gpu_vram_bytes: int

    def summary(self) -> str:
        gpu_info = "Não disponível / Incompatível"
        if self.cuda_available and self.gpu_name:
            free_mb = self.free_gpu_vram_bytes / (1024 * 1024)
            usable_mb = self.usable_gpu_vram_bytes / (1024 * 1024)
            gpu_info = f"{self.gpu_name} (Livre: {free_mb:.0f} MB | Teto Seguro: {usable_mb:.0f} MB)"

        return (
            f"[ResourceGuard] CPU Cores Totais: {self.total_cpu_cores} | "
            f"Carga 1m: {self.current_cpu_load:.2f} | "
            f"Threads Alocadas: {self.usable_cpu_threads} | "
            f"GPU: {gpu_info}"
        )


def get_available_cpu_cores(safety_margin: int = 1) -> int:
    """
    Calcula quantos cores podem ser utilizados considerando a afinidade
    do processo atual e a carga média do sistema (load average).
    """
    try:
        affinity_cores = len(os.sched_getaffinity(0))
    except (AttributeError, NotImplementedError):
        affinity_cores = os.cpu_count() or 1

    try:
        load_1m, _, _ = os.getloadavg()
    except (AttributeError, OSError):
        load_1m = 0.0

    # Cores efetivamente desocupados
    idle_cores = max(1, int(affinity_cores - load_1m))
    usable_cores = max(1, idle_cores - safety_margin)
    return min(usable_cores, affinity_cores)


def get_available_gpu_memory(device_idx: int = 0, safety_factor: float = 0.8) -> int:
    """
    Retorna a quantidade de bytes de VRAM livre que podemos alocar
    aplicando uma margem de segurança para evitar OOM.
    """
    if not torch.cuda.is_available():
        return 0

    try:
        free_bytes, _ = torch.cuda.mem_get_info(device_idx)
        return int(free_bytes * safety_factor)
    except Exception:
        return 0


def configure_system_resources(
    max_threads: Optional[int] = None,
    gpu_mem_limit_bytes: Optional[int] = None,
    gpu_device_idx: int = 0,
    gpu_safety_factor: float = 0.8,
) -> SystemResourcePlan:
    """
    Detecta os recursos ociosos, configura tetos no PyTorch e devolve
    o plano de recursos para orquestrar o pipeline e o ONNX Runtime.
    """
    try:
        total_cores = len(os.sched_getaffinity(0))
    except Exception:
        total_cores = os.cpu_count() or 1

    try:
        load_1m, _, _ = os.getloadavg()
    except Exception:
        load_1m = 0.0

    if max_threads is not None and max_threads > 0:
        usable_threads = min(max_threads, total_cores)
    else:
        usable_threads = get_available_cpu_cores(safety_margin=1)

    # 1. Configura limites de threads na CPU para o PyTorch
    torch.set_num_threads(usable_threads)
    torch.set_num_interop_threads(max(1, usable_threads // 2))

    # 2. Configura limites de VRAM se houver GPU CUDA válida
    cuda_available = torch.cuda.is_available()
    gpu_name = None
    total_vram = 0
    free_vram = 0
    usable_vram = 0

    if cuda_available:
        try:
            capability = torch.cuda.get_device_capability(gpu_device_idx)
            # GPUs com compute capability < 7.0 não são compatíveis com PyTorch moderno
            if capability[0] < 7:
                cuda_available = False
            else:
                gpu_name = torch.cuda.get_device_name(gpu_device_idx)
                free_vram, total_vram = torch.cuda.mem_get_info(gpu_device_idx)
                
                if gpu_mem_limit_bytes is not None and gpu_mem_limit_bytes > 0:
                    usable_vram = min(gpu_mem_limit_bytes, free_vram)
                else:
                    usable_vram = int(free_vram * gpu_safety_factor)

                if total_vram > 0 and usable_vram > 0:
                    fraction = min(1.0, max(0.01, usable_vram / total_vram))
                    torch.cuda.set_per_process_memory_fraction(fraction, device=gpu_device_idx)
        except Exception:
            cuda_available = False

    plan = SystemResourcePlan(
        total_cpu_cores=total_cores,
        usable_cpu_threads=usable_threads,
        current_cpu_load=load_1m,
        cuda_available=cuda_available,
        gpu_name=gpu_name,
        total_gpu_vram_bytes=total_vram,
        free_gpu_vram_bytes=free_vram,
        usable_gpu_vram_bytes=usable_vram,
    )

    print(plan.summary())
    return plan


def get_onnx_session_options(num_threads: Optional[int] = None) -> Any:
    """Configura SessionOptions do ONNX Runtime para respeitar o teto de threads."""
    import onnxruntime as ort

    threads = num_threads or get_available_cpu_cores(safety_margin=1)
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return opts


def get_onnx_providers(
    gpu_mem_limit_bytes: Optional[int] = None,
    device_id: int = 0,
) -> List[Any]:
    """
    Retorna a lista de execution providers com limites de memória rígidos
    para o CUDAExecutionProvider (se disponível).
    """
    import onnxruntime as ort

    available = ort.get_available_providers()
    providers: List[Any] = []

    if "CUDAExecutionProvider" in available:
        usable_mem = gpu_mem_limit_bytes or get_available_gpu_memory(device_idx=device_id, safety_factor=0.8)
        cuda_options: Dict[str, Any] = {
            "device_id": device_id,
            "arena_extend_strategy": "kSameAsRequested",
        }
        if usable_mem > 0:
            cuda_options["gpu_mem_limit"] = usable_mem

        providers.append(("CUDAExecutionProvider", cuda_options))

    providers.append("CPUExecutionProvider")
    return providers
