# EmbeddingBench

Laboratório de benchmarking geométrico de representações latentes e otimização de parâmetros de ISP (emulação de câmera) para a balança de autoatendimento.

---

## 🚀 Como Executar

### 1. Instalação do Ambiente
Gerenciado via `uv`:
```bash
uv sync --extra dev
```

### 2. Execução da Busca de Emulação (Grid Search)

#### A. Modelo Fine-Tuned (MobileNetV4 treinado com ArcFace + Triplet Loss)
```bash
uv run python scripts/run_emulation_search.py --model-type onnx
```
*Gera: `outputs/emulation_search_finetuned_onnx.csv`*

#### B. Modelo Base Pretrained (ImageNet MobileNetV4 Medium)
```bash
uv run python scripts/run_emulation_search.py --model-type mobilenetv4_base
```
*Gera: `outputs/emulation_search_mobilenetv4_base_pretrained.csv`*

#### C. Foundation Model (DINOv3 ConvNeXt Tiny PyTorch)
```bash
uv run python scripts/run_emulation_search.py --model-type dinov3_tiny
```
*Gera: `outputs/emulation_search_dinov3_tiny.csv`*

---

### 3. Gerar Gráficos Comparativos (t-SNE 2D)

Gera as projeções 2D comparando:
1. **Base Pretrained (ImageNet) vs Fine-Tuned (Balança)** no Baseline puro.
2. **Fine-Tuned Baseline vs Fine-Tuned com Emulação Otimizada**.

```bash
uv run python scripts/generate_cluster_plots.py
```
*Os gráficos são salvos em `outputs/plots/`.*

---

### 4. Rodar a Suíte de Testes
```bash
uv run pytest tests/
```

---

## 🛡️ Gestão Inteligente de Hardware (Resource Guard)

O pipeline conta com o módulo `src/core/resource_guard.py` que afere dinamicamente o estado da máquina antes de alocar recursos:

* **CPU Cores:** Em vez de usar todos os cores cegamente, calcula os cores efetivamente desocupados baseando-se no `load average` e na afinidade do processo (`sched_getaffinity`), reservando uma margem de segurança para o sistema operacional.
* **GPU / VRAM:** Consulta a VRAM livre em tempo real (`torch.cuda.mem_get_info`) e aplica uma margem de segurança (padrão: 80% do livre) para evitar *Out of Memory* (OOM).
* **Parâmetros de Linha de Comando:**
  ```bash
  # Limitar manualmente a 4 threads e teto de 1500 MB de VRAM:
  uv run python scripts/run_emulation_search.py --max-threads 4 --gpu-mem-limit-mb 1500
  ```

---

## 📐 Métricas de Avaliação dos Clusters

Todos os vetores são L2-normalizados na hiperesfera unitária ($\|v\|_2 = 1.0$), avaliando a similaridade por cosseno:

1. **`silhouette_cosine`**: Coesão intra-classe vs separação inter-classe $[-1, 1]$ (maior é melhor).
2. **`fisher_ratio`**: Razão entre a separação entre centróides de classes distintas e a dispersão interna das amostras ao redor de seu centróide (maior é melhor).
3. **`davies_bouldin`**: Razão de similaridade entre clusters (menor é melhor).
4. **`calinski_harabasz`**: Razão de dispersão de variância entre clusters vs dentro dos clusters (maior é melhor).
