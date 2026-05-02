# EdgeLLM: Efficient Deployment of Large Language Models on Resource-Constrained Edge Devices

**Aryan Kumar Singh (26769) · Bhavesh Sukhariya (26042) · Kavya Duvvuri (27494) · Lavish Singh (26318)**

*Edge AI — Course Project Report*

🔗 [https://github.com/KavyaD02/EdgeLLM](https://github.com/KavyaD02/EdgeLLM)

---

## Abstract

LLMs exhibit impressive functionality but are impractical for deployment outside cloud environments owing to massive memory and computational overhead. This paper introduces **EdgeLLM**, a full-fledged model compression framework to make LLMs practical on commodity edge devices such as the Raspberry Pi 5. We propose a three-step compression pipeline tailored to the Qwen2-0.5B model: (1) *SparseGPT with Layer-Aware Sparsity Sensitivity (LASS)*, an efficient sparse pruning algorithm that assigns sparsity budgets to individual layers according to their output tolerance to sparsity; (2) *Knowledge Distillation (KD)*, which mitigates accuracy loss resulting from sparse pruning; and (3) *SG-GPTQ with NF4 4-bit quantisation*, an effective compression technique that scales down the model size by 4.4× to 215 MB from the 942 MB of the original model. The final model demonstrates a perplexity score of 21.58 on WikiText-2, merely +3.49 more than the baseline model.

---

## 1. Problem Statement, Motivation & Objectives

The fast development of LLMs has led to the creation of systems with impressive abilities — ranging from code generation to multilingual reasoning — but their usage has been limited to cloud-based implementations. Even smaller production-size open models consume several gigabytes of memory and require GPU-based computation for interactive inference. Such resource requirements pose formidable obstacles for tasks that require fast inference, privacy protection, or offline inference capabilities — characteristics typical of edge applications, IoT implementations, and mobile apps.

Edge AI circumvents such obstacles through the design of model compression with the target inference hardware architecture. The critical task is to traverse the *quality-efficiency trade-off curve*: aggressive compression is resource-efficient but deteriorates model quality, whereas mild compression maintains quality but fails to fit the model on constrained devices. Thus, a successful model compression pipeline should be principled, well-ordered, and mutually supportive at each stage.

**Key objectives of this project:**

- Implement **LASS** (Layer-Aware Sparsity Sensitivity), an extension of SparseGPT using a heterogeneous pruning strategy guided by sensitivity, which allocates more efficient sparsity budgets for different layers.
- Show **Knowledge Distillation after Pruning**, which retains sparsity masks to compensate for the loss of perplexity resulting from removing weights.
- Use **GPTQ/NF4 4-bit quantization** to compress the model by a factor of 4.4×, enabling deployment on a Raspberry Pi.
- Deliver a fully **reproducible end-to-end workflow** — from calibration to compression and inference using `llama.cpp` — tested on Qwen2-0.5B, TinyLlama-1.1B, and OPT-1.3B.
- Demonstrate that capable LMs can run **offline on a 8 GB Raspberry Pi 5** in real time without any GPU.

---

## 2. Proposed Solution (Overview)

EdgeLLM compresses pre-trained LLMs through three sequential stages, each targeting a different aspect of the efficiency-quality trade-off.

```
┌─────────────────┐     ┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Qwen2-0.5B    │────▶│  SparseGPT+LASS  │────▶│    Knowledge     │────▶│   GPTQ + NF4    │────▶│   Edge Model    │
│   Dense         │     │   Pruning        │     │  Distillation    │     │   4-bit          │     │   (215 MB)      │
│   (942 MB)      │     │                  │     │                  │     │                  │     │   RPi Deploy    │
├─────────────────┤     ├──────────────────┤     ├──────────────────┤     ├──────────────────┤     ├─────────────────┤
│ PPL: 18.10      │     │ PPL: 19.67       │     │ PPL: 19.38       │     │ PPL: 21.58       │     │ 4.4× smaller    │
│ Baseline        │     │ Sparsity: 21.96% │     │ Recovered +0.29  │     │ Size: 215 MB     │     │ llama.cpp       │
└─────────────────┘     └──────────────────┘     └──────────────────┘     └──────────────────┘     └─────────────────┘
```

**Data → Model → Deployment → Output:**

1. **Data:** 128 calibration samples are taken from WikiText-2 (with 2048 tokens each); this data set serves as training for all the compression stages; the WikiText-2 test partition is utilized for evaluating perplexity.
2. **Model:** The dense version of Qwen2-0.5B checkpoint is subjected to compression in the following sequence: (i) pruning, based on LASS, and achieving ~22% weight reduction through sensitivity-aware budgets; (ii) knowledge distillation with the aim of recovering perplexity degradation caused by pruning; (iii) quantization through GPTQ+NF4 into 4-bit weights.
3. **Deployment:** Quantized weights are converted to GGUF format and deployed using `llama.cpp` framework on Raspberry Pi 5.
4. **Output:** Real-time text generation on the edge device, fully offline, with no GPU requirement.

---

## 3. Hardware & Software Setup

### Hardware

| Component | Specification |
|---|---|
| Training / Compression | Kaggle Cloud — NVIDIA Tesla T4 (16 GB VRAM), 29 GB RAM |
| Edge Inference Device | Raspberry Pi 5 Model B, 8 GB LPDDR4X RAM |
| Edge CPU | ARM Cortex-A76 @ 2.4 GHz (quad-core) |
| Edge Storage | 32 GB microSD card (Class 10) |

### Software

| Tool / Framework | Purpose |
|---|---|
| Python 3.12 | Core language for compression pipeline |
| PyTorch 2.x | Model loading, forward passes, gradient computation |
| HuggingFace `transformers` | Qwen2-0.5B / TinyLlama / OPT-1.3B checkpoint management |
| `bitsandbytes` | NF4 quantisation support |
| `llama.cpp` | Optimised CPU inference, GGUF format |
| WikiText-2 (`datasets`) | Calibration and evaluation data |
| Kaggle Notebooks | Cloud GPU environment for training |
| Raspberry Pi OS 64-bit (Bookworm) | Edge operating system |

---

## 4. Data Collection & Dataset Preparation

All compression and evaluation stages use the publicly available **WikiText-2** dataset, a clean English Wikipedia corpus commonly used in language modelling research.

| Split | Samples | Usage |
|---|---|---|
| Calibration (train) | 128 | LASS sensitivity scan, SparseGPT Hessian estimation, GPTQ calibration |
| KD fine-tuning (train) | Full train split | Knowledge distillation training |
| Evaluation (test) | Full test split | Perplexity measurement |

**Preprocessing:**
- The calibration texts are tokenized using the tokenizer of the model and then cut or padded to **2048 tokens**.
- For the sensitivity estimation in the LASS test, only **4 calibration texts** are considered to limit computational cost while obtaining an accurate sensitivity measurement.
- Perplexity evaluation is performed with **step 512 tokens** on the entire WikiText-2 dataset without any boundary effects.
- No further processing or data augmentation techniques are used; the original Wikipedia texts cover all necessary calibration cases.

---

## 5. Model Design, Training & Evaluation

### 5.1 Base Model

The language model that we employ is **Qwen2-0.5B**. This language model comes with a total of 0.5 billion parameters and adopts the causal approach for modeling. The model includes the following design elements: GQA (grouped-query attention), SwiGLU activation, RoPE (rotary position embedding), and 24 layers. Even with such a small size, this model performs well on benchmarks. Additional experiments are reported for **TinyLlama-1.1B** and **OPT-1.3B** to validate generalisability.

### 5.2 Stage 1: SparseGPT with LASS Pruning

SparseGPT performs one-shot unstructured pruning using the Optimal Brain Surgeon (OBS) framework. For each linear layer with weight **W**, it solves:

$$\min_{\hat{W}} \|WX - \hat{W}X\|_F^2 \quad \text{s.t.} \quad \|\hat{W}\|_0 \leq (1-s) \cdot d_{\text{out}} \cdot d_{\text{in}}$$

where **X** is the layer's input activation and $s$ is the target sparsity.

**LASS** extends SparseGPT with a two-phase sensitivity-aware budget allocation:

**Phase 1 — Sensitivity Pre-scan.** For each transformer layer $\ell$, we measure the relative output error from temporarily pruning each linear sublayer in place:

$$\epsilon_\ell = \frac{1}{|\mathcal{N}_\ell|} \sum_{n \in \mathcal{N}_\ell} \frac{\|h_{\text{orig}} - h_{\text{pruned}}^{(n)}\|_2}{\|h_{\text{orig}}\|_2 + \varepsilon}$$

It is important to note that the sensitivity of the output $h_{\text{pruned}}^{(n)}$ is calculated *at the moment of pruning the weights* and not at the time of unpruning the weights; this is the crucial step that enables LASS to function correctly, and without this, all sensitivities become equal to zero.

**Phase 2 — Budget Allocation.** Inversing the sensitivities and clipping them to the interval $[s_{\min}, s_{\max}] = [0.10, 0.80]$, and rescaled so the parameter-weighted mean matches the global target $s_{\text{target}} = 0.50$.

### 5.3 Stage 2: Knowledge Distillation

The pruned student model is fine-tuned against the dense original teacher using a combined loss:

$$\mathcal{L}_{\text{KD}} = \alpha \cdot \mathcal{L}_{\text{CE}}(y, \hat{y}_s) + (1-\alpha) \cdot \mathcal{L}_{\text{KL}}(\hat{p}_t \| \hat{p}_s)$$

with temperature $T=2$, weight $\alpha=0.5$, and 3 training epochs. A critical novelty: **sparsity masks are re-applied after every gradient step**, preventing gradient updates from silently filling pruned-zero weights.

### 5.4 Stage 3: GPTQ + NF4 4-bit Quantisation

GPTQ performs quantisation on each weight matrix layer-wise with approximate second-order information by updating each column using the Cholesky decomposition. NF4 quantises the weights to the nearest 4-bit numbers spaced according to the quantiles of the standard normal distribution, leading to minimal quantisation error for normally distributed transformer weights. The double quantisation step reduces the scale factors too. The GPTQ+NF4 workflow outputs in GGUF Q4\_K\_M format for `llama.cpp`.

### 5.5 Evaluation Metric

Perplexity (PPL) on the WikiText-2 test set:

$$
\mathrm{PPL} = \exp\left(-\frac{1}{N}\sum_{i=1}^{N}\log p_\theta(x_i \mid x_{<i})\right)
$$

Lower PPL indicates better language modelling quality. Stride 512 is used for all evaluations.

### Compression Hyperparameters

| Stage | Parameter | Value |
|---|---|---|
| SparseGPT + LASS | Global sparsity target | 0.50 |
| | Min / max layer sparsity | 0.10 / 0.80 |
| | Block size | 128 |
| | Damping factor λ | 0.01 |
| | LASS scan samples | 4 |
| Knowledge Distillation | Temperature T | 2.0 |
| | KD weight α | 0.5 |
| | Training epochs | 3 |
| Quantisation | GPTQ bits | 4 |
| | NF4 double quantisation | Enabled |

---

## 6. Model Compression & Efficiency Metrics

### 6.1 Main Results — Qwen2-0.5B

*Full ablation results on Qwen2-0.5B (WikiText-2 PPL, lower is better)*

| Model | PPL ↓ | ΔPPL | Sparsity | Size | Compression |
|---|---|---|---|---|---|
| Original (dense) | 18.10 | — | 0.00% | 942 MB | 1.0× |
| Uniform SparseGPT | 19.85 | +1.75 | 21.96% | 942 MB | 1.0× |
| **LASS Pruned (ours)** | **19.67** | +1.57 | 21.96% | 942 MB | 1.0× |
| After KD | 19.38 | +1.28 | 21.96% | 942 MB | 1.0× |
| SG-GPTQ (4-bit) | 19.36 | +1.26 | 22.31% | 942 MB | 1.0× |
| **NF4 4-bit (final)** | **21.58** | +3.49 | 22.31% | **215 MB** | **4.4×** |

### 6.2 Stage-by-Stage Contribution

| Stage | PPL Change | Interpretation |
|---|---|---|
| LASS vs. Uniform SparseGPT | −0.18 | Smarter sparsity allocation |
| KD after pruning | −0.29 | Teacher-guided recovery |
| GPTQ 4-bit | −0.02 | Near-lossless quantisation |
| NF4 4-bit (size reduction) | +2.22 | Traded for 4.4× compression |
| **Net vs. dense baseline** | **+3.49** | **215 MB deployable model** |

### 6.3 Cross-Model Results

| Model | Variant | PPL ↓ | ΔPPL | Sparsity | Size |
|---|---|---|---|---|---|
| TinyLlama-1.1B | Original | 10.25 | — | 0.02% | 2.98 GB |
| | LASS+KD+GPTQ | 10.98 | +0.72 | 26.5% | 2.98 GB |
| | NF4 4-bit | 11.89 | +1.63 | 26.5% | 356 MB |
| OPT-1.3B | Original | 21.26 | — | 0.00% | 2.5 GB |
| | LASS+KD+GPTQ | 21.50 | +0.24 | 26.76% | 2.5 GB |
| | NF4 4-bit | 22.05 | +0.79 | 26.76% | 391 MB |

### 6.4 Trade-offs Observed

The key trade-off is that of model size versus quality of perplexity. Quantisation using NF4 results in the largest perplexity penalty (+2.22 for Qwen2), while achieving the biggest reduction in size (4.4×). The LASS (unstructured sparsity) method doesn't lead to faster inference on CPU when using vanilla PyTorch, as the 942 MB sparse model is just as slow as the denser version. Only after applying GPTQ + NF4 is inference on the Raspberry Pi 5 possible.

---

## 7. Model Deployment & On-Device Performance

### 7.1 Deployment Steps

1. **Export:** Save the NF4 4-bit model from the KD+GPTQ pipeline in HuggingFace format.
2. **Convert:** Convert to GGUF Q4\_K\_M using `llama.cpp`'s conversion utility:
```bash
python convert_hf_to_gguf.py ./nf4_model \
    --outfile qwen2-0.5b-edge-q4_0.gguf \
    --outtype q4_0
```
3. **Transfer:** Copy the 215 MB GGUF file to the Raspberry Pi 5 via `scp`.
4. **Build:** Compile `llama.cpp` on the Raspberry Pi 5 (ARM-optimised NEON build).
5. **Serve:** Run inference with a fixed 512-token context window to bound KV-cache memory:
```bash
./llama-cli \
    -m qwen2-0.5b-edge-q4_0.gguf \
    --ctx-size 512  --threads 4  \
    --temp 0.7  --repeat-penalty 1.1 \
    -p "Your prompt here"
```

### 7.2 Deployment Stack on Raspberry Pi 5

```
┌─────────────────────────────────────────┐
│       Application / Interactive CLI      │
├─────────────────────────────────────────┤
│  GGUF Q4_K_M — qwen2-0.5b-edge (215 MB) │
├─────────────────────────────────────────┤
│    llama.cpp — ARM-optimised NEON build  │
├─────────────────────────────────────────┤
│   Raspberry Pi OS 64-bit (Bookworm)      │
├─────────────────────────────────────────┤
│  Raspberry Pi 5 — ARM A76 @2.4GHz 8GB   │
└─────────────────────────────────────────┘
```

### 7.3 Memory Footprint on Raspberry Pi 5

| Component | Memory |
|---|---|
| Model weights (GGUF Q4) | ~215 MB |
| KV cache (512 ctx) | ~180 MB |
| `llama.cpp` runtime + OS overhead | ~300 MB |
| **Total** | **~695 MB** |
| Available headroom | >7.3 GB |

### 7.4 Inference Performance

| Metric | Value |
|---|---|
| Token throughput | 3–8 tokens/sec |
| Time to first token | 5–10 s |
| CPU utilisation (all cores) | ~380% |
| Peak temperature | 55–72 °C |
| Context window | 512 tokens (fixed) |

The model generates coherent, contextually relevant text in real time with no GPU or cloud connectivity required.

---

## 8. Conclusions & Limitations

### 8.1 Key Outcomes

EdgeLLM has shown that even a powerful generative language model can be pruned and run on a Raspberry Pi 5 following a rigorous three-stage pipeline. The final, 215 MB NF4 model attains the following results:

- Perplexity of **21.58** — a mere 19% increase from the baseline model which weighs in at 942 MB.
- **4.4×** compression — easily fitting under the 8 GB limit of the Raspberry Pi 5's memory.
- Generation at a rate of **3–8 tokens/sec** on the Raspberry Pi 5's quad-core ARM CPU.
- **No need for GPU, cloud, or even an internet connection** during deployment.

LASS yields a consistent PPL gain of −0.18 compared to the uniform SparseGPT with the same level of sparsity. The knowledge distillation phase brings back another −0.29 PPL gain using the teacher's soft target outputs. Cross-model evaluations on TinyLlama-1.1B and OPT-1.3B confirm the generalisability of the pipeline.

### 8.2 Limitations

- **Unstructured sparsity:** The LASS technique yields unstructured sparse weights that do not speed up inference on CPUs because of their lack of sparse-friendly kernels. The 22% sparsity does not provide any latency benefits in regular runtimes; only the GPTQ/NF4 compression decreases latency since it makes the model more compact.
- **Perplexity gap:** The trained model underperforms the dense model by 3.49 PPL points. This difference can be critical when fine-tuning the model for downstream applications.
- **Fixed context window:** The inference context is fixed at 512 tokens to limit KV-cache memory usage, thus restricting inference on documents exceeding this size and multi-turn interactions.
- **No downstream task evaluation:** The study presents results only for WikiText-2 perplexity; generalization for MMLU, HumanEval, or other datasets was not demonstrated.
- **Knowledge distillation in large models:** Knowledge distillation did not help reduce perplexity in TinyLlama-1.1B or OPT-1.3B; its benefit seems to apply only to small-scale models.

---

## 9. Future Work

- **Structured pruning:** Instead of using unstructured pruning, head pruning and channel pruning will produce hardware-efficient compressions and latency reductions on the CPU.
- **QLoRA tuning post-quantisation:** The application of LoRA fine-tuning to the quantised model could further reduce the perplexity gap.
- **Speculative decoding:** Running the edge model along with a draft model on the Raspberry Pi 5 could increase the number of tokens processed per second.
- **Downstream benchmarks:** Conducting experiments on MMLU, BoolQ, and HumanEval datasets can give a comprehensive overview of the model's practicality.
- **Dynamic context management:** Sliding window attention and context summarisation techniques can help solve the context length issue by avoiding any increase in peak memory usage.

---

## 10. Challenges & Mitigation

| Challenge | Root Cause | Mitigation |
|---|---|---|
| LASS sensitivities all zero (every layer looks equally robust) | Sensitivity measured *after* weight restoration instead of while weights remain pruned | Corrected measurement order: prune → measure output error → restore; verified by observing non-zero ε values across layers |
| KD erasing pruning structure | Gradient updates silently fill pruned-zero entries | Re-apply binary sparsity masks after every optimiser step during KD fine-tuning |
| CUDA OOM on Kaggle T4 during distillation | Full-precision teacher + student in memory simultaneously | Load teacher in FP16, student in FP32; batch size 1 with gradient accumulation |
| llama.cpp build fails on Raspberry Pi OS | Missing ARM NEON headers in default toolchain | Install `build-essential` and `libopenblas-dev`; build with `LLAMA_OPENBLAS=1` |
| Perplexity spikes with very high NF4 compression | NF4 quantisation error amplified by earlier unstructured sparsity | Fixed global sparsity target at 0.50; LASS clips individual layers to [0.10, 0.80] to prevent catastrophic layers |

---

## 11. References

1. Han, S., Pool, J., Tran, J., & Dally, W. (2015). Learning both weights and connections for efficient neural networks. *NeurIPS*.
2. Hassibi, B., & Stork, D. G. (1992). Second order derivatives for network pruning: Optimal Brain Surgeon. *NeurIPS*.
3. Frantar, E., & Alistarh, D. (2023). SparseGPT: Massive Language Models Can be Accurately Pruned in One Shot. *ICML 2023*.
4. Hinton, G., Vinyals, O., & Dean, J. (2015). Distilling the Knowledge in a Neural Network. *NIPS Workshop on Deep Learning*.
5. Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). DistilBERT, a distilled version of BERT. *arXiv:1910.01108*.
6. Frantar, E., Ashkboos, S., Hoefler, T., & Alistarh, D. (2022). GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers. *ICLR 2023*.
7. Dettmers, T., Pagnoni, A., Holtzman, A., & Zettlemoyer, L. (2023). QLoRA: Efficient Finetuning of Quantized LLMs. *NeurIPS 2023*.
8. Gerganov, G. et al. (2023). llama.cpp: Inference of LLaMA model in pure C/C++. GitHub repository.
9. Qwen Team (2024). Qwen2 Technical Report. *arXiv:2407.10671*.
10. Merity, S., Xiong, C., Bradbury, J., & Socher, R. (2017). Pointer Sentinel Mixture Models. *ICLR 2017*.
11. Chu, X. et al. (2023). MobileVLM: A Fast, Reproducible and Strong Vision Language Assistant for Mobile Devices. *arXiv:2312.16886*.
