# EdgeLLM — Model Compression & Edge Deployment Toolkit

**Course Link:** https://www.samy101.com/edge-ai-26/

This repository demonstrates a full compression and deployment pipeline that makes large language models practical on commodity edge devices (for example, Raspberry Pi 5). The pipeline compresses Qwen2-0.5B using layer-aware structured sparsity, knowledge distillation, and SG-GPTQ 4-bit quantisation to produce a small, fast model suitable for on-device inference.

## Abstract

LLMs exhibit impressive functionality but are impractical for deployment outside cloud environments owing to massive memory and computational overhead. This project implements EdgeLLM, a full-fledged model compression framework to make LLMs practical on commodity edge devices. The pipeline used for Qwen2-0.5B comprises three steps:

1. SparseGPT with Layer-Aware Sparsity Sensitivity (LASS): an efficient sparse pruning algorithm that assigns sparsity budgets to individual layers according to their output tolerance to sparsity.
2. Knowledge Distillation (KD): mitigates accuracy loss resulting from sparse pruning.
3. SG-GPTQ with NF4 4-bit quantisation: an effective compression technique that scales the model size down by ~4.4× (942 MB → 215 MB).

The final compressed model achieves a WikiText-2 perplexity of 21.58 (≈ +3.49 from the baseline), while being small and fast enough to run on edge hardware.

## Repository Layout

- `notebooks/` — Jupyter notebooks for experiments, training, quantisation, and evaluation.
- `third_party/llama.cpp/` — vendor build and helper scripts used for deploying quantised/gguf models with `llama.cpp` tooling.
- `README.md` — this file.

## Key Features

- Reproducible three-step compression pipeline: sparsity-aware pruning (LASS), knowledge distillation, and SG-GPTQ NF4 quantisation.
- Working artifacts: compressed GGUF models ready for runtime on CPU-bound devices.
- Notebook-driven experiments for conversion, evaluation, and lightweight inference.

## Quick Start

### Run Compression & Evaluation on Kaggle

For best results, download and run the notebooks from Kaggle where free GPU compute is available:

1. Download the notebooks:
   - `notebooks/qwen.ipynb` - Qwen2-0.5b compression
   - `notebooks/tinyllama.ipynb` - TinyLlama1.1b compression
   - `notebooks/opt1-3.ipynb` - OPT1-3 compression

2. Upload to Kaggle Notebook environment.
3. Follow notebook cells in order:
   - Data preparation & pruning
   - Knowledge distillation
   - SG-GPTQ quantisation
   - Model export to `.gguf` and evaluation

4. Download the resulting `.gguf` models and test results on to your local machine(CPU machine).



## Notebooks

Open the notebooks in `notebooks/` to reproduce experiments and analyses:

- `qwen.ipynb` — compression steps, KD, and quantisation applied to Qwen2-0.5B.
- `tinyllama.ipynb` — smaller model experiments and speed/size trade-offs.
- `opt1-3.ipynb` — additional experiments, baselines, and comparisons.

Each notebook contains detailed, runnable steps for data preparation, pruning, distillation, quantisation, and evaluation. They also show how to export final models to the `gguf` format for `llama.cpp` consumption.

## Model Conversion & Compression Pipeline (High-level)

1. Sparse Pruning (SparseGPT + LASS)
	- Analyze layer sensitivity and assign per-layer sparsity budgets.
	- Run SparseGPT-style pruning to induce sparsity while preserving important weights.

2. Knowledge Distillation
	- Use a teacher–student setup to fine-tune the pruned model and recover accuracy.
	- Distillation scripts and parameter choices are documented in the notebooks.

3. SG-GPTQ (NF4) Quantisation
	- Convert the distilled/pruned model to 4-bit NF4 representation using SG-GPTQ.
	- Export the quantised model to `.gguf` for use with `llama.cpp` and other runtime backends.

## Evaluation

- The final Qwen2-0.5B compressed model achieves WikiText-2 perplexity ≈ 21.58 (baseline +3.49).
- Notebooks provide the evaluation scripts and the exact commands used to compute perplexity and other metrics.

## Reproducing Experiments on Kaggle

1. Open each notebook in the Kaggle environment (GPU enabled).
2. Follow notebook cells in order:
   - **Data preparation**: Load WikiText-2 or custom dataset
   - **SparseGPT + LASS**: Run layer-aware sparsity analysis and pruning
   - **Knowledge Distillation**: Fine-tune the pruned model with teacher guidance
   - **SG-GPTQ Quantisation**: Convert to 4-bit NF4 and export to `.gguf`
   - **Evaluation**: Compute perplexity and other metrics

3. Download resulting models and results to your `models/` directory.

## Raspberry Pi Setup & Inference Deployment

This section covers deploying the quantised GGUF models on Raspberry Pi 5 for edge inference.

### Pre-Deployment Checklist (Laptop & Pi)

Before attempting SSH connection or deployment:

- **Laptop connected to hotspot/WiFi**
- **Raspberry Pi connected to same hotspot/WiFi**
- **Hotspot band set to 2.4 GHz** (recommended for better range)
- **Raspberry Pi IP obtained** (run `hostname -I` on Pi)
- **SSH enabled on Raspberry Pi**

### Step 1: Obtain Raspberry Pi Username and IP Address

On the Raspberry Pi, open a terminal and run the following commands **one at a time**:

1. Get the Raspberry Pi username:
   ```bash
   whoami
   ```
   Example output:
   ```
   rpi10
   ```

2. Get the Raspberry Pi IP address:
   ```bash
   hostname -I
   ```
   Example output:
   ```
   192.168.1.25
   ```

**Note the username and IP address for later steps.**

### Step 2: Connect to Raspberry Pi via SSH from VS Code (Laptop)

1. On your laptop, open **VS Code**.
2. Press **Ctrl + Shift + P** (or **Cmd + Shift + P** on macOS).
3. Search for and select: **Remote-SSH: Connect to Host**
4. Select: **Add New SSH Host**
5. Enter the SSH command:
   ```
   ssh <username>@<ip_address>
   ```
   Example:
   ```
   ssh rpi10@192.168.1.25
   ```
6. Select a SSH config file location when prompted.
7. Choose **Linux** as the remote platform.
8. Enter the Raspberry Pi password when prompted.

After successful connection, VS Code will open the Raspberry Pi folder remotely.

### Step 3: Prepare Raspberry Pi (First-Time Setup)

SSH into your Raspberry Pi and run the setup commands:

```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install required build tools
sudo apt install -y git cmake build-essential

# Optional but recommended for monitoring
sudo apt install -y htop
```

### Step 4: Clone llama.cpp on Raspberry Pi

```bash
cd ~
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
```

### Step 5: Build llama.cpp (Optimized for Raspberry Pi 5)

```bash
cmake -B build
cmake --build build -j$(nproc)
```

This creates the `llama-server` binary in `build/bin/`.

### Step 6: Transfer Quantised Model to Raspberry Pi

On your **laptop**, transfer the `.gguf` model to the Pi:

```bash
scp models/tinyllama.gguf pi@<your_pi_ip>:~/models/
```

Or for Qwen2.5:

```bash
scp models/qwen2.5.gguf pi@<your_pi_ip>:~/models/
```

On the **Raspberry Pi**, ensure the models directory exists:

```bash
mkdir -p ~/models
ls ~/models/  # Verify the .gguf file is present
```

### Step 7: Create Inference Script on Raspberry Pi

Create a script to run the inference server. On the Raspberry Pi:

```bash
nano ~/smoke_test.sh
```

Paste the following content:

```bash
#!/bin/bash

MODEL=~/models/tinyllama-q8_0.gguf
SERVER=~/llama.cpp/build/bin/llama-server

echo "Starting LLM server on Raspberry Pi..."

$SERVER \
  -m $MODEL \
  -c 2048 \
  -t $(nproc) \
  --host 0.0.0.0 \
  --port 8080
```

Make it executable:

```bash
chmod +x ~/smoke_test.sh
```

### Step 8: Run Inference Server on Raspberry Pi

```bash
~/smoke_test.sh
```

The server will start and listen on `http://<pi-ip>:8080/`.

### Step 9: Test Inference from Your Serving API

From your laptop or another machine, query the server:

```bash
curl -X POST http://<pi_ip>:8080/completion \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Hello, world!",
    "n_predict": 50
  }'
```

Replace `<pi_ip>` with your Raspberry Pi IP address (e.g., `192.168.1.25`).

### Performance Notes

- **Quantised model size**: ~215 MB (TinyLLaMA/Qwen2.5)
- **Memory usage on Pi**: ~500–800 MB for inference
- **Inference speed**: 1–3 tokens/second (depends on model size and `nproc` threads)
- **Optimal settings**: Use `-t $(nproc)` to utilize all CPU threads on Pi 5

## Third-Party Components

- `third_party/llama.cpp/` — used for fast CPU inference of GGUF models; follow its `README` for building and platform-specific notes.
- Any other third-party tools used for conversion/quantisation are documented inline in the notebooks.

## Troubleshooting

- If inference fails with memory errors, ensure you are using the quantised `.gguf` model, not the original checkpoint.
- For build issues with `llama.cpp`, consult `third_party/llama.cpp/README.md` and the project's issue tracker.
- If tokenizer mismatches occur, confirm that the tokenizer used in conversion matches the runtime tokenizer.

## Contributing

Contributions are welcome — open an issue or a pull request for fixes, improvements, or additional conversion scripts. When contributing:

- Describe the OS and environment used for testing.
- Include commands and minimal reproduction steps.

## Citation

If you use this work in research, please cite the underlying ideas and the compression pipeline concepts implemented here. Example citation (adapt to your citation format):

> EdgeLLM: Layer-aware sparsity + KD + SG-GPTQ for edge deployment. Perplexity improvements and compression details are provided in the project documentation and notebooks.

## License

Check repository root for a license file. Third-party components (e.g., `llama.cpp`) include their own licenses and must be followed.

## Contact

For questions, open an issue or contact the maintainers via the repository issue tracker.

---

For step-by-step reproduction and parametrization, see the notebooks in `notebooks/` and create your own `models/` directory.
