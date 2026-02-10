# DPZO Scaling Law

## Setup

```bash
conda env create -n dpzero -f environments.yml
conda activate dpzero
```

## Usage

All commands should be run from within the `dp-aggzo/` or `dpzo_scaling_law/` directory.

### Single-GPU (Baseline, from [DP-AggZO](https://github.com/erguteb/dp-aggzo))

```bash
cd dp-aggzo

CUDA_VISIBLE_DEVICES=0 \
  MODEL=facebook/opt-125m TASK=SQuAD MODE=ft \
  LR=1e-5 EPS=1e-3 EVAL_STEPS=125 \
  DP_SAMPLE_RATE=0.032 DP_EPS=2.0 STEPS=500 \
  N=4 DP_CLIP=7.5 \
  bash examples/dpaggzo.sh
```

### Multi-GPU (DP-AggZO + distZO2 Parallel Perturbation)

```bash
cd dpzo_scaling_law

# 2 GPUs (1 pair)
CUDA_VISIBLE_DEVICES=0,1 \
  MODEL=facebook/opt-125m TASK=SQuAD MODE=ft \
  LR=1e-5 EPS=1e-3 EVAL_STEPS=125 \
  DP_SAMPLE_RATE=0.032 DP_EPS=2.0 STEPS=500 \
  N=4 DP_CLIP=7.5 NGPU=2 \
  bash examples/dpaggzo.sh
```

### Key Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `MODEL` | HuggingFace model name | `facebook/opt-125m` |
| `TASK` | Downstream task | `SQuAD` |
| `N` | Number of ZO perturbation directions | `4` |
| `NGPU` | Number of GPUs (must be even, default 1) | `2` |
| `DP_EPS` | Differential privacy epsilon | `2.0` |
| `DP_CLIP` | Gradient clipping threshold | `7.5` |
| `DP_SAMPLE_RATE` | Poisson sampling rate | `0.032` |
| `STEPS` | Total training steps | `500` |
| `LR` | Learning rate | `1e-5` |
| `EPS` | ZO perturbation epsilon | `1e-3` |

