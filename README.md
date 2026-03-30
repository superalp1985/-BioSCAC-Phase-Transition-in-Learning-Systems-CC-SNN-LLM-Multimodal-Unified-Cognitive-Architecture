# BioSCAC: Phase Transition in Learning Systems
# CC-SNN + LLM Multimodal Unified Cognitive Architecture

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
[![Patent](https://img.shields.io/badge/Patent-202610386399.4-blue.svg)]()

> **学习系统的相变临界点：基于 CC-SNN 与 LLM 的多模态统一认知架构**
>
> *Phase Transition Critical Points in Learning Systems: A Multimodal Unified Cognitive Architecture Based on CC-SNN and LLM*

**Author:** Bingqin Wang (王秉钦), Beijing National Accounting Institute

**Patent:** 202610386399.4 — 一种基于收敛约束脉冲神经网络的高效推理方法及系统

---

## Core Claims (Experimentally Verified ✅)

### 1. Phase Transition Critical Point
When knowledge complexity $C > \sqrt{N}$, single-tier learning (LLM alone) **necessarily fails**.

| C | LLM Direct | CC-SNN+LUT | Improvement |
|---|-----------|------------|-------------|
| 4 | 95.0% | 100.0% | +5.0% |
| 8 | 52.5% | 97.5% | +45.0% |
| 16 | 12.5% | 80.0% | +67.5% |
| 64 | 0.0% | 77.5% | +77.5% |
| 128 | 0.0% | 40.0% | +40.0% |

### 2. SNN-LLM Optimal Ratio Law
$$K^* = \max\left(2, \frac{C^{2/3}}{(N/C)^{1/3}}\right) \quad (C > \sqrt{N})$$

### 3. True Multimodal via CC-SNN (Not Pseudo-Multimodal)

Any **text-only** LLM becomes truly multimodal through CC-SNN front-end:

| Stage | Task | CC-SNN+LUT | LLM (Pseudo-MM) | Speedup |
|-------|------|-----------|-----------------|---------|
| 1: Text | Curriculum classification | **96.8%** | 71.0% | 10,000× |
| 2: Vision | 8×8 pixel art | **96.7%** | 30.0% | 14,000× |
| 2: Audio | Synthesized sounds | **86.7%** | 23.3% | 3,300× |
| 3: AV | Vision + Audio joint | **100.0%** | 20.0% | **13,000×** |

**The LLM reads raw pixel/audio data at 20-30% (random guessing). CC-SNN+LUT achieves 86-100%. That's the difference between "describing signals as text" and "actually perceiving them."**

---

## Architecture

```
Physical Signal → Modality Encoder → CC-SNN (block-diagonal local connectivity)
                                         ↓
                                    Bucket ID → LUT → Output
                                         ↑
                              (Learning phase: LLM builds LUT)
                              (Inference phase: Zero LLM calls)
```

### Key Innovation: Local Connectivity (block_map)

Each modality connects to its own SNN neuron subgroup — no cross-talk:

- **Vision**: R/G/B/Edge/Texture → 5 independent neuron pools
- **Audio**: 32 cochlear channels → frequency-band pools
- **Olfactory/Tactile/Proprioception**: Same pattern

**Adding a new modality = adding an encoder + a block. SNN and LLM: zero code changes.**

---

## Modality Encoders (Biologically Inspired)

| Modality | Encoder | Bio Basis | Channels |
|----------|---------|-----------|----------|
| Vision | PrismVisualV2 | L/M/S cones + ganglion cells + V1 | 320-dim (8×8×5) |
| Auditory | RealCochlea | Basilar membrane, 80-8000Hz log | 160-dim (32ch×5) |
| Olfactory | OlfactoryEncoder | Olfactory bulb glomeruli | 32-dim (8 clusters) |
| Tactile | TactileEncoder | Merkel/Meissner (slow/fast adapt) | 32-dim (pressure+vibration) |
| Proprioception | ProprioceptionEncoder | Muscle spindle + Golgi tendon | 18-dim (angle+vel+torque) |

---

## Repository Structure

```
├── README.md                          # This file
├── LICENSE                            # CC BY-NC 4.0
├── paper/
│   └── paper.md                       # Full paper (Chinese + English abstract)
├── code/
│   ├── stage1_text.py                 # Stage 1: Text classification
│   ├── stage2_multimodal.py           # Stage 2: Single-modality (vision + audio)
│   ├── stage3_ultimate.py             # Stage 3: True multimodal (vision + audio joint)
│   └── experiment_real.py             # Phase transition sweep (C=4→128)
├── model/
│   ├── neurons.py                     # D-LIF neurons + SCAC controller + SCAC-STDP
│   ├── prism_visual_v2.py             # RGB visual encoder (local connectivity)
│   ├── real_cochlea.py                # 32-channel cochlear encoder
│   ├── modality_framework.py          # Universal modality framework
│   └── fame.py                        # 8-dim FAME emotion engine
├── results/
│   ├── stage1_results.json
│   ├── stage2_results.json
│   ├── stage3_results.json
│   └── real_validation.json           # Phase transition data
└── docs/
    └── figures/                       # Paper figures
```

---

## Quick Start

### Requirements
```bash
pip install torch numpy scipy scikit-learn requests
# For LLM baseline comparison:
# Ollama with qwen2.5:3b-instruct-q4_K_M (optional)
```

### Run All Three Stages
```bash
# Stage 1: Text (requires Ollama for LLM baseline)
python code/stage1_text.py

# Stage 2: Vision + Audio (requires Ollama)
python code/stage2_multimodal.py

# Stage 3: True Multimodal (requires Ollama)
python code/stage3_ultimate.py

# Phase transition sweep (requires Ollama)
python code/experiment_real.py
```

### Run Without Ollama (SNN+LUT only)
The CC-SNN+LUT pipeline runs standalone — no LLM needed for inference.
Ollama is only required for the LLM baseline comparison.

---

## Theoretical Foundation

This work builds on three prior theories by the author:

1. **SCAC** (Semantic Closed-loop Auto-Correction): Convergence guarantees via Banach contraction mapping
2. **FAME** (Emotion as Fundamental Property of Matter): 8-dimensional continuous emotion space for system state control
3. **OPT** (Onto-Plasticity Theory): Carbon-silicon isomorphism, stability-plasticity balance

---

## Citation

```bibtex
@article{wang2026biosac,
  title={Phase Transition Critical Points in Learning Systems: A Multimodal Unified Cognitive Architecture Based on CC-SNN and LLM},
  author={Wang, Bingqin},
  year={2026},
  institution={Beijing National Accounting Institute},
  note={Patent: 202610386399.4}
}
```

---

## License

This work is licensed under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).

- ✅ Academic research, education, personal use
- ✅ Attribution required
- ❌ Commercial use without separate license
- ❌ Patent claims (covered by 202610386399.4)

For commercial licensing inquiries, contact the author.

---

*"Making a blind, deaf text model see and hear — without changing a single line of the LLM."*
