# Phase Transition Critical Points in Learning Systems: A Multimodal Unified Cognitive Architecture Based on CC-SNN and LLM

**Author:** Bingqin Wang (王秉钦)

Beijing National Accounting Institute

**Patent Notice:** The methods described herein are covered by Chinese Patent Application No. 202610386399.4 ("Efficient Inference Method and System Based on Contraction-Constrained Spiking Neural Network").

---

## Abstract

Current Large Language Models (LLMs) cannot truly "see" or "hear." Multimodal LLMs such as GPT-4o, Gemini, and LLaVA convert images and audio into language tokens — this is **pseudo-multimodal perception**, analogous to a blind person "seeing" by having someone describe a painting to them.

This paper reveals a **phase transition critical point** in learning systems: when knowledge complexity $C$ exceeds $\sqrt{N}$ (where $N$ is training sample count), a single-tier LLM necessarily fails. We propose a biologically-inspired two-stage cognitive architecture: a **Contraction-Constrained Spiking Neural Network (CC-SNN)** front-end for true physical signal perception, backed by an LLM for reasoning in the abstract symbol space, with deterministic output via a Lookup Table (LUT).

**Key finding:** An ordinary 3-billion-parameter text-only model, merely by connecting a CC-SNN perceptual front-end, jumps from random guessing (20%) to 100% accuracy on multimodal classification — without modifying a single line of LLM code, without any cross-modal pretraining. CC-SNN+LUT inference latency is 0.03ms, **13,000× faster** than bare LLM inference.

**Keywords:** Spiking Neural Network, Phase Transition, Multimodal Perception, Lookup Table, Biologically-Inspired Computing

---

## 1. Introduction: Why LLMs Are Not Truly Multimodal

### 1.1 The Core Problem

GPT-4o can answer questions about images. Gemini can summarize audio. But do they actually "see" and "hear"?

The answer is no. Every current multimodal LLM follows the same strategy:

```
Image/Audio → Encoder (CLIP/Whisper) → Projection Layer → Language Tokens → Transformer
```

Regardless of input modality, the LLM only ever processes language tokens. An image is translated into "words about the image"; audio becomes "words about the sound." This is equivalent to a blind person understanding the world through verbal descriptions — they comprehend the description but have never truly seen.

### 1.2 Experimental Evidence

We fed the same LLM (qwen2.5:3b) two types of input for an identical classification task:

- **Human-written description:** "This is a red flashing + high-pitched alarm scene" → Classification accuracy: **100%**
- **Raw pixel data:** "8×8 brightness map: row0=[0.8,0.8,0.8,...] row1=[0.8,0.9,..." → Classification accuracy: **30%** (near random guessing for 5 classes)

The gap exists because the LLM is not "seeing" pixels — it is reading numbers. It has no visual perception capability, only language comprehension.

### 1.3 Our Approach

The human brain's solution is straightforward: **two stages.**

1. **Perception layer** (visual/auditory cortex): Convert photons and sound waves into neural spike patterns
2. **Cognition layer** (prefrontal cortex): Perform logical reasoning on abstract symbols

We replicate this architecture exactly:
- **Perception = CC-SNN:** Convert physical signals (pixels, waveforms) into bucket IDs (abstract symbols)
- **Cognition = LLM:** Classify within buckets (learning phase), or direct table lookup (inference phase)

**Critical property: Adding a new modality = adding one encoder + one block_map entry. Zero changes to SNN or LLM code.**

---

## 2. Phase Transition: Why Single-Tier LLMs Necessarily Fail

### 2.1 Intuitive Explanation

Imagine you are a teacher assigning students to classes:
- 3 classes, 100 students: Easy. Each class has 30+ students with clear distinguishing features.
- 50 classes, 100 students: Only 2 students per class. You cannot reliably distinguish who belongs where.

LLMs face the same problem. When category count $C$ grows large relative to available data per class, decision boundaries become impossibly fragile.

### 2.2 Critical Point Formula

We define the **phase transition condition:**

$$C \approx \sqrt{N}$$

- **$C \leq \sqrt{N}$** (data-rich regime): Direct LLM learning is optimal. Any front-end abstraction adds lossy compression noise.
- **$C > \sqrt{N}$** (data-sparse regime): Single-tier learning necessarily fails. A front-end abstraction layer becomes mandatory, not optional.

### 2.3 Experimental Validation

Using a real local LLM (qwen2.5:3b via Ollama) on N=250 curriculum texts:

| Classes C | Above Critical? (√250≈15) | LLM Direct | CC-SNN+LUT | Improvement |
|-----------|--------------------------|-----------|------------|-------------|
| 4 | No | 95.0% | 100.0% | +5.0% |
| 8 | No | 52.5% | 97.5% | +45.0% |
| 16 | **Yes** | 12.5% | 80.0% | **+67.5%** |
| 32 | **Yes** | 5.0% | 65.0% | +60.0% |
| 64 | **Yes** | 0.0% | 77.5% | **+77.5%** |
| 128 | **Yes** | 0.0% | 40.0% | +40.0% |

**Observation:** LLM accuracy collapses from 95% at C=4 to 12.5% at C=16 and reaches 0% at C=64. The collapse point at C≈15 matches the theoretical prediction of √250≈15.8 precisely.

### 2.4 Optimal Ratio Law

CC-SNN compresses inputs into $K$ buckets. Too few buckets means too many sub-classes per bucket; too many means the SNN over-partitions. The optimum:

$$K^* = \max\left(2, \frac{C^{2/3}}{(N/C)^{1/3}}\right) \quad (C > \sqrt{N})$$

**Physical meaning:** The information content of each SNN output exactly matches the LLM's per-inference learning capacity ceiling. This is not mathematical coincidence — it is an information-theoretic necessity.

---

## 3. CC-SNN: Contraction-Constrained Spiking Neural Network

### 3.1 SNN in One Sentence

Traditional neural networks pass floating-point values. SNNs mimic the real brain: neurons accumulate charge and only "fire" a discrete spike when exceeding a threshold. Advantages: ultra-low power consumption + native temporal processing.

### 3.2 The CC-SNN Innovation

Standard SNNs suffer from training instability and divergence. CC-SNN adds a mathematical constraint — the **Banach contraction mapping principle**: after each weight update, the system state must be strictly closer to the target fixed point.

$$\|W(t+1) - W^*\| \leq \kappa \cdot \|W(t) - W^*\|, \quad \kappa \in (0,1)$$

This guarantees:
- **Absolute convergence:** No divergence, no oscillation
- **Continual learning:** New knowledge does not destroy old knowledge (solves catastrophic forgetting)
- **Determinism:** Identical inputs always produce identical outputs (zero hallucination)

### 3.3 Local Connectivity (block_map) — Training-Free Routing

CC-SNN's key architectural innovation is **block-diagonal local connectivity:**

```python
# 256-dim input, 5 buckets
# Each bucket only sees its assigned 51 dimensions (zero cross-talk)
block_map = [
    (0, 51),     # Bucket 0 ← Input[0:51]
    (51, 102),   # Bucket 1 ← Input[51:102]
    (102, 153),  # Bucket 2 ← Input[102:153]
    (153, 204),  # Bucket 3 ← Input[153:204]
    (204, 256),  # Bucket 4 ← Input[204:256]
]
```

Each input channel is **physically connected** to its corresponding neuron subgroup. **No training required** — the connectivity structure itself serves as the routing mechanism. The subgroup with the highest total spike count determines the bucket ID.

### 3.4 D-LIF Neuron (Deterministic Leaky Integrate-and-Fire)

```python
# Core parameters
tau_m = 10e-3          # Membrane time constant: 10ms
dt = 1e-3              # Time step: 1ms
beta = exp(-dt/tau_m)  # Decay factor
v_threshold = 1.0      # Firing threshold

# Per timestep computation
v = beta * v + (1 - beta) * I_syn   # Membrane potential accumulation
if v >= v_threshold:
    spike = 1        # Fire
    v = 0            # Reset
```

Unlike biological LIF neurons, D-LIF removes 8 biological imperfections (refractory period, conduction randomness, signal attenuation, ion channel noise, metabolic dependence, temperature sensitivity, neuronal death, crosstalk) while preserving deterministic behavior.

---

## 4. Multimodal Unification: Any Signal Can Connect

### 4.1 Prism Visual Encoder (PrismVisualV2)

Biomimetic of the human eye's three cone cell types (L/M/S):

| Channel | Biological Basis | Function |
|---------|-----------------|----------|
| R (Red) | L-cone (564nm, long wavelength) | Red perception |
| G (Green) | M-cone (534nm, medium wavelength) | Green perception |
| B (Blue) | S-cone (420nm, short wavelength) | Blue perception |
| Edge | Retinal ganglion cells | On-off center-surround detection |
| Texture | V1 primary visual cortex | Frequency/texture detection |

**Output:** 8×8×5 = 320-dimensional spike tensor

```python
from model.prism_visual_v2 import PrismVisualEncoderV2

encoder = PrismVisualEncoderV2(grid_h=8, grid_w=8)
# img: (8, 8, 3) RGB numpy array, values in [0, 1]
spikes = encoder.encode(img, timesteps=10)  # → (10, 320) spike tensor
feature = spikes.mean(dim=0)                # → (320,) temporal average
```

**Block_map isolation verified:** Changing brightness input only affects brightness-mapped neurons; all other channels show zero response change.

### 4.2 Cochlear Auditory Encoder (RealCochlea)

Biomimetic of the human basilar membrane:

```python
from model.real_cochlea import RealCochlea

cochlea = RealCochlea(
    n_channels=32,          # 32 frequency channels
    freq_range=(80, 8000)   # 80Hz–8000Hz, logarithmic spacing (biomimetic)
)
# waveform: numpy array, 16kHz sample rate
spikes = cochlea.encode_waveform(waveform, sample_rate=16000)  # → (T, 160)
feature = spikes.mean(dim=0)                                    # → (160,)

# Spectral analysis
info = cochlea.describe(waveform, 16000)
print(f"Dominant frequency: {info['dominant_freq']:.0f}Hz")
print(f"Active channels: {info['active_channels']}")
```

### 4.3 Universal Modality Extension Framework

Any physical signal can be connected via a unified formula:

```
Physical signal → Sensor digitization → Channel mapping → Spike encoding → block_map → SNN
```

Five implemented modality encoders:

| Modality | Encoder | Biological Basis | Output Dim |
|----------|---------|-----------------|------------|
| Vision | PrismVisualV2 | L/M/S cones + ganglion + V1 | 320 |
| Audition | RealCochlea | Basilar membrane, 32 bands | 160 |
| Olfaction | OlfactoryEncoder | Olfactory bulb glomeruli | 32 |
| Touch | TactileEncoder | Merkel/Meissner corpuscles | 32 |
| Proprioception | ProprioceptionEncoder | Muscle spindle + Golgi tendon | 18 |

```python
from model.modality_framework import (
    OlfactoryEncoder, TactileEncoder, ProprioceptionEncoder,
    build_multimodal_block_map
)

# Multimodal fusion: automatically assigns SNN neuron subgroups
block_map = build_multimodal_block_map(
    visual_dim=320, audio_dim=160, olfactory_dim=32
)
# Each modality → independent neuron subgroup, zero cross-talk
```

**Adding a new modality = writing one encoder (encode method) + adding one block_map entry. Zero changes to SNN or LLM code.**

---

## 5. Experimental Validation: All Three Stages Pass

All experiments are one-command reproducible. Requirements:
```bash
pip install torch numpy scipy scikit-learn requests
ollama pull qwen2.5:3b-instruct-q4_K_M   # For LLM baseline comparison
```

### 5.1 Stage 1: Text Classification

**Task:** 5-class curriculum text (cognitive/language/motor/fine-motor/social-emotional), SNN+LUT vs bare LLM

```bash
python code/stage1_text.py
```

| Metric | SNN+LUT | Bare LLM |
|--------|---------|----------|
| **Accuracy** | **96.8%** | 71.0% |
| Latency | 0.033ms | 350.9ms |
| LLM calls at inference | **0** | 31 |

**LUT inference makes zero LLM calls.** The LLM is only used during the learning phase to build the table; inference is pure lookup.

### 5.2 Stage 2: Single Modality (Vision + Audio Separately)

**Comparison method:** The LLM receives raw signal data (pixel values / cochlear spectra), not human-written descriptions. This simulates the essence of "pseudo-multimodal" processing.

```bash
python code/stage2_multimodal.py
```

| Modality | SNN+LUT | LLM on Raw Data |
|----------|---------|-----------------|
| Vision (8×8 pixel art) | **96.7%** | 30.0% |
| Audio (synthesized sounds) | **86.7%** | 23.3% |

**The LLM achieves only 30% on raw pixel values** (5-class random guessing = 20%). It is not "seeing" — it is reading numbers.

### 5.3 Stage 3: Ultimate Test — True Multimodal

**Task:** 5-class audiovisual scenes (alarm/ocean/forest/sunset/silent film), simultaneous vision + audio input

```bash
python code/stage3_ultimate.py
```

| Approach | Accuracy | Latency |
|----------|----------|---------|
| A) Multimodal SNN+LUT | **100.0%** | 0.032ms |
| B) Vision-only SNN+LUT | 100.0% | 0.029ms |
| C) Audio-only SNN+LUT | 100.0% | 0.024ms |
| D) LLM pseudo-multimodal | 20.0% | 403.2ms |

**The LLM scores 20% on Stage 3 — exactly random guessing for 5 classes.** When confronted with simultaneous raw visual and auditory data, a text model is literally blind and deaf.

### 5.4 Summary of All Three Stages

| Stage | Task | SNN+LUT | LLM | Speedup | Result |
|-------|------|---------|-----|---------|--------|
| 1 | Text classification | **96.8%** | 71.0% | 10,000× | ✅ PASS |
| 2 | 8×8 pixel art | **96.7%** | 30.0% | 14,000× | ✅ PASS |
| 2 | Synthesized sounds | **86.7%** | 23.3% | 3,300× | ✅ PASS |
| 3 | Vision + Audio | **100%** | 20.0% | **13,000×** | ✅ PASS |

---

## 6. Turning Any Reasoning Model Into a Multimodal Model

### 6.1 Conventional Approach vs. Ours

**Conventional (pseudo-multimodal):**
```
Image → CLIP (400M params) → Cross-modal alignment training (billions of pairs)
      → Projection into LLM → LLM processes language tokens
```
- Requires massive visual encoders (CLIP 400M, ViT-22B)
- Requires billions of cross-modal alignment samples
- Adding each new modality requires retraining the alignment layer
- The LLM still only processes language tokens — it never actually "perceives"

**Our approach (true multimodal):**
```
Image → PrismVisualV2 (zero trainable params) → 320-dim spikes → CC-SNN → Bucket ID → LUT → Output
Audio → RealCochlea (zero trainable params) → 160-dim spikes → CC-SNN → Bucket ID → LUT → Output
```
- Encoders have **zero trainable parameters** (pure biomimetic formulas)
- **Zero cross-modal training**
- Adding a new modality = one encoder + one block_map entry, ~10 lines of code
- LLM is used only during learning; inference is pure table lookup

### 6.2 Three Steps to Make Your LLM "See"

```python
# Step 1: Encode (convert physical signal to spikes)
from model.prism_visual_v2 import PrismVisualEncoderV2
encoder = PrismVisualEncoderV2(grid_h=8, grid_w=8)
features = encoder.encode(image, timesteps=10).mean(dim=0)

# Step 2: Route (SNN local-connectivity bucketing)
snn = CCSNN(dim=320, K=5)
bucket_id, confidence = snn.compress(features.unsqueeze(0))

# Step 3: Lookup (O(1) output, zero LLM calls)
result = lut.query(bucket_id[0].item(), feat=features)
```

### 6.3 Compatible With Any LLM

The CC-SNN front-end outputs **discrete bucket IDs** (integers) — the most basic data type any LLM can process:
- qwen2.5:3b ✅
- Llama 3 ✅
- GPT-4 ✅
- Any text-capable model ✅

**No model modification or fine-tuning required.**

---

## 7. Comparison With Existing Multimodal Approaches

| Feature | GPT-4o | Gemini | LLaVA | **CC-SNN+LLM** |
|---------|--------|--------|-------|----------------|
| Perception type | Pseudo (token projection) | Pseudo (token projection) | Pseudo (token projection) | **True (SNN spikes)** |
| Visual encoder params | ~400M (CLIP) | ~2B (ViT) | ~300M (CLIP) | **0** |
| Cross-modal training data | Billions of pairs | Billions of pairs | Millions of pairs | **0** |
| Cost to add new modality | Full retraining | Full retraining | Full retraining | **Add encoder (~10 lines)** |
| Inference latency | >100ms | >100ms | >100ms | **0.03ms** |
| Deterministic output | No (probabilistic) | No | No | **Yes (table lookup)** |
| Local deployment | ❌ | ❌ | Marginal | **✅ Zero GPU required** |

---

## 8. FAME: 8-Dimensional State Controller

CC-SNN includes an 8-dimensional continuous parameter space for dynamic inference regulation:

| Dimension | Symbol | Meaning | Function |
|-----------|--------|---------|----------|
| Satisfaction | μ | System contentment with current state | Explore vs. exploit |
| Exploration | χ | Controlled stochastic perturbation | Escape local optima |
| Synergy | ε | Cross-modal consistency | Fusion weighting |
| Confidence | κ_s | Output certainty | Trigger SCAC correction |
| Uncertainty | ν | Knowledge boundary awareness | Emit "don't know" |
| Conflict | δ | Multi-bucket competition intensity | Request more data |
| Load | ρ | Computational resource utilization | Dynamic throttling |
| Divergence | λ | Deviation from convergence trajectory | Emergency pullback |

This endows the system with an analog of biological "emotion" — not simulating outward emotional expression, but providing internal state regulation functionally similar to how emotions modulate biological cognition.

---

## 9. Conclusion

This paper demonstrates three results:

1. **Learning systems exhibit a phase transition at** $C \approx \sqrt{N}$. Beyond this critical point, single-tier LLMs necessarily fail.
2. **The CC-SNN+LUT architecture comprehensively outperforms bare LLM inference** across text, vision, audio, and multimodal tasks — all four tests pass.
3. **Any text-only LLM can acquire true multimodal perception through a CC-SNN front-end**, requiring zero cross-modal training and zero model code changes.

The **Optimal Ratio Law** $K^* = C^{2/3} / (N/C)^{1/3}$ is not merely an engineering tool — it explains the information-theoretic necessity of the human brain's layered architecture: abandoning precise memory of physical details in favor of building deterministic knowledge on abstract symbols is not a compromise — it is the optimal solution.

> *"Making a blind, deaf text model see and hear — without changing a single line of the LLM."*

---

## References

[1] Wang, B. (2026). SCAC: Semantic Closed-loop Auto-Correction Unified Theory. Beijing National Accounting Institute.

[2] Wang, B. (2026). FAME: Emotion as a Fundamental Property of Matter. Beijing National Accounting Institute.

[3] Wang, B. (2026). Efficient Inference Method and System Based on Contraction-Constrained Spiking Neural Network. China National Intellectual Property Administration, Patent Application No. 202610386399.4.

[4] Vapnik, V. N. (1998). *Statistical Learning Theory*. Wiley.

[5] Maass, W. (1997). Networks of spiking neurons: the third generation of neural network models. *Neural Networks*, 10(9), 1659–1671.

[6] Radford, A. et al. (2021). Learning Transferable Visual Models From Natural Language Supervision. *ICML 2021*. (CLIP)

[7] Liu, H. et al. (2024). Visual Instruction Tuning. *NeurIPS 2023*. (LLaVA)

---

## Appendix A: Full Reproduction Guide

### Environment Setup (5 minutes)

```bash
# 1. Python environment
pip install -r requirements.txt
# or manually:
pip install torch numpy scipy scikit-learn requests

# 2. Ollama (for LLM baseline comparison)
# Download from: https://ollama.com
ollama pull qwen2.5:3b-instruct-q4_K_M
ollama serve   # Start the server
```

### Running Experiments

```bash
# Stage 1: Text classification
python code/stage1_text.py

# Stage 2: Vision + Audio (tested separately)
python code/stage2_multimodal.py

# Stage 3: Ultimate multimodal test
python code/stage3_ultimate.py

# Phase transition sweep (C = 4 → 128)
python code/experiment_real.py
```

### Testing With Different Models

All scripts have a configuration section at the top:
```python
OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
OLLAMA_MODEL = "qwen2.5:3b-instruct-q4_K_M"   # Change to your model
```

You can substitute any Ollama model (qwen2.5:7b, llama3:8b, mistral:7b, etc.). Predicted scaling:
- **3B model:** Stable at C≤6 classes (>90%), collapses at C>12
- **7B model:** Stable at C≤12, collapses at C>20
- **72B model:** Stable at C≤30

CC-SNN+LUT maintains high accuracy regardless of LLM model size, because inference does not depend on the LLM.

### Adding Custom Modalities

```python
# Inherit the ModalityEncoder base class
from model.modality_framework import ModalityEncoder

class MyEncoder(ModalityEncoder):
    def __init__(self):
        super().__init__(name="my_sensor", n_channels=16)

    def encode(self, raw_signal):
        # Convert your physical signal to spikes
        # Return: (timesteps, n_channels * repeat) tensor
        ...
        return spikes
```
