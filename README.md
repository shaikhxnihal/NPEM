# 🧠 NPEM — NeuroPlastic Echo Modulation

A research prototype for low-overhead continual learning using spectral activation memory and adaptive plasticity gating.

---

## 🚀 Overview

NPEM (NeuroPlastic Echo Modulation) is an experimental continual-learning framework designed to reduce catastrophic forgetting in neural networks without relying on replay buffers or large memory storage.

The project explores a biologically inspired idea:

> Neural activation patterns may contain stable frequency-domain signatures that can be used as lightweight memory traces during continual learning.

Instead of storing previous training samples, NPEM captures compressed spectral "echo signatures" from neuron activations and uses them to modulate future learning updates.

This repository is currently a **research prototype** focused on:

* continual learning experiments,
* activation-spectrum analysis,
* adaptive gradient gating,
* low-memory retention strategies.

---

# ⚠️ Current Project Status

This repository is an early-stage experimental implementation and should be considered:

* a proof-of-concept,
* a research exploration,
* not a production-ready framework.

Several components are simplified for experimentation and visualization purposes.

Current limitations include:

* simplified benchmark setup,
* partial continual-learning pipeline,
* incomplete replay/healing mechanism,
* limited benchmarking,
* no formal theoretical guarantees yet.

---

# 🧩 Core Ideas

## 1. Echo Signatures

Neuron activations are aggregated and transformed into the frequency domain using FFT.

The dominant spectral peaks are stored as compact activation fingerprints:

```python
freq = fft.rfft(hist)
mag = freq.abs()
top_idx = mag.topk(k).indices
```

These signatures act as lightweight memory traces representing previously learned behavior.

---

## 2. Adaptive Plasticity Gating

During future learning tasks, current activation signatures are compared against stored echoes.

If overlap is high:

* gradients are reduced,
* sensitive parameters become more stable.

If overlap is low:

* learning remains flexible.

This creates a balance between:

* stability (retaining old knowledge),
* plasticity (learning new knowledge).

---

## 3. Low-Memory Continual Learning

Unlike replay-based methods, NPEM does not store previous datasets.

Instead, it stores only compact spectral summaries of activations.

The goal is to explore:

* lightweight retention,
* edge-compatible continual learning,
* replay-free adaptation.

---

# 🔬 Current Architecture

Pipeline:

1. Forward activations are collected
2. Activation histories are transformed via FFT
3. Spectral peaks are stored in an echo bank
4. Echo overlap estimates parameter stability
5. Gradients are selectively modulated

---

# 🧪 Experimental Setup

Current prototype uses:

* PyTorch
* MLP architecture
* MNIST-based continual learning experiments
* simplified Permuted-MNIST style tasks

---

# 📈 Research Goals

This project investigates whether frequency-domain activation summaries can:

* reduce catastrophic forgetting,
* provide compact memory representations,
* support task-agnostic continual learning,
* improve edge-device compatibility.

---

# 🛠 Current Features

✅ FFT-based activation signature extraction
✅ Echo memory bank
✅ Adaptive gradient modulation
✅ Lightweight continual-learning prototype
✅ Interactive Flask demo dashboard

---

# ⚠️ Not Yet Implemented / In Progress

❌ Full continual-stream learning
❌ Formal theoretical analysis
❌ Real edge-device optimization
❌ Complete replay/healing system
❌ Large-scale benchmark validation
❌ Production deployment pipeline

---

# 📊 Planned Benchmarks

Future evaluation targets:

* Permuted MNIST
* Split CIFAR-100
* TinyImageNet
* CORe50
* robotic continual-learning tasks

Baseline comparisons planned against:

* EWC
* GEM
* Replay methods
* LwF
* Synaptic Intelligence (SI)

---

# 🧠 Inspiration

NPEM is inspired by:

* neuroplasticity,
* synaptic consolidation,
* memory stabilization,
* spectral signal analysis.

The project explores whether neural activation frequencies can act as compressed memory carriers during continual learning.

---

# 🖥️ Running the Demo

## Install dependencies

```bash
pip install torch torchvision flask matplotlib numpy
```

## Run the application

```bash
python app.py
```

Open:

```text
http://localhost:5000
```

---

# 📂 Project Structure

```text
.
├── app.py
├── templates/
├── static/
├── data/
└── README.md
```

---

# ⚠️ Research Disclaimer

This repository is experimental research software.

Results shown in the demo should not yet be interpreted as state-of-the-art continual learning performance.

The implementation is intended for:

* experimentation,
* research discussion,
* prototype development,
* educational exploration.

Further validation, benchmarking, and theoretical analysis are required.

---

# 🤝 Contributions

Contributions, discussions, and research feedback are welcome.

Areas of interest:

* continual learning,
* memory-efficient AI,
* edge AI,
* spectral neural analysis,
* adaptive plasticity methods.

---

# 📜 License

MIT License

---

# 📬 Contact

Created by Nihal Shaikh

GitHub:
https://github.com/shaikhxnihal/NPEM
