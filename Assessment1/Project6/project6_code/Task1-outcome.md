# Task 1 Outcome — Baseline cVAE

**Source:** `Assessment1/Project6/project6_code/cVAE_DiscriminationLoss.ipynb`  
**Task sheet role:** Train a standard cVAE on `mnist_custom` with **7 latent dimensions**, until convergence / reasonable samples, as the **control baseline**.

---

## Verdict

**Task 1 is complete.**

The notebook implements and trains the baseline (control) cVAE end-to-end: dataset → architecture → training → sanity checks → saved artefacts. There is no discriminator work in this notebook yet (that is Task 2). Formal 12×10 comparison grids and quantitative hypothesis metrics are out of scope for Task 1 and belong to Task 3.

---

## Task-sheet checklist

| Requirement | Status | Evidence |
|-------------|--------|----------|
| Use supplied modified MNIST (`mnist_custom`) | Done | Loaded `mnist_custom-1.pt` |
| Standard cVAE from Tutorial 8.3 | Done | Conditional encoder/decoder + reparameterisation; recon + β·KL |
| Latent dimensions = **7** | Done | `Z_DIM = 7` |
| Train until convergence / reasonable samples | Done | 50 epochs; stable loss; recon + conditional samples look digit-like |
| Model usable as baseline for comparison | Done | Checkpoint + history saved |

---

## What was implemented

### Dataset

| Item | Value |
|------|--------|
| File | `mnist_custom-1.pt` (~107 MB) |
| Keys | `train_images`, `train_labels`, `test_images`, `test_labels` |
| Train / test | 60,000 / 10,000 |
| Image shape | `(1, 20, 20)` — **20×20**, not standard 28×28 MNIST |
| Pixel range | `[0, 1]`, `float32` |
| Classes | 0–9 (roughly balanced) |
| Batch size | 1000 train / 1000 test |

Architecture was adapted for **20×20** inputs (tutorial-style CNN strides fit this size).

### Model

- Conditional encoder / decoder / `cVAE` wrapper (Tutorial 8.3 style)
- Conditioning via **one-hot** class vectors
- Loss: `100 * MSE` reconstruction + `β * KL`, with **`β = 0.1`**
- **~350,107** trainable parameters
- Device: CUDA (`NVIDIA A16-4Q`) during the recorded run

### Training setup

| Hyperparameter | Value |
|----------------|--------|
| Epochs | 50 |
| Optimiser | AdamW (`lr=0.001`, `betas=(0.75, 0.99)`, `eps=1e-5`) |
| Seed | 42 |
| Train forward | Stochastic `z` via reparameterisation |
| Val / test forward | Deterministic decode from `μ` (tutorial-style) |

---

## Findings

### Training behaviour

Loss fell steadily and flattened by the end of training (reasonable convergence for this setup):

| Stage | Train total | Test total | Test recon | Test KL |
|-------|-------------|------------|------------|---------|
| Epoch 1 | 9.4802 | 5.9786 | 5.9188 | 0.5980 |
| Epoch 25 | 2.5367 | 2.2317 | 1.4988 | 7.3287 |
| Epoch 50 (final) | **2.3302** | **2.0212** | **1.2318** | **7.8942** |

Interpretation:

- **Reconstruction loss** dropped sharply early, then kept improving slowly → the model learned to reconstruct digits.
- **KL** rose over training (prior regularisation becoming active) while total loss still fell → expected VAE trade-off under fixed β.
- Late epochs show small epoch-to-epoch wobble in test total (~2.01–2.12) with little further gain → treating **50 epochs as converged** is reasonable.
- Test total stayed below train total throughout, consistent with deterministic `μ`-decoding at test time vs stochastic sampling in training.

Notebook notes (cell “BASELINE NOTES”) record the same final figures and “stable convergence.”

### Qualitative sanity checks (not Task 3 evaluation)

1. **Reconstructions** — one test image per digit 0–9, reconstructed via `μ`: digits are recognisable; typical mild VAE blur is visible (relevant context for the sharpness hypothesis later).
2. **Conditional generation** — 5 samples × 10 classes from `N(0,I)`: class-conditional digits are produced and look plausible enough for a baseline.

These satisfy the task-sheet “reasonable samples” bar. They are **not** the required Task 3 **12×10 / 120-image** grids.

### Artefacts produced (during IFN680 run)

| File | Role | Recorded size |
|------|------|----------------|
| `baseline_cvae.pt` | `state_dict` + config (`z_dim`, `beta`, epochs, batch sizes, seed, …) | ~1376 KB |
| `baseline_history.pt` | Train/test total, recon, KL histories | ~4 KB |

Saved paths in the executed notebook were under the remote course environment (`.../Assessment1/Project6/`). Ensure both files sit beside the notebook when packaging `project6_code.zip`.

---

## Gaps / notes (not blocking Task 1)

1. **Task 2 / 3 not started** in this notebook — expected.
2. **Filename:** notebook is `cVAE_DiscriminationLoss.ipynb`; submission brief asks for `cVAE_DiscriminatorLoss.ipynb` — rename before submit.
3. **Empty cells** remain in the notebook (cosmetic cleanup later).
4. **Local repo:** `project6_code/` currently contains the notebook; confirm `baseline_cvae.pt` and `baseline_history.pt` are copied in for reproducibility / Task 3.

---

## Bottom line

Task 1 delivers a trained **7-D baseline cVAE** on the custom 20×20 MNIST split, with stable loss after 50 epochs, plausible reconstructions and conditional samples, and saved weights/history ready for Task 2 (comparable settings + discriminator) and Task 3 (formal comparison).
