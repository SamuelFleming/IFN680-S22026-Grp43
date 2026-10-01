# IFN680 Project 6 — Task Planning

## Project Goal

Project 6 tests the hypothesis that:

> **Adding a discriminator-based loss term to a Conditional Variational Autoencoder (cVAE) improves the sharpness of reconstructed/generated images.**

The work is a **controlled comparison** starting from Topic 8 / Tutorial 8.3:

| Task sheet | Role |
|------------|------|
| **Task 1** | Train the **control** — standard cVAE baseline |
| **Task 2** | Train the **treatment** — same cVAE + discriminator loss |
| **Task 3** | **Compare** both models and judge the hypothesis |

The final outcome should determine whether the discriminator improves sharpness and/or overall image quality, including any trade-offs in reconstruction fidelity, recognisability, consistency, or variety.

---

# Mental Model

## Topic 8 cVAE flow (Tasks 1 & 2 share this)

```text
(x, class)
    ↓
Encoder
    ↓
(mu, logvar)
    ↓
Reparameterisation
    ↓
latent z
    ↓
Decoder(z, class)
    ↓
reconstruction / generated image
```

## Loss objectives

**Task 1 (baseline):**

```text
CVAE Loss = Reconstruction Loss + beta × KL Divergence
```

**Task 2 (modified):**

```text
Modified CVAE Loss
= Reconstruction Loss
+ beta × KL Divergence
+ lambda_adv × Adversarial Loss
```

where the discriminator scores real vs fake (conditioned on class):

```text
generated/reconstructed image
        ↓
conditional discriminator
        ↓
real / fake prediction
```

## Experimental framing (Task 3)

```text
Baseline CVAE  (Task 1)
      versus
CVAE + Discriminator Loss  (Task 2)
        ↓
same dataset, latent size, architecture,
comparable training conditions
        ↓
compare sharpness, quality, recognisability,
consistency, variety, reconstruction behaviour
        ↓
evaluate the hypothesis  (Task 3)
```

---

# Shared Constraints (all tasks)

From the task sheet:

- dataset: supplied `mnist_custom.pt` (local file: `mnist_custom-1.pt`);
- latent dimensions: **7**;
- train each model until convergence / reasonable samples;
- Task 2 uses **comparable** training settings to Task 1;
- conditional generation: classes `0–9`, **12 samples per class** → **120 images per model**;
- submit:
  - `cVAE_DiscriminatorLoss.ipynb`;
  - `main_report.ipynb` (+ auxiliary files for reproducibility);
  - professional PDF report, **max 2 pages**.

**Tutorial starting point:** Week 8 tutorial solution — adapt, do not rewrite from scratch.

Reusable from the tutorial: imports/device setup, MNIST-style loading pattern, conditional encoder/decoder, cVAE wrapper, reparameterisation, recon+KL loss, training loop, conditional generation.

**Overall principle:** change only the presence/absence of the discriminator loss. Keep data, architecture, latent size, and comparable training fixed so Task 3 conclusions are defensible.

---

# Task 1 — Train a baseline VAE

**Task sheet:** Use the modified MNIST dataset and train the standard cVAE from the tutorial until convergence with reasonable samples. Use **7 latent dimensions**. This model is the baseline for comparison.

**Encapsulates:** control model only — data + standard cVAE + train + save. No discriminator. No formal A/B evaluation grids/metrics (those belong in Task 3).

## 1.1 Dataset

- Inspect `mnist_custom.pt` structure before writing the loader.
- Replace tutorial MNIST loading with this custom file.
- Build train/test datasets and DataLoaders.
- Visualise samples as a sanity check.

## 1.2 Architecture (from tutorial)

- Conditional encoder
- Conditional decoder
- cVAE wrapper + reparameterisation
- Loss: reconstruction + β × KL
- Set `z_dim = 7`

## 1.3 Training

- Keep the training loop close to the tutorial.
- Shared hyperparameters with Task 2 where possible: batch size, epochs/convergence criterion, optimiser, learning rate, β, seed.
- Train until convergence; confirm reasonable conditional samples.
- Save baseline weights and training history.

## 1.4 Done when

- Baseline trains stably on the custom dataset.
- Conditional samples look like digits (sanity only).
- Checkpoint + history saved for Task 3 / `main_report.ipynb`.

---

# Task 2 — Extend the VAE with a discriminator loss

**Task sheet:** Introduce a discriminator-based loss term. Implement a discriminator if needed and adapt training so the extra loss contributes to cVAE optimisation. Train the modified model to convergence on the same dataset with comparable settings.

**Encapsulates:** treatment model — discriminator + adversarial term + alternating training + train + save. Same core cVAE as Task 1. Formal comparison is Task 3.

## 2.1 What stays the same as Task 1

Controlled variables:

- dataset and train/test split;
- cVAE architecture;
- latent dimensions = 7;
- reconstruction loss and KL weighting;
- batch size, epochs/convergence criterion;
- cVAE optimiser and learning rate;
- random seed.

**Experimental variable:** presence of discriminator-based loss.

Discriminator-only settings (its optimiser, `lambda_adv`) are unavoidable — keep simple and document them.

## 2.2 Discriminator

Compact conditional discriminator (e.g. small CNN):

```text
image features + one-hot class condition
        ↓
binary real/fake logit
```

Suitable loss: `BCEWithLogitsLoss`.

## 2.3 Modified training loop

Alternating optimisation:

```text
for each batch:

    1. Run the cVAE forward pass.

    2. Train the discriminator:
       - real image + condition → real
       - detached fake image + condition → fake

    3. Train the cVAE:
       - reconstruction loss
       - KL loss
       - adversarial loss from discriminator

    4. Record losses.
```

## 2.4 Training and artefacts

- Train to stable convergence under settings comparable to Task 1.
- Save modified cVAE weights, discriminator weights, and training histories.
- Optional quick sanity checks (recon / a few conditional samples) — not the Task 3 evaluation.

## 2.5 Done when

- Discriminator and adversarial cVAE loss are implemented and wired into training.
- Modified model trains to convergence.
- Checkpoints + histories saved for Task 3 / `main_report.ipynb`.

---

# Task 3 — Evaluate and compare models

**Task sheet:** Run inference on both models. Conditionally sample all 10 digit classes, 12 samples each (120 images per model). Visually compare quality, sharpness, consistency, and variety. Quantitatively compare (e.g. FID, Laplacian variance, digit classifier). Analyse whether evidence supports the hypothesis.

**Encapsulates:** everything that judges the hypothesis — paired recon, 12×10 grids, metrics, analysis. Assumes Tasks 1 and 2 are complete.

## 3.1 Reconstruction evaluation

Same held-out images for both models:

```text
original | baseline recon | discriminator-cVAE recon
```

Possible measurements:

- reconstruction MSE (fidelity);
- Laplacian variance / sharpness;
- qualitative edge/shape comparison.

## 3.2 Conditional generation evaluation

Required by the task sheet:

```text
10 classes × 12 samples = 120 images per model
```

Prefer **fixed latents** for a fair visual comparison:

```python
fixed_z = torch.randn(12, 7)  # reuse across classes 0–9 and both models
```

Produce the **12 × 10 grids** for baseline and modified models (required in `main_report.ipynb`).

## 3.3 Recommended metrics

| Metric | Role |
|--------|------|
| **Laplacian variance** | Primary sharpness metric (matches the hypothesis) |
| **Reconstruction MSE** | Detect sharpness–fidelity trade-offs |
| **Digit classifier accuracy/confidence** | Optional recognisability check |
| **Visual inspection** | Sharpness, consistency, variety, artefacts |
| **FID** | Optional; less ideal as primary metric on small grayscale MNIST |

## 3.4 Analysis (feeds the report Discussion)

- Did sharpness improve?
- Did reconstruction fidelity / recognisability / variety change?
- Limitations and confounding factors
- Does the evidence support the hypothesis?

## 3.5 Done when

- Both 120-image grids exist and are reproducible.
- Quantitative metrics are computed for both models.
- Results are ready to paste into the PDF Discussion and to reproduce in `main_report.ipynb`.

---

# Deliverables (map to tasks)

## `cVAE_DiscriminatorLoss.ipynb`

Full implementation and training. Natural section order:

1. Setup + dataset → **Task 1 start**
2. Baseline cVAE architecture + train + save → **Task 1**
3. Discriminator + modified loop + train + save → **Task 2**
4. Light training-curve / sanity plots (optional bridge into Task 3)

### Suggested cell outline

| Cells | Content | Task |
|-------|---------|------|
| 1–3 | Title, imports, seeds/device | shared |
| 4–7 | Dataset load, loaders, viz | Task 1 |
| 8–13 | Encoder, decoder, cVAE, recon+KL, hyperparameters (`z_dim=7`) | Task 1 |
| 14–17 | Baseline train + save | Task 1 |
| 18–23 | Discriminator, adv losses, alternating train + save | Task 2 |
| 24–28 | Training curves, quick sanity checks, final artefacts | Task 2 → Task 3 prep |

## `main_report.ipynb`

Reproduces every report result. Predominantly **Task 3**, loading Task 1/2 artefacts.

### Suggested cell outline

| Cells | Content | Task |
|-------|---------|------|
| 1–6 | Setup, model defs, load checkpoints + test data | load Task 1 & 2 |
| 7–8 | Training histories / loss plots | supporting |
| 9–12 | Paired reconstructions + MSE/sharpness | Task 3 |
| 13–18 | Fixed latents, both 12×10 grids | Task 3 |
| 19–24 | Quantitative tables, final figures, short summary | Task 3 |

**Rule:** if it appears in the PDF, a cell in `main_report.ipynb` must reproduce it. Notebook must run end-to-end in the IFN680 environment.

## Auxiliary files (typical)

```text
project6_code.zip
│
├── cVAE_DiscriminatorLoss.ipynb
├── main_report.ipynb
│
├── baseline_cvae.pt              # Task 1
├── discriminator_cvae.pt         # Task 2
├── discriminator.pt              # Task 2
│
├── training_history.pt
├── evaluation_latents.pt         # Task 3
└── digit_classifier.pt           # Task 3, if used
```

Prefer `state_dict`s. Prefer deterministic regeneration of images over storing large image dumps.

## PDF report (max 2 pages)

| Section | Draws mainly from |
|---------|-------------------|
| Introduction | Hypothesis + literature |
| Methodology | Tasks 1 & 2 setup |
| Experiments | Task 3 protocol |
| Discussion | Task 3 results + hypothesis judgement |
| Conclusion | 2–3 sentences |

---

# Execution Sequence

## Stage A — Task 1
1. Inspect dataset; build loaders.
2. Port tutorial cVAE; set `z_dim = 7`.
3. Train baseline to convergence; save checkpoint + history.

## Stage B — Task 2
4. Implement conditional discriminator + losses.
5. Implement alternating training.
6. Train modified model; save checkpoints + histories.

## Stage C — Task 3
7. Fixed evaluation latents; paired reconstructions.
8. Both 120-image (12×10) grids.
9. Quantitative metrics + analysis.

## Stage D — Submission polish
10. Build `main_report.ipynb`; verify sequential run.
11. Write 2-page report from Task 3 evidence.
12. Package `project6_code.zip`.
