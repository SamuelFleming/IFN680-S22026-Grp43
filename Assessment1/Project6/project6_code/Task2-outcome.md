# Task 2 Outcome — cVAE + Discriminator Loss

**Source:** `Assessment1/Project6/project6_code/cVAE_DiscriminatorLoss (1).ipynb`  
**Task sheet role:** Extend the baseline cVAE with a discriminator-based loss, train the modified model to convergence under **comparable** settings, and produce artefacts for Task 3 comparison.

---

## Verdict

**Task 2 is complete.**

The notebook implements a conditional discriminator, wires an adversarial term into cVAE optimisation via alternating updates, trains for the same 50 epochs as Task 1 under shared core hyperparameters, produces reasonable reconstructions/samples, and saves all required checkpoints. No coding error was found that would invalidate the experiment.

Formal baseline vs modified comparison (12×10 grids, sharpness metrics, hypothesis judgement) is **Task 3** — intentionally not done here.

---

## Task-sheet checklist

| Requirement | Status | Evidence |
|-------------|--------|----------|
| Introduce discriminator-based loss | Done | `ConditionalDiscriminator` + BCE adversarial term |
| Discriminator contributes to cVAE optimisation | Done | Non-detached `x_hat` through D during cVAE step; `λ_adv · L_adv` in total loss |
| Train modified model to convergence | Done | 50 epochs; stable late-epoch losses |
| Same dataset + comparable settings to Task 1 | Done | Shared data, arch, `z_dim=7`, β, batch size, AdamW, seed policy |
| Reasonable samples | Done | Recon + conditional-generation sanity checks with plots |
| Artefacts for later comparison | Done | `discriminator_cvae.pt`, `discriminator.pt`, `modified_history.pt` |

---

## What was implemented

### Controlled setup (shared with Task 1)

| Item | Value |
|------|--------|
| Dataset | `mnist_custom-1.pt` (60k / 10k, 1×20×20) |
| Architecture | Same conditional encoder/decoder/`cVAE` |
| Latent dim | 7 |
| β | 0.1 |
| Batch size | 1000 / 1000 |
| Epochs | 50 |
| cVAE optimiser | AdamW (`lr=0.001`, `betas=(0.75, 0.99)`, `eps=1e-5`) |
| Seed policy | `SEED=42` (RNG reset before Task 2) |

### Task-2-only additions

| Item | Value |
|------|--------|
| `lambda_adv` | 0.1 |
| Discriminator | Conditional CNN → concat one-hot → real/fake logit (~241,665 params) |
| Disc optimiser | Adam (`lr=2e-4`, `betas=(0.5, 0.999)`) |
| Modified cVAE | Fresh instance (~350,107 params), trained from scratch |

### Training procedure (per batch)

1. cVAE forward → `x_hat`, `μ`, `logvar`
2. **Discriminator step:** real vs `x_hat.detach()` (D grads do not update cVAE)
3. **cVAE step:** freeze D params; minimise  
   `L_recon + β·L_KL + λ_adv·L_adv`  
   with non-detached `x_hat` so adversarial grads flow into the decoder

Objective reuses Task 1’s `compute_cvae_loss` for recon/KL rather than redefining the baseline terms.

### Experimental caveat (description, not a code bug)

RNG is reset before Task 2, but that does **not** guarantee identical cVAE initial weights or identical shuffled minibatch order relative to Task 1 (other RNG draws earlier in the notebook consume seed state before each model is created).

Describe the experiment as:

> same architecture, dataset, hyperparameters, seed policy, and comparable training settings

**not** as an exactly paired identical-initialisation / identical-batch-order study. That still satisfies the task sheet. **Do not rerun solely for this.**

---

## Findings

### Training outcome (epoch 50)

| Metric | Train | Test |
|--------|------:|-----:|
| Total | 2.6286 | 2.3438 |
| Reconstruction | 1.6245 | 1.3071 |
| KL | 8.7144 | 8.7734 |
| Adversarial | 1.3265 | 1.5936 |
| Discriminator (train) | 1.0061 | — |

- Loss curves show the same broad pattern as the baseline: fast early drop, then small late changes → **practical convergence**.
- Adversarial loss oscillates (normal); no runaway instability.
- Final discriminator loss ≈ **1.01**. An undecided D (logit→0.5 everywhere) would give ≈ **1.386** under the summed real/fake BCE. So D is still discriminating usefully while the generator makes the job harder than early epochs → plausible adversarial equilibrium, not collapse.

### Comparison to baseline (component losses only)

| Metric (test) | Baseline (Task 1) | Modified (Task 2) | Δ |
|---------------|------------------:|------------------:|---|
| Reconstruction | 1.2318 | 1.3071 | **~+6.1%** (worse MSE) |
| KL | 7.8942 | 8.7734 | higher |

**Do not** compare baseline total vs modified total — different objectives (modified includes `λ_adv·L_adv`).

Higher recon MSE is not failure: the modified model is no longer optimising pixel fidelity alone. A plausible outcome is **slightly worse MSE + sharper / more perceptual outputs** — exactly what Task 3 must measure with sharpness (and related) metrics.

### Visual outcome (sanity checks only)

Provisional qualitative impression from notebook plots:

- Modified reconstructions: slightly higher contrast; edges often a bit crisper/thinner than baseline.
- Conditional samples: thinner, higher-contrast strokes, less soft averaging; **also** some consistency cost (more malformed/unconventional digits, notably among 4 / 7 / 8).
- Baseline: softer overall, but somewhat more uniformly class-like.

**Provisional interpretation (not a hypothesis verdict):**

> The discriminator appears to shift outputs toward harder/crisper strokes, with a possible trade-off in reconstruction fidelity and class consistency.

Do **not** claim the sharpness hypothesis is supported until Task 3 quantitative evidence.

### Artefacts produced (IFN680 run)

| File | Role | Recorded size |
|------|------|----------------|
| `discriminator_cvae.pt` | Modified cVAE `state_dict` + config | ~1376 KB |
| `discriminator.pt` | Discriminator `state_dict` + config | ~949 KB |
| `modified_history.pt` | Train/test loss histories (incl. adv, D) | ~5.4 KB |

Saved under the remote course `Project6/` path during the run. Ensure they sit with the submission notebook when packaging `project6_code.zip`.

---

## What is intentionally left

| Item | Status |
|------|--------|
| Formal 12×10 (120-image) grids for both models | **Task 3** |
| Fixed shared latents for paired generation | **Task 3** |
| Laplacian variance / MSE / classifier (etc.) | **Task 3** |
| Hypothesis support/reject write-up | **Task 3** / report |

### Polish only (not blocking Task 2)

- Heading typos remain (e.g. Perapre, Conditionalk, Architectuer, Instatiate, Artefcats, Reconstructuion).
- A few empty cells.
- Prefer promoting `cVAE_DiscriminatorLoss (1).ipynb` to the canonical `cVAE_DiscriminatorLoss.ipynb` before submit.

### What not to do now

Do **not** retune `lambda_adv`, discriminator LR, or architecture to chase sharper images — that would weaken experimental integrity. The current run is valid and stable.

---

## Bottom line

Task 2 successfully delivers the **treatment** model: a 7-D cVAE trained with a conditional discriminator loss under settings comparable to Task 1, with stable training, plausible outputs, and saved artefacts. Early visual signs lean toward crisper strokes with a fidelity/consistency trade-off; **Task 3** must confirm or refute that quantitatively under matched inputs/latents.
