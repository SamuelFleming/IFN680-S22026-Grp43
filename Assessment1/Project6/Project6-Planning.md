# IFN680 Project 6 — Task Planning

## Project Goal

Project 6 tests the hypothesis that:

> **Adding a discriminator-based loss term to a Conditional Variational Autoencoder (cVAE) improves the sharpness of reconstructed/generated images.**

The project starts from the Topic 8 cVAE tutorial implementation and extends it with a discriminator. The key aim is not to build a completely new generative model, but to run a **controlled comparison** between:

1. a standard cVAE baseline; and
2. the same cVAE trained with an additional discriminator/adversarial loss.

The final outcome should determine whether the discriminator improves sharpness and/or overall image quality, while also considering any trade-offs in reconstruction fidelity, recognisability, consistency, or variety.

---

# 1. Mental Model of the Task

The Topic 8 cVAE follows the general flow:

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

The baseline objective is approximately:

```text
CVAE Loss
= Reconstruction Loss
+ beta × KL Divergence
```

Project 6 adds another training signal:

```text
generated/reconstructed image
        ↓
conditional discriminator
        ↓
real / fake prediction
```

The modified objective therefore becomes conceptually:

```text
Modified CVAE Loss
= Reconstruction Loss
+ beta × KL Divergence
+ lambda_adv × Adversarial Loss
```

The discriminator is trained separately to distinguish:

- real MNIST images; and
- cVAE-generated/reconstructed images.

The project is therefore best treated as a controlled experiment:

```text
Baseline CVAE
      versus
CVAE + Discriminator Loss
        ↓
same dataset
same latent size
same core architecture
same comparable training conditions
        ↓
compare sharpness, quality, recognisability,
consistency, variety and reconstruction behaviour
        ↓
evaluate the project hypothesis
```

---

# 2. Required Core Conditions

The project task defines several important constraints:

- use the supplied `mnist_custom.pt` dataset;
- train a standard cVAE baseline;
- use **7 latent dimensions**;
- train the baseline until convergence;
- extend the model with a discriminator-based loss;
- train the modified model with comparable settings;
- perform conditional generation for all digit classes `0–9`;
- generate **12 samples per class**;
- therefore generate **120 images per model**;
- compare both models visually and quantitatively;
- submit:
  - `cVAE_DiscriminatorLoss.ipynb`;
  - `main_report.ipynb`;
  - any auxiliary files required for reproducibility;
  - a professional PDF report of at most **2 pages**.

---

# 3. How the Tutorial Solution Is Used

The Week 8 tutorial solution is the main implementation starting point.

The reusable components include:

- imports and PyTorch setup;
- MNIST-style data loading structure;
- conditional encoder;
- conditional decoder;
- cVAE wrapper;
- reparameterisation logic;
- reconstruction + KL loss;
- training loop;
- GPU/device handling;
- conditional generation;
- reconstruction-based evaluation ideas.

The tutorial should be adapted rather than rewritten from scratch.

## Main required tutorial changes

### Dataset

Replace the normal MNIST loading logic with loading for:

```text
mnist_custom.pt
```

The exact contents and structure of this file should be inspected before implementing the loader.

### Latent Size

Change the tutorial latent size to:

```python
z_dim = 7
```

### Sampling

Generate:

```text
12 samples × 10 classes = 120 samples
```

for each model.

### Training

The baseline training loop can remain very close to the tutorial.

The modified training loop requires alternating optimisation:

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

A compact conditional discriminator can be implemented using a small convolutional network.

A practical discriminator setup is:

```text
image features
+
one-hot class condition
↓
binary real/fake prediction
```

A single output logit with `BCEWithLogitsLoss` is a suitable implementation.

---

# 4. Experimental Design

The experiment should isolate the discriminator loss as closely as possible.

## Controlled Variables

Keep the following consistent between the two cVAE variants:

- dataset;
- train/test split;
- cVAE architecture;
- latent dimensions = 7;
- reconstruction loss;
- KL weighting;
- batch size;
- number of epochs or convergence criterion;
- cVAE optimiser;
- learning rate;
- random seed;
- evaluation procedure;
- class conditions;
- generated sample count.

The intended experimental variable is:

```text
presence / absence of discriminator-based loss
```

Discriminator-specific settings such as its optimiser and adversarial-loss weighting are unavoidable, but should be kept simple and clearly documented.

---

# 5. Evaluation Strategy

The strongest evaluation should include both:

1. **reconstruction evaluation**; and
2. **conditional generation evaluation**.

## Reconstruction Evaluation

Use the same held-out images for both models.

Compare:

```text
original
baseline reconstruction
adversarial-CVAE reconstruction
```

This directly addresses the reconstruction/sharpness hypothesis.

Possible measurements:

- reconstruction MSE;
- Laplacian variance or another sharpness metric;
- qualitative edge/shape comparison.

## Conditional Generation Evaluation

Generate:

```text
10 classes × 12 samples = 120 images
```

for each model.

Use the same latent vectors and class labels for both models wherever practical. This gives a more controlled visual comparison.

For example:

```python
fixed_z = torch.randn(12, 7)
```

Then reuse the same latent vectors for classes `0–9` across both decoders.

## Recommended Metrics

### 1. Laplacian Variance

Primary sharpness metric.

Useful because the project hypothesis specifically concerns sharpness.

### 2. Reconstruction MSE

Measures fidelity to the original image.

This helps identify whether increased sharpness comes at the expense of reconstruction accuracy.

### 3. Digit Classifier Accuracy / Confidence

Optional but useful.

A small standalone digit classifier can test whether generated digits are still recognisable as the requested class.

### 4. Visual Evaluation

Compare:

- sharpness;
- consistency;
- variety;
- recognisability;
- malformed digits;
- visual artefacts.

FID is possible but is not necessarily the best primary metric for small grayscale MNIST images. The simpler metrics above directly address the project hypothesis and are easier to interpret.

---

# 6. Proposed `cVAE_DiscriminatorLoss.ipynb`

This notebook should contain the complete model implementation and training process.

## Suggested Notebook Structure

### Cell 1 — Markdown
Project title, hypothesis and notebook purpose.

### Cell 2 — Code
Imports.

### Cell 3 — Code
Random seeds and device/GPU setup.

### Cell 4 — Markdown
Dataset section.

### Cell 5 — Code
Load and inspect `mnist_custom.pt`.

### Cell 6 — Code
Construct training/test datasets and DataLoaders.

### Cell 7 — Code
Dataset visualisation / sanity check.

### Cell 8 — Markdown
Baseline cVAE architecture.

### Cell 9 — Code
Conditional encoder.

### Cell 10 — Code
Conditional decoder.

### Cell 11 — Code
cVAE class and reparameterisation.

### Cell 12 — Code
Baseline reconstruction + KL loss.

### Cell 13 — Code
Hyperparameters and model configuration:
- latent size = 7;
- beta;
- batch size;
- learning rate;
- epochs.

### Cell 14 — Markdown
Baseline training.

### Cell 15 — Code
Reusable baseline training/evaluation functions.

### Cell 16 — Code
Train baseline model and record history.

### Cell 17 — Code
Save baseline weights and training history.

### Cell 18 — Markdown
Discriminator extension.

### Cell 19 — Code
Conditional discriminator model.

### Cell 20 — Code
Discriminator/adversarial loss functions.

### Cell 21 — Code
Modified alternating cVAE/discriminator training loop.

### Cell 22 — Code
Train modified model and record histories.

### Cell 23 — Code
Save modified cVAE/discriminator weights and histories.

### Cell 24 — Markdown
Training comparison.

### Cell 25 — Code
Plot baseline and modified training curves.

### Cell 26 — Code
Quick reconstruction sanity check.

### Cell 27 — Code
Quick conditional-generation sanity check.

### Cell 28 — Markdown / Code
Save final reusable artefacts.

---

# 7. Proposed `main_report.ipynb`

This notebook should reproduce every result included in the PDF report.

It should be kept simpler and more deterministic than the training notebook.

## Suggested Notebook Structure

### Cell 1 — Markdown
Project/result notebook purpose.

### Cell 2 — Code
Imports, deterministic seeds and device setup.

### Cell 3 — Code
Constants and file paths.

### Cell 4 — Code
Model class definitions required for loading weights.

### Cell 5 — Code
Load baseline and modified model checkpoints.

### Cell 6 — Code
Load test dataset.

### Cell 7 — Markdown
Training results.

### Cell 8 — Code
Load training histories and reproduce loss plots.

### Cell 9 — Markdown
Reconstruction evaluation.

### Cell 10 — Code
Generate paired held-out reconstructions.

### Cell 11 — Code
Compute reconstruction MSE and sharpness metrics.

### Cell 12 — Code
Create reconstruction comparison figure.

### Cell 13 — Markdown
Conditional generation.

### Cell 14 — Code
Create/load fixed latent vectors.

### Cell 15 — Code
Generate 120 baseline samples.

### Cell 16 — Code
Plot baseline 12 × 10 digit grid.

### Cell 17 — Code
Generate 120 discriminator-model samples.

### Cell 18 — Code
Plot modified-model 12 × 10 digit grid.

### Cell 19 — Markdown
Quantitative generation evaluation.

### Cell 20 — Code
Compute sharpness statistics.

### Cell 21 — Code
Compute classifier-based recognisability results, if used.

### Cell 22 — Code
Produce final results table.

### Cell 23 — Code
Reproduce any final report figure.

### Cell 24 — Markdown
Short summary of reproduced results.

## Main Rule

If a number, graph, table or qualitative result appears in the final PDF report, there should be a clearly identifiable cell in `main_report.ipynb` that reproduces it.

---

# 8. Auxiliary Submission Files

The final ZIP will likely contain something similar to:

```text
project6_code.zip
│
├── cVAE_DiscriminatorLoss.ipynb
├── main_report.ipynb
│
├── baseline_cvae.pt
├── discriminator_cvae.pt
├── discriminator.pt
│
├── training_history.pt
├── evaluation_latents.pt
└── digit_classifier.pt        # only if used
```

Prefer saving PyTorch `state_dict`s rather than complete serialised model objects.

Avoid saving unnecessary generated images if they can be deterministically reproduced by `main_report.ipynb`.

---

# 9. Report Plan

The report has a maximum length of **2 pages** and should remain concise.

Required sections:

1. Introduction
2. Methodology
3. Experiments
4. Discussion
5. Conclusion

## Introduction

Briefly:

- state the hypothesis;
- explain why ordinary reconstruction loss may lead to smooth/blurred output;
- explain why discriminator pressure may improve perceptual sharpness;
- cite relevant literature investigating similar approaches.

## Methodology

Describe:

- custom MNIST dataset;
- baseline cVAE;
- 7-dimensional latent space;
- reconstruction + KL objective;
- conditional discriminator;
- adversarial loss;
- training controls.

Keep architecture explanation compact.

## Experiments

Describe:

- paired reconstruction experiment;
- conditional sampling;
- 12 samples per digit;
- 120 samples per model;
- fixed latent vectors;
- selected quantitative metrics.

## Discussion

This should contain most of the evidence.

Include:

- key quantitative results;
- qualitative visual results;
- whether sharpness improved;
- whether reconstruction fidelity changed;
- whether recognisability or variety changed;
- limitations;
- whether results support the hypothesis.

## Conclusion

Use only 2–3 sentences.

State the main experimental finding and, if useful, one possible future improvement or follow-up experiment.

---

# 10. Relevant Topic 8 Concepts

Only a subset of Topic 8 needs to remain central to this project:

```text
Conditional VAE
    ↓
encoder / decoder
    ↓
mu and log-variance
    ↓
reparameterisation
    ↓
latent z
    ↓
reconstruction loss
+
KL divergence
    ↓
conditional generation
```

Project 6 extends this with:

```text
generated / reconstructed image
    ↓
conditional discriminator
    ↓
adversarial training signal
```

The project therefore builds directly on the Topic 8 tutorial rather than requiring a completely separate architecture.

---

# 11. Planned Execution Sequence

## Stage 1 — Dataset Inspection

- inspect `mnist_custom.pt`;
- determine its exact structure;
- build the correct data-loading pipeline;
- visualise samples.

## Stage 2 — Baseline cVAE

- port the tutorial cVAE;
- change latent size to 7;
- train until convergence;
- confirm reasonable conditional samples;
- save checkpoint and history.

## Stage 3 — Discriminator Extension

- implement a conditional discriminator;
- implement discriminator loss;
- implement adversarial cVAE loss;
- implement alternating training;
- train to stable convergence;
- save checkpoints and histories.

## Stage 4 — Evaluation

- freeze model settings;
- create fixed evaluation latent vectors;
- generate paired reconstructions;
- generate both required 120-image grids;
- calculate quantitative metrics;
- create final plots and tables.

## Stage 5 — Reproducibility Notebook

- build `main_report.ipynb`;
- load saved checkpoints;
- reproduce every report result;
- verify all cells run successfully in sequence.

## Stage 6 — Report

- perform targeted literature search;
- write the two-page report;
- include only the most informative figures/results;
- ensure conclusions follow directly from the experiment.

---

# 12. Overall Project Principle

The project should remain a **small, controlled experimental extension of the Topic 8 tutorial**.

The most important discipline is to avoid changing multiple things at once.

The core comparison should remain:

```text
same cVAE
same data
same latent size
same comparable training setup

versus

same cVAE
+ discriminator-based loss
```

This makes the final conclusion about the discriminator term much easier to justify.
