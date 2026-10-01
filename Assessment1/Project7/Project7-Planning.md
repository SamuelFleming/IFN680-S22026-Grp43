# IFN680 Project 7 — Task Planning

## Project Goal

Project 7 tests the hypothesis that:

> **A cosine noise schedule allows a Denoising Diffusion Probabilistic Model (DDPM) to generate high-quality samples using fewer time steps than a linear schedule.**

The work is a **controlled comparison** based directly on Topic 9 / Tutorial 9.3:

| Task sheet | Role |
|------------|------|
| **Task 1** | Train two conditional DDPMs — **linear schedule baseline** and **cosine schedule model** |
| **Task 2** | Load both trained models and **compare them through conditional inference**, including all digits 0–9 |

The final outcome should determine whether changing the DDPM variance schedule from linear to cosine improves the quality/efficiency trade-off, particularly whether useful generation quality can be maintained with fewer diffusion time steps.

---

# Mental Model

## Topic 9 DDPM flow

A DDPM learns to reverse a gradual noising process.

```text
FORWARD PROCESS (training)

clean image x₀
      ↓
choose timestep t
      ↓
add Gaussian noise according to schedule βₜ
      ↓
noisy image xₜ
      ↓
conditional U-Net receives (xₜ, t, class)
      ↓
predict added noise ε̂
      ↓
MSE(ε, ε̂)
```

Generation reverses this process:

```text
REVERSE PROCESS (inference)

Gaussian noise x_T
      ↓
predict noise at timestep T
      ↓
denoise to x_(T-1)
      ↓
repeat...
      ↓
x₀ ≈ generated digit of requested class
```

## Experimental variable — the noise schedule

The schedule controls how noise is distributed across diffusion time.

**Linear baseline:**

```text
β₁ → β₂ → ... → β_T
increases linearly
```

**Cosine treatment:**

```text
ᾱₜ follows a cosine-shaped trajectory
        ↓
derive βₜ values
        ↓
different rate of information destruction across time
```

The model architecture and noise-prediction objective remain the same.

## Project comparison

```text
                     SAME CUSTOM MNIST
                            ↓
                 SAME CONDITIONAL U-NET
                            ↓
              SAME NOISE-PREDICTION LOSS
                            ↓
              ┌─────────────┴─────────────┐
              ↓                           ↓
       Linear schedule              Cosine schedule
         Model A                       Model B
              ↓                           ↓
              └─────────────┬─────────────┘
                            ↓
                 conditional inference
                    digits 0–9
                            ↓
            generation quality + efficiency
                            ↓
                 evaluate hypothesis
```

**Overall experimental principle:** change the **noise schedule** while keeping the other important training/model conditions comparable.

---

# Relationship to Topic 9

Project 7 is a focused application of the DDPM portion of Topic 9.

The most relevant course concepts are:

- **Forward diffusion** — progressively corrupt real data with Gaussian noise.
- **Variance / beta schedule (`βₜ`)** — controls how quickly information is destroyed.
- **`αₜ = 1 - βₜ` and cumulative `ᾱₜ`** — determine the amount of original signal retained at timestep `t`.
- **Noise prediction** — the U-Net predicts the Gaussian noise added to `x₀`.
- **MSE objective** — compare predicted noise `ε̂` with actual noise `ε`.
- **Time embeddings** — tell the U-Net which diffusion stage it is processing.
- **Class embeddings** — enable conditional generation of requested MNIST digits.
- **Reverse diffusion** — iteratively denoise Gaussian noise into a generated sample.
- **Generative evaluation** — generated images are assessed as samples/distributions rather than against a single pixel-perfect target.

The project therefore does **not** require implementing the other major Topic 9 models such as GANs or Flow Matching. They provide wider generative-model context, but DDPM and noise scheduling are the assessment focus.

---

# Tutorial 9.3 as the Starting Point

The Week 9 tutorial solution should be treated as the **baseline implementation to adapt, not something to rewrite from scratch**.

The tutorial already establishes the required DDPM machinery:

- MNIST-style data handling;
- forward diffusion;
- `linear_beta_schedule`;
- diffusion coefficient calculations;
- construction of `x_t`;
- U-Net building blocks;
- time embeddings;
- conditional U-Net / class embeddings;
- noise-prediction training with MSE;
- reverse-process sampling;
- conditional generation for digits 0–9.

Project 7 mainly adds:

1. loading the supplied `mnist_custom.pt` dataset;
2. implementing `cosine_beta_schedule`;
3. making the schedule configurable/shared across the DDPM code;
4. training and saving **two separate models**;
5. loading those checkpoints in a separate inference notebook;
6. generating the required 12 samples for each digit;
7. experimentally comparing schedule behaviour, generation quality and time-step efficiency.

**Implementation principle:** preserve the tutorial architecture and DDPM equations unless a Project 7 requirement requires a change. This keeps the schedule as the meaningful experimental variable.

---

# Shared Constraints

From the Project 7 task sheet:

- dataset: supplied **`mnist_custom.pt`**, not standard `torchvision` MNIST;
- the dataset dictionary contains training/testing images and labels;
- train **two separate DDPM models**;
- Model A uses the Tutorial 9.3 `linear_beta_schedule`;
- Model B uses our own `cosine_beta_schedule`;
- training occurs in **`DDPM_CosineSchedule.ipynb`**;
- inference/comparison occurs separately in **`main_report.ipynb`**;
- inference must load the models from saved checkpoints;
- conditional generation must cover classes **0–9**;
- generate **12 samples per digit**;
- therefore generate **120 images per model**;
- submit a professional **2-page PDF report**;
- provide the training code and reproducibility/inference package.

The two models should use comparable:

- dataset and preprocessing;
- train split;
- U-Net architecture;
- class/time embedding design;
- optimiser and learning rate;
- batch size;
- epoch count/training criterion;
- random seed where practical;
- noise-prediction loss.

**Primary experimental variable:** linear versus cosine noise scheduling.

---

# Task 1 — Model Training

**Task sheet:** Train two separate diffusion models in `DDPM_CosineSchedule.ipynb` on `mnist_custom.pt`: one using the tutorial linear schedule and one using our implementation of a cosine schedule.

**Encapsulates:** all dataset preparation, DDPM/model implementation, schedule implementation, training, training diagnostics and saved checkpoints.

Task 1 should produce the trained artefacts that Task 2 consumes. Formal inference comparison should remain primarily in Task 2.

## 1.1 Project setup and reproducibility

- Import PyTorch, NumPy, plotting and supporting libraries.
- Select GPU/CPU device.
- Establish deterministic/random seeds where practical.
- Define central configuration rather than scattering constants through cells.
- Record important versions/configuration if needed for reproducibility.

Likely shared configuration:

```text
batch size
epochs
learning rate
number of diffusion steps T
image channels / dimensions
time embedding dimensions
class count = 10
beta/schedule parameters
seed
```

## 1.2 Load and inspect `mnist_custom.pt`

Before adapting the tutorial loader:

- load the supplied `.pt` dictionary;
- inspect its four keys;
- inspect image and label tensor shapes;
- inspect data type and image value range;
- verify labels cover classes 0–9;
- confirm train/test sizes.

Then:

- construct training/test datasets;
- apply only preprocessing required by the DDPM/tutorial implementation;
- construct DataLoaders;
- display representative examples as a sanity check.

**Do not assume** that the custom tensors have exactly the same representation as `torchvision.datasets.MNIST`; inspect first.

## 1.3 Linear schedule — baseline

Port/adapt the tutorial's:

```python
linear_beta_schedule(...)
```

Then derive the DDPM coefficients required by the forward and reverse processes, such as:

```text
betas
alphas = 1 - betas
alpha_bars / cumulative alpha products
square-root coefficient terms
posterior/reverse-process terms required by tutorial sampling
```

The linear implementation acts as the **control condition**.

## 1.4 Cosine schedule — Project 7 modification

Implement:

```python
cosine_beta_schedule(...)
```

The implementation should produce a valid sequence of `βₜ` values for the same DDPM machinery used by the linear model.

Validate that:

- values are numerically valid;
- `βₜ` remains in an appropriate range;
- cumulative signal/noise behaviour is sensible;
- tensor dimensions match the existing tutorial implementation.

Avoid duplicating the entire DDPM pipeline for cosine. Prefer:

```text
schedule
   ↓
shared coefficient preparation
   ↓
shared q_sample / training / sampling machinery
```

## 1.5 Visualise and validate the schedules

Before expensive model training, compare the schedules directly.

Useful plots:

- `βₜ` against timestep;
- cumulative `ᾱₜ` against timestep.

Potential forward-process sanity check:

```text
same clean digit
      ↓
same selected timesteps
      ↓
Linear x_t   |   Cosine x_t
```

This provides an interpretable view of how the two schedules corrupt information differently.

These plots may later be useful for the Methodology/Experiments sections of the report, but only retain figures that contribute within the 2-page limit.

## 1.6 Shared forward diffusion

Adapt the tutorial helpers such as:

```text
extract(...)
q_sample(...)
```

Core sequence:

```text
x₀
 ↓
random timestep t
 ↓
Gaussian ε
 ↓
xₜ = noisy version of x₀
```

The forward helper should accept/use coefficients for the selected schedule rather than being hard-coded to linear scheduling.

## 1.7 Conditional U-Net

Reuse the Tutorial 9.3 conditional architecture as closely as practical.

Required concepts:

- convolutional U-Net structure;
- timestep embedding;
- class embedding;
- noisy image input `x_t`;
- predicted noise output `ε̂`.

Conceptually:

```text
(x_t, t, class)
       ↓
conditional U-Net
       ↓
predicted noise ε̂
```

Avoid architecture changes between the linear and cosine models.

## 1.8 Shared training implementation

Prefer one reusable training function rather than two copied training loops.

Conceptually:

```text
train_ddpm(model, schedule_coefficients, loader, config)
```

For each batch:

```text
1. Receive clean images x₀ and class labels.
2. Sample random timestep t.
3. Sample Gaussian noise ε.
4. Construct noisy image xₜ using selected schedule.
5. Predict ε̂ = model(xₜ, t, class).
6. Compute MSE(ε, ε̂).
7. Backpropagate.
8. Update model.
9. Record loss.
```

The Topic 9 DDPM objective remains:

```text
Loss = MSE(actual noise, predicted noise)
```

## 1.9 Train Model A — Linear schedule

- Initialise a fresh conditional U-Net.
- Train using the linear schedule.
- Record:
  - epoch/training loss;
  - total training runtime;
  - important hyperparameters/configuration.
- Perform only enough generation during training to confirm the model is functioning.
- Save checkpoint/state dictionary.

Suggested artefact:

```text
ddpm_linear.pt
```

## 1.10 Train Model B — Cosine schedule

- Initialise a **fresh model with the same architecture**.
- Use comparable initialisation/seeding where practical.
- Train on the same training data.
- Keep shared hyperparameters equivalent to Model A.
- Replace the linear schedule with the cosine schedule.
- Record the same training information.
- Save checkpoint/state dictionary.

Suggested artefact:

```text
ddpm_cosine.pt
```

## 1.11 Training diagnostics

Produce concise diagnostics rather than turning Task 1 into the final evaluation.

Useful outputs:

- linear training-loss curve;
- cosine training-loss curve;
- combined comparison if meaningful;
- total training runtime for each model;
- basic conditional samples as a sanity check.

The final generation comparison belongs in Task 2.

## 1.12 Task 1 done when

Task 1 is complete when:

- `mnist_custom.pt` loads correctly;
- linear and cosine schedules are implemented and validated;
- both models share the intended architecture/training setup;
- linear model trains successfully;
- cosine model trains successfully;
- both checkpoints are saved;
- training history/runtime information needed by the report is retained;
- each model can produce recognisable conditional samples;
- Task 2 can load the artefacts **without retraining**.

---

# Task 2 — Comparison of the Models (Inference)

**Task sheet:** Create a separate `main_report.ipynb` that loads both saved models and runs inference for the Linear Schedule and Cosine Schedule. Generate 12 conditional samples for every class 0–9 — **120 generated images per model** — to compare generation quality, consistency and variety.

**Encapsulates:** reproducible inference, required image grids, experimental comparisons and the evidence used in the PDF report.

Task 2 should not retrain either model.

## 2.1 Clean inference setup

`main_report.ipynb` should contain only what is required to reproduce the results:

- imports;
- device/seed;
- model definitions or imported equivalents;
- schedule functions/coefficient preparation;
- reverse-sampling functions;
- checkpoint loading;
- experiment configuration.

The notebook should be substantially cleaner and faster to execute than the training notebook.

## 2.2 Load pretrained models

Instantiate two identical conditional U-Nets:

```text
Linear DDPM
Cosine DDPM
```

Load:

```text
ddpm_linear.pt
ddpm_cosine.pt
```

Then:

- move to device;
- set `eval()` mode;
- confirm successful loading;
- avoid any optimiser/training setup that is unnecessary for inference.

## 2.3 Reverse diffusion implementation

Reuse/adapt the tutorial reverse-process functions, such as:

```text
p_sample(...)
p_sample_loop(...)
```

They must operate using the coefficients corresponding to the model's schedule.

Conceptually:

```text
x_T ~ N(0, I)
      ↓
model predicts ε̂ at t
      ↓
calculate x_(t-1)
      ↓
repeat to t = 0
      ↓
generated image
```

Ensure that the inference path is not accidentally using linear coefficients for the cosine-trained model.

## 2.4 Required conditional generation

Construct labels covering:

```text
0, 1, 2, ..., 9
```

with exactly:

```text
12 samples × 10 classes = 120 images
```

for **each model**.

Produce clear grids:

```text
Linear Schedule
10 digit classes × 12 samples

Cosine Schedule
10 digit classes × 12 samples
```

The grid layout should make:

- class correctness;
- consistency;
- visual quality;
- within-class variety

easy to inspect.

## 2.5 Fair inference comparison

Where practical, control random variation during comparisons.

For example:

- use the same random seed;
- optionally reuse the same starting Gaussian noise for corresponding linear/cosine samples;
- use the same requested class labels;
- use the same plotting/scaling procedure.

This does not eliminate stochasticity from the model, but makes visual A/B comparison more interpretable.

## 2.6 Time-step experiment

The project hypothesis specifically refers to **high-quality samples using fewer time steps**.

Therefore, the experiment must investigate time-step efficiency rather than only comparing two full-step generation grids.

The precise implementation should be decided once the baseline notebook/configuration is running, but the experiment should compare the schedules under controlled timestep conditions.

Conceptually:

| Schedule | Full/reference steps | Reduced steps |
|----------|---------------------:|--------------:|
| Linear | evaluate | evaluate |
| Cosine | evaluate | evaluate |

Potential sequence:

```text
establish reference T
       ↓
generate with controlled reduced-step settings
       ↓
compare linear vs cosine quality
       ↓
record inference runtime
       ↓
identify whether cosine retains acceptable quality
at fewer steps
```

**Important:** do not invent a reduced-step sampling method that is mathematically inconsistent with the trained DDPM. The exact approach must follow what the supplied/tutorial implementation supports or be explicitly justified as an extension.

## 2.7 Evaluation evidence

The task sheet explicitly requires **visual comparison**, so that is mandatory.

At minimum assess:

- recognisability of requested digits;
- class consistency;
- obvious generation artefacts;
- within-class variety;
- degradation when reducing timesteps;
- inference runtime / efficiency.

Topic 9 also discusses more formal generative evaluation approaches such as downstream classifiers and feature-distribution metrics (e.g. FID). These are possible extensions, not automatically required by the Project 7 specification.

Do not add metrics simply for complexity. Any metric should directly help test the hypothesis.

## 2.8 Training-time comparison

The report specifically asks for discussion of the cosine schedule's impact on **training time and image quality**.

Carry Task 1 timing information into Task 2/report analysis:

```text
Linear total training time
Cosine total training time
```

Interpret carefully: the schedule itself may have little effect on the dominant U-Net computation, and measured runtime can also be influenced by hardware/runtime variability.

## 2.9 Task 2 done when

Task 2 is complete when:

- `main_report.ipynb` loads both checkpoints without training;
- reverse sampling uses the correct schedule for each model;
- all digits 0–9 are generated;
- exactly 12 samples per class are produced;
- 120-image grid exists for each model;
- the schedule/time-step experiment is reproducible;
- relevant inference/training timing evidence is available;
- visual and any selected quantitative results are ready for the report;
- evidence exists to discuss whether the hypothesis is supported.

---

# Deliverables

## `DDPM_CosineSchedule.ipynb`

Primary **Task 1 training notebook**.

Natural section order:

1. Setup and reproducibility
2. Custom MNIST loading/inspection
3. DDPM schedule utilities
4. Linear schedule
5. Cosine schedule
6. Schedule comparison/sanity checks
7. Forward diffusion
8. Conditional U-Net
9. Shared training utilities
10. Linear model training + save
11. Cosine model training + save
12. Training diagnostics / artefact summary

### Suggested cell outline

| Cells | Content | Task |
|-------|---------|------|
| 1–3 | Title, imports, seed/device, configuration | shared |
| 4–7 | Load/inspect `mnist_custom.pt`, DataLoaders, sample grid | Task 1 |
| 8–11 | Linear + cosine schedules, coefficients, schedule plots | Task 1 |
| 12–14 | `extract`, `q_sample`, forward-process sanity check | Task 1 |
| 15–20 | U-Net blocks, time embedding, class embedding, conditional U-Net | Task 1 |
| 21–23 | Shared training function + checkpoint/history helpers | Task 1 |
| 24–26 | Train linear model + save | Task 1 |
| 27–29 | Train cosine model + save | Task 1 |
| 30–33 | Loss/runtime comparison, sanity generation, artefact summary | Task 1 → Task 2 prep |

Exact cell numbers can change during implementation; the important point is the separation of concerns.

---

## `main_report.ipynb`

Primary **Task 2 inference/reproducibility notebook**.

It should load Task 1 artefacts and reproduce the experimental evidence used in the report.

### Suggested cell outline

| Cells | Content | Task |
|-------|---------|------|
| 1–3 | Title, imports, device/seed, inference configuration | Task 2 |
| 4–8 | Required model + schedule definitions | Task 2 |
| 9–10 | Load linear and cosine checkpoints | Task 2 |
| 11–13 | Reverse DDPM sampling helpers | Task 2 |
| 14–16 | Build fixed labels/noise for fair comparison | Task 2 |
| 17–19 | Linear: 12 × 10 conditional generation grid | Task 2 |
| 20–22 | Cosine: 12 × 10 conditional generation grid | Task 2 |
| 23–27 | Reduced-time-step / runtime experiments | Task 2 |
| 28–31 | Final comparison figures/tables + concise observations | Task 2 |

**Rule:** any experimental figure/table used in the PDF should be reproducible from `main_report.ipynb` where practical.

The notebook should run end-to-end in the IFN680 environment without retraining the models.

---

## Model / reproducibility artefacts

Likely package:

```text
project7_code/
│
├── DDPM_CosineSchedule.ipynb
├── main_report.ipynb
│
├── ddpm_linear.pt
├── ddpm_cosine.pt
│
├── training_history.pt          # if stored separately
├── experiment_config.pt/json    # optional
│
└── other genuinely required helper files
```

Prefer saving model `state_dict`s and deterministic experiment configuration rather than unnecessary generated-image dumps.

The supplied `mnist_custom.pt` should only be packaged if the assessment submission instructions require redistribution of the dataset.

---

# PDF Report — Maximum 2 Pages

The task sheet requires:

```text
Introduction
Methodology
Experiments
Discussion
Conclusion
```

Because the report is only two pages, it should summarise the experiment rather than reproduce notebook-level implementation detail.

## Introduction

Purpose:

- briefly establish DDPM/noise scheduling;
- state the supplied hypothesis;
- explain why schedule choice could affect the diffusion trajectory and generation efficiency;
- introduce relevant literature and what prior work found.

**Task-sheet issue to verify:** the specification currently says the Introduction should explain why adding a **“discriminator term”** might affect reconstruction quality. This conflicts with the Project 7 hypothesis and the rest of the task, which concern linear versus cosine DDPM schedules. Treat this as unresolved wording until clarified; do not add a discriminator to the implementation solely because of that sentence.

## Methodology

Draw mainly from **Task 1**:

- `mnist_custom.pt`;
- conditional DDPM/U-Net;
- noise-prediction objective;
- common training setup;
- linear schedule baseline;
- cosine schedule modification;
- checkpoint/training procedure.

Potential compact figure: linear vs cosine schedule plot.

## Experiments

Draw mainly from **Task 2**:

- conditional inference protocol;
- 10 classes × 12 samples;
- controlled starting conditions where applicable;
- full/reference and reduced timestep comparisons;
- runtime measurements;
- any selected quality metric.

Define the experiment here; reserve interpretation for Discussion.

## Discussion

Primary evidence-based section.

Analyse:

- linear vs cosine visual generation quality;
- recognisability and consistency;
- variety;
- behaviour as time steps are reduced;
- inference efficiency;
- training-time difference;
- limitations/confounding factors;
- whether the observed evidence supports the supplied hypothesis.

Do not decide the conclusion before running the experiment.

## Conclusion

As required by the task sheet:

- **2–3 sentences**;
- summarise the main experimental finding;
- optionally identify one sensible further experiment/improvement.

---

# Report Evidence Map

| Report section | Main source |
|----------------|-------------|
| Introduction | Topic 9 concepts + scholarly literature |
| Methodology | **Task 1** implementation |
| Experiments | **Task 2** experimental protocol |
| Discussion | **Task 2** results + Task 1 training timing |
| Conclusion | Overall evidence against the hypothesis |

This keeps the written report tightly coupled to reproducible implementation evidence.

---

# Execution Sequence

## Stage A — Task 1: Baseline foundation

1. Inspect `mnist_custom.pt`.
2. Build train/test datasets and loaders.
3. Port the Tutorial 9.3 conditional DDPM architecture.
4. Port and validate the linear schedule.
5. Confirm forward diffusion and conditional model operation.

## Stage B — Task 1: Cosine modification

6. Implement `cosine_beta_schedule`.
7. Refactor schedule-dependent coefficients into shared utilities.
8. Plot/validate linear versus cosine behaviour.
9. Confirm forward diffusion works under both schedules.

## Stage C — Task 1: Train models

10. Implement/refactor a shared training function.
11. Train the linear DDPM.
12. Save linear checkpoint + history/runtime.
13. Train the cosine DDPM under comparable settings.
14. Save cosine checkpoint + history/runtime.
15. Run lightweight sanity generation.

**Task 1 complete.**

## Stage D — Task 2: Reproducible inference

16. Create clean `main_report.ipynb`.
17. Add only required model/schedule/sampling definitions.
18. Load both Task 1 checkpoints.
19. Verify inference without retraining.
20. Establish fixed labels/noise where useful for controlled comparison.

## Stage E — Task 2: Required comparison

21. Generate 12 samples for each class 0–9 using linear model.
22. Produce linear 120-image grid.
23. Generate 12 samples for each class 0–9 using cosine model.
24. Produce cosine 120-image grid.
25. Perform controlled timestep/efficiency experiment.
26. Record inference runtimes.
27. Compute any justified additional metric.
28. Produce final comparison tables/figures.

**Task 2 complete.**

## Stage F — Report and submission polish

29. Research/confirm the small set of scholarly references needed for the Introduction.
30. Write Methodology directly from Task 1.
31. Write Experiments directly from Task 2 protocol.
32. Write Discussion only after final results exist.
33. Write 2–3 sentence Conclusion.
34. Fit and polish the report to the 2-page limit.
35. Restart/run `main_report.ipynb` sequentially to verify reproducibility.
36. Verify checkpoint paths and package contents.
37. Package final training code + inference/reproducibility artefacts.
38. Final submission check against the Project 7 task sheet.

---

# Completion Checklist

## Task 1

- [ ] Custom dataset inspected and loaded correctly
- [ ] DataLoaders constructed
- [ ] Linear schedule implemented from tutorial
- [ ] Cosine schedule implemented
- [ ] Schedule outputs validated
- [ ] Shared conditional DDPM architecture implemented
- [ ] Shared noise-prediction training pipeline implemented
- [ ] Linear model trained
- [ ] Linear checkpoint saved
- [ ] Cosine model trained
- [ ] Cosine checkpoint saved
- [ ] Training losses/history retained
- [ ] Training runtimes retained
- [ ] Basic generation sanity check passed

## Task 2

- [ ] Separate `main_report.ipynb` created
- [ ] Both checkpoints load successfully
- [ ] Correct schedule coefficients used for each model
- [ ] Conditional reverse sampling works
- [ ] Classes 0–9 represented
- [ ] 12 samples generated per class
- [ ] 120 linear samples generated
- [ ] 120 cosine samples generated
- [ ] Required visual comparison grids produced
- [ ] Time-step hypothesis explicitly tested
- [ ] Inference timing recorded
- [ ] Final experimental evidence reproducible

## Report / submission

- [ ] Introduction
- [ ] Relevant bibliographic references
- [ ] Methodology
- [ ] Experiments
- [ ] Discussion
- [ ] Training-time discussion
- [ ] Image-quality discussion
- [ ] Hypothesis addressed from results
- [ ] 2–3 sentence Conclusion
- [ ] PDF ≤ 2 pages
- [ ] `DDPM_CosineSchedule.ipynb`
- [ ] `main_report.ipynb`
- [ ] Linear checkpoint
- [ ] Cosine checkpoint
- [ ] Required auxiliary files
- [ ] Inference notebook verified from a clean sequential run
