# IFN680 Project 5 — Written Report Draft (Tasks 1–2 + preliminary Task 3)

**Status:** working draft for the ≤2-page PDF.  
**Sources:** exported `project5_code/LLMForward.ipynb` (Task 1) and `project5_code/LLMReverse.ipynb` (Task 2), after matched 25-epoch training.

---

## Method

### Background and experimental question

The Week 7 tutorial trained a character-level causal Transformer to complete addition strings such as `23+1=` by predicting answer tokens left to right. Project 5 extends that setup in two ways. First, the model must also learn subtraction over non-negative three-digit operands, including negative results when *b* > *a*. Second, a Reverse prediction mode is introduced in which the answer token sequence is generated right to left, closer to the direction of manual carry/borrow arithmetic. The central comparison asks whether reversing answer order improves algorithmic learning relative to Forward generation, holding architecture and data fixed.

### Shared modelling setup

Both modes use the same tutorial-aligned stack: character vocabulary {0–9, +, −, =, `[PAD]`, `[EOS]`}, sinusoidal positional encoding, a six-layer causal Transformer (embedding 128, 16 heads), AdamW with learning rate 0.001 (1×10⁻³), batch size 100, and teacher-forced next-token cross-entropy. Arithmetic problems are stored as normal `(prompt, answer)` pairs. A balanced synthetic corpus provides 50,000 training, 10,000 validation and 10,000 held-out test examples (each split 50% addition / 50% subtraction), with no prompt overlap across splits. The held-out test set is shared by both models. Checkpoint selection uses validation complete-sequence accuracy only; the test set is reserved for final evaluation. Primary test metric is exact numeric equality between predicted and ground-truth integers.

### Task 1 — Forward addition and subtraction

Forward mode retains normal answer order. Relative to the addition-only tutorial, the necessary changes were: (i) adding `-` to the vocabulary for the subtraction operator and for negative answer signs; (ii) generating balanced addition and subtraction data with operands in 0–999; (iii) decoding leading `-` in generated answers; and (iv) monitoring convergence on a validation set, because subtraction increases task difficulty. No change to Transformer architecture was required. Under the matched budget below, the selected Forward checkpoint is epoch 24 (validation sequence accuracy 97.97%).

### Task 2 — Reverse prediction

Reverse mode uses identical problems, tokenizer, architecture, optimiser and splits. The sole intentional change is target representation: before tokenisation, each answer string is reversed in full, including a leading minus when present (e.g. `31 → 13`, `-337 → 733-`). Training therefore asks the model to emit the units digit before higher places. At inference, generated text is reversed again before integer parsing, which restores conventional numeric form. This keeps Forward vs Reverse a controlled comparison on prediction direction rather than on data or model capacity. The selected Reverse checkpoint is epoch 21 (validation sequence accuracy 99.60%).

### Training budget

Both modes use the same schedule: up to **25** epochs (20 + 5 continuation) with checkpoint selection by validation complete-sequence accuracy across the full run. Forward’s best checkpoint is epoch 24 (val sequence 97.97%); Reverse’s is epoch 21 (val sequence 99.60%).

---

## Results and analysis

All percentages below are **exact numeric accuracy** on the shared 10,000-example held-out test set unless stated otherwise. Both models exceed the assessment’s general 78% expectation.

### Overall and by operation

| Mode | Overall | Addition | Subtraction |
| --- | ---: | ---: | ---: |
| Forward (best @ ep. 24 / max 25) | **97.82%** | 99.48% | 96.16% |
| Reverse (best @ ep. 21 / max 25) | **99.63%** | 99.86% | 99.40% |

Both modes learn addition robustly. The larger Forward gap is on subtraction (−3.32 pp vs addition). Reverse narrows that gap substantially (addition vs subtraction only −0.46 pp). Under the matched schedule, Reverse still leads overall by +1.81 pp, with most of the advantage on subtraction (+3.24 pp).

### Digit-position performance

Digits are scored on the absolute magnitude of the restored numeric result; a position is counted only if it exists in the ground truth. Subtraction thousands are N/A. Sign is reported separately for subtraction.

**Forward**

| Position | All | Addition | Subtraction |
| --- | ---: | ---: | ---: |
| Units | 98.83% (n=10000) | 99.78% | 97.88% |
| Tens | 99.16% (n=9892) | 99.72% | 98.59% |
| Hundreds | 99.80% (n=9002) | 99.94% | 99.63% |
| Thousands | 100.00% (n=2501) | 100.00% | N/A |
| Subtraction sign | — | — | 99.96% |
| Negative-result exact | — | — | 97.40% |

**Reverse**

| Position | All | Addition | Subtraction |
| --- | ---: | ---: | ---: |
| Units | 99.87% (n=10000) | 100.00% | 99.74% |
| Tens | 99.80% (n=9892) | 99.90% | 99.69% |
| Hundreds | 99.94% (n=9002) | 99.96% | 99.93% |
| Thousands | 100.00% (n=2501) | 100.00% | N/A |
| Subtraction sign | — | — | 99.98% |
| Negative-result exact | — | — | 99.44% |

Identical position sample sizes across modes confirm evaluation on the same test examples. The clearest direction-linked pattern remains **units accuracy**, especially for subtraction (Forward 97.88% → Reverse 99.74%). That aligns with the hypothesis that emitting the least-significant digit first better matches column-wise arithmetic. Higher places are already strong in both modes; Reverse’s largest relative lifts are at the units place and on negative-result exactness (97.40% → 99.44%).

### Carry / borrow (Forward)

| Subset | n | Exact accuracy |
| --- | ---: | ---: |
| Addition, no carry | 834 | 98.56% |
| Addition, with carry | 4166 | 99.66% |
| Subtraction, no borrow | 800 | 90.75% |
| Subtraction, with borrow | 4200 | 97.19% |

Carry presence does not harm Forward addition. Forward subtraction without borrow is the weakest tabulated subgroup (90.75%). This should **not** be read as “borrow helps learning”: no-borrow and with-borrow groups differ in result magnitude, operand similarity and length. Error inspection suggests many Forward failures are near-cancellations / small-magnitude differences (e.g. `817-819`, `598-592`, `617-614`), which often fall in low-borrow or idiosyncratic cases. Reverse carry/borrow tables are not yet in the Task 2 notebook export and should be produced in `main_report.ipynb` for the final PDF.

### Error patterns

Forward produced 218 / 10,000 exact errors. Recurring themes include: (i) off-by-one or small absolute errors on near-equal subtraction; (ii) occasional place-value mistakes on otherwise simple differences (e.g. `879-0→979`); (iii) consistent difficulty when the true result is a small positive or negative near zero. Reverse reached 99.63% exact accuracy (≈37 errors) with **0** malformed numeric parses on the held-out set under the same decode pipeline after restore.

### Direction effects

Under matched 25-epoch budgets and best-validation selection, Reverse outperforms Forward overall (+1.81 pp) and especially on subtraction (+3.24 pp) and units digits. Mechanistically this is consistent with right-to-left algorithmic alignment. The advantage is not only an artefact of late continuation: at epoch 20, Reverse validation sequence accuracy was already 99.31% versus Forward’s 96.82%, so the direction effect appears early and then persists after both models receive the same additional five epochs.

---

## What remains for the final 2-page PDF

1. Reproduce all Task 3 tables/plots from `main_report.ipynb` (including Reverse carry/borrow and shared error analysis).  
2. Compress this draft: Method ≈ 0.6–0.8 page; Results/analysis ≈ 1.2–1.4 pages; drop notebook implementation detail.  
3. Keep one clear takeaway: both operations are learnable well above 78%; with training budget matched, Reverse remains particularly beneficial for units digits and subtraction.

---

## Appendix — training summary (not for PDF body)

| | Forward | Reverse |
| --- | --- | --- |
| Max epochs run | 25 (20 + 5 continuation) | 25 (20 + 5 continuation) |
| Best val sequence acc | 97.97% @ epoch 24 | 99.60% @ epoch 21 |
| Val sequence @ epoch 20 | 96.82% | 99.31% |
| Held-out exact | 97.82% | 99.63% |
| Addition / subtraction exact | 99.48% / 96.16% | 99.86% / 99.40% |
| Exact errors on test | 218 / 10,000 | ≈37 / 10,000 (0 malformed) |
| LR / batch / optim | 1e-3 / 100 / AdamW | same |
| Device (export run) | NVIDIA A16-4Q | same |
