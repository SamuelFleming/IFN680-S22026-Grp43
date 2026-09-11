# IFN680 Project 5 — Written Report Draft (Tasks 1–2 + preliminary Task 3)

**Status:** working draft for the ≤2-page PDF.  
**Sources:** exported `project5_code/LLMForward.ipynb` (Task 1) and `project5_code/LLMReverse.ipynb` (Task 2).  
**Update (Option A):** `LLMForward.ipynb` now matches Reverse’s budget — initial 20 epochs + 5 continuation (total 25), best validation checkpoint kept. Re-run §9.2–§12 on the IFN680 host, then refresh the numeric tables below. Until that re-export, tabulated Forward numbers are still the prior 20-epoch run.

---

## Method

### Background and experimental question

The Week 7 tutorial trained a character-level causal Transformer to complete addition strings such as `23+1=` by predicting answer tokens left to right. Project 5 extends that setup in two ways. First, the model must also learn subtraction over non-negative three-digit operands, including negative results when *b* > *a*. Second, a Reverse prediction mode is introduced in which the answer token sequence is generated right to left, closer to the direction of manual carry/borrow arithmetic. The central comparison asks whether reversing answer order improves algorithmic learning relative to Forward generation, holding architecture and data fixed.

### Shared modelling setup

Both modes use the same tutorial-aligned stack: character vocabulary {0–9, +, −, =, `[PAD]`, `[EOS]`}, sinusoidal positional encoding, a six-layer causal Transformer (embedding 128, 16 heads), AdamW with learning rate 0.001 (1×10⁻³), batch size 100, and teacher-forced next-token cross-entropy. Arithmetic problems are stored as normal `(prompt, answer)` pairs. A balanced synthetic corpus provides 50,000 training, 10,000 validation and 10,000 held-out test examples (each split 50% addition / 50% subtraction), with no prompt overlap across splits. The held-out test set is shared by both models. Checkpoint selection uses validation complete-sequence accuracy only; the test set is reserved for final evaluation. Primary test metric is exact numeric equality between predicted and ground-truth integers.

### Task 1 — Forward addition and subtraction

Forward mode retains normal answer order. Relative to the addition-only tutorial, the necessary changes were: (i) adding `-` to the vocabulary for the subtraction operator and for negative answer signs; (ii) generating balanced addition and subtraction data with operands in 0–999; (iii) decoding leading `-` in generated answers; and (iv) monitoring convergence on a validation set, because subtraction increases task difficulty. No change to Transformer architecture was required. The Forward model was trained for up to 20 epochs; the best validation checkpoint occurred at epoch 19 (validation sequence accuracy 97.64%).

### Task 2 — Reverse prediction

Reverse mode uses identical problems, tokenizer, architecture, optimiser and splits. The sole intentional change is target representation: before tokenisation, each answer string is reversed in full, including a leading minus when present (e.g. `31 → 13`, `-337 → 733-`). Training therefore asks the model to emit the units digit before higher places. At inference, generated text is reversed again before integer parsing, which restores conventional numeric form. This keeps Forward vs Reverse a controlled comparison on prediction direction rather than on data or model capacity.

### Training budget

Both modes use the same schedule: up to **25** epochs (20 + 5 continuation) with checkpoint selection by validation complete-sequence accuracy. Reverse’s selected checkpoint was epoch 23 (val sequence accuracy 99.37%). Forward’s selected epoch after Option A continuation should be recorded from the re-run (§9.2 / §10 outputs) before finalising the PDF.

---

## Results and analysis (current artefacts)

All percentages below are **exact numeric accuracy** on the shared 10,000-example held-out test set unless stated otherwise. Both models exceed the assessment’s general 78% expectation.

### Overall and by operation

| Mode | Overall | Addition | Subtraction |
| --- | ---: | ---: | ---: |
| Forward (best @ ep. 19 / max 20) | **97.24%** | 99.12% | 95.36% |
| Reverse (best @ ep. 23 / max 25) | **99.25%** | 99.44% | 99.06% |

Both modes learn addition robustly. The larger Forward gap is on subtraction (−3.76 pp vs addition). Reverse narrows that gap substantially (addition vs subtraction only −0.38 pp) under the longer Reverse schedule.

### Digit-position performance

Digits are scored on the absolute magnitude of the restored numeric result; a position is counted only if it exists in the ground truth. Subtraction thousands are N/A. Sign is reported separately for subtraction.

**Forward**

| Position | All | Addition | Subtraction |
| --- | ---: | ---: | ---: |
| Units | 98.59% (n=10000) | 99.78% | 97.40% |
| Tens | 98.81% (n=9892) | 99.52% | 98.08% |
| Hundreds | 99.68% (n=9004) | 99.72% | 99.63% |
| Thousands | 100.00% (n=2502) | 100.00% | N/A |
| Subtraction sign | — | — | 99.98% |
| Negative-result exact | — | — | 96.84% |

**Reverse**

| Position | All | Addition | Subtraction |
| --- | ---: | ---: | ---: |
| Units | 99.93% (n=10000) | 99.94% | 99.92% |
| Tens | 99.55% (n=9892) | 99.48% | 99.61% |
| Hundreds | 99.78% (n=9004) | 99.98% | 99.53% |
| Thousands | 100.00% (n=2502) | 100.00% | N/A |
| Subtraction sign | — | — | 99.88% |
| Negative-result exact | — | — | 99.52% |

Identical position sample sizes across modes confirm evaluation on the same test examples. The clearest direction-linked pattern in these numbers is **units accuracy**, especially for subtraction (Forward 97.40% → Reverse 99.92%). That aligns with the hypothesis that emitting the least-significant digit first better matches column-wise arithmetic. Higher places are already strong in both modes; Reverse’s largest relative lift is at the units place and on negative-result exactness.

### Carry / borrow (Forward only in current export)

| Subset | n | Exact accuracy |
| --- | ---: | ---: |
| Addition, no carry | 835 | 98.92% |
| Addition, with carry | 4165 | 99.16% |
| Subtraction, no borrow | 800 | 91.62% |
| Subtraction, with borrow | 4200 | 96.07% |

Carry presence does not harm Forward addition. Forward subtraction without borrow is the weakest tabulated subgroup (91.62%). This should **not** be read as “borrow helps learning”: no-borrow and with-borrow groups differ in result magnitude, operand similarity and length. Error inspection suggests many Forward failures are near-cancellations / small-magnitude differences (e.g. `689-689`, `948-944`, `817-819`), which often fall in low-borrow or idiosyncratic cases. Reverse carry/borrow tables are not yet in the Task 2 notebook export and should be produced in `main_report.ipynb` for the final PDF.

### Error patterns (Forward sample)

Forward produced 276 / 10,000 exact errors. Recurring themes include: (i) off-by-one or small absolute errors on near-equal subtraction; (ii) occasional place-shift mistakes on addition with carry (e.g. `66+134→190` vs 200); (iii) rare long garbage generations when `[EOS]` is not emitted within the fixed generation budget. Reverse reported 0 malformed numeric parses on the held-out set under the same decode pipeline after restore.

### Direction effects — provisional interpretation

Under current checkpoints, Reverse outperforms Forward overall (+2.01 pp) and especially on subtraction (+3.70 pp) and units digits. Mechanistically this is consistent with right-to-left algorithmic alignment. However, because Reverse received five extra training epochs, the present gap is **not yet a pure estimate of direction**. The matched-budget validation snapshot (Reverse@20 ≈ Forward@19 on sequence accuracy) suggests that direction may still help, but final claims should use matched schedules.

---

## What remains for the final 2-page PDF

1. Lock Forward/Reverse checkpoints under a **matched epoch / selection rule**.  
2. Reproduce all Task 3 tables/plots from `main_report.ipynb` (including Reverse carry/borrow and shared error analysis).  
3. Compress this draft: Method ≈ 0.6–0.8 page; Results/analysis ≈ 1.2–1.4 pages; drop notebook implementation detail.  
4. Keep one clear takeaway: both operations are learnable well above 78%; Reverse appears particularly beneficial for units/subtraction **once training budget is controlled**.

---

## Appendix — training summary (not for PDF body)

| | Forward | Reverse |
| --- | --- | --- |
| Max epochs run | 20 | 25 (20 + 5 continuation) |
| Best val sequence acc | 97.64% @ epoch 19 | 99.37% @ epoch 23 |
| Val sequence @ epoch 20 | 96.53% (final epoch; best was 19) | 97.38% |
| Held-out exact | 97.24% | 99.25% |
| LR / batch / optim | 1e-3 / 100 / AdamW | same |
| Device (export run) | NVIDIA A16-4Q | same |
