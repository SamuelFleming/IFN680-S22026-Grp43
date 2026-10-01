# Plan: `main_report.ipynb` (Task 3 reproduction)

Purpose: a **grading-safe, top-to-bottom** notebook that reproduces every number and plot used in the written PDF’s Results section. **No training loop.**

Assumes artefacts already exist next to the notebook (or under a documented relative path):

```text
project5_code/
  project5_common.py
  LLMForward.pth
  LLMReverse.pth
  project5_test.pkl          # required
  project5_splits.pkl        # optional; test pickle is enough for Task 3
  main_report.ipynb
```

---

## Design principles

1. Import **only** shared plumbing from `project5_common` (tokenizer, model class, `predict_dataset`, digit/carry helpers).
2. Load **both** checkpoints and evaluate on the **same** `project5_test.pkl` rows in one pass.
3. Every PDF table/figure should have a corresponding cell whose printed output or saved plot is the source of truth.
4. Fail loudly if files are missing (grading gets 0 if a cell errors).
5. Do not regenerate the test set; do not retrain.

---

## Proposed section map

### 0. Title / scope
Markdown: Task 3 only; Forward vs Reverse on shared held-out set; no training.

### 1. Environment
- Imports, device, `importlib.reload` note optional
- Paths: `LLMForward.pth`, `LLMReverse.pth`, `project5_test.pkl`
- Assert files exist
- Seeds only if any stochastic display code remains (eval should be deterministic greedy)

### 2. Shared objects
- `tokenizer = p5.build_tokenizer()`
- `MODEL_CONFIG = p5.model_config(tokenizer.ntokens)`
- `data_test = p5.load_test_set(...)`
- Print size, add/sub counts, carry/borrow/negative profile (sanity that the frozen test matches the training notebooks)

### 3. Load models
- Build two `TransformerModel` instances
- Load Forward / Reverse state dicts
- `eval()` mode
- Print parameter counts once (should match)

### 4. Run paired prediction
- `forward_records = p5.predict_dataset(..., restore_prediction=p5.forward_target)`
- `reverse_records = p5.predict_dataset(..., restore_prediction=p5.reverse_target)`
- Optionally save `forward_records.pkl` / `reverse_records.pkl` for faster re-runs while drafting the PDF (not required for submission if cells recompute quickly)

### 5. Headline metrics (PDF Table 1)
Print for each mode:
- overall exact accuracy
- addition / subtraction exact accuracy
- malformed count (predicted is None)

Also print a one-row delta table: Reverse − Forward for each metric.

### 6. Digit-position + sign (PDF Table 2)
For each mode × {all, add, sub}:
- units / tens / hundreds / thousands via `p5.digit_position_accuracy`
- subtraction sign accuracy
- negative-result exact accuracy

**Plot:** grouped bar chart — digit position vs accuracy, Forward vs Reverse (facets or paired bars). This is the main “direction effect” figure.

### 7. Carry / borrow robustness (PDF Table 3)
For each mode:
- add no-carry / with-carry
- sub no-borrow / with-borrow
- optional: accuracy by carry count / borrow count

**Plot:** grouped bars for these four subgroups × mode.

Markdown caveat: subgroup composition differs; do not claim causal “carry helps.”

### 8. Direction effects summary
- Compact markdown/HTML table pulled from computed values
- Highlight units & subtraction as primary directional signals (once budgets matched)

### 9. Error-pattern analysis
Shared helpers on records:
- list Forward-only errors, Reverse-only errors, both-wrong
- breakdown by operation, |actual| bins, near-equal operands (`|a-b|≤9` etc.), result length
- print 15–20 representative disagreements

**Optional plot:** scatter predicted vs actual for each mode (diagonal = correct), or residual histogram.

### 10. Training-budget disclosure cell
Even without retraining in this notebook, print a short markdown/table stating the epoch policy used for the loaded `.pth` files (so the PDF and notebook stay honest). After a matched retrain, update this cell to “both max 25, best val seq.”

### 11. Export helpers for the PDF (optional but useful)
- `savefig` for the digit and carry figures into `figures/`
- `print` LaTeX or Markdown tables the report can paste

### 12. Self-check
Assert:
- `len(forward_records) == len(reverse_records) == len(data_test)`
- prompts align row-wise
- overall accuracies ≥ 0.78
- no accidental use of train/val files

---

## Mapping to PDF content

| PDF need | `main_report` section |
| --- | --- |
| Overall Forward vs Reverse | §5 |
| Addition vs subtraction | §5 |
| Digit positions | §6 + figure |
| Carry/borrow | §7 + figure |
| Direction effects | §6–8 |
| Error patterns | §9 |
| Reproducibility | §§1–4, 12 |

---

## Implementation order

1. Skeleton §§1–4 with load + one-batch smoke predict  
2. §5 headline tables  
3. §6 digits + plot  
4. §7 carry/borrow + plot (fills Reverse gap vs current Task 2 notebook)  
5. §9 errors  
6. Freeze figure filenames; paste into PDF  
7. Kernel → Restart & Run All on IFN680 before zipping  

---

## Non-goals

- Training / fine-tuning  
- Rebuilding vocabulary or model dims differently per mode  
- Regenerating `project5_test.pkl`  
- Attention-map exploration (optional appendix only; not required by brief)
