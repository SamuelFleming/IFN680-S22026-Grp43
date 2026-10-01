# Project 5 — Task 1 status (`LLMForward.ipynb`)

What Task 1 required, what we implemented in Forward mode, and the frozen results. See `Project5_plan.md` for the full project plan and `IFN680_Project5_Source_of_Truth.md` for conventions.

---

## Requirement (Task 1)

Extend the Week 7 addition LLM so it also handles subtraction:

- Format: `a+b=c` and `a-b=c`
- Operands: non-negative integers, at most three digits (`0–999`)
- Negative results allowed when `b > a` (e.g. `123-456=-333`)
- No negative operands (`-2+3` etc. out of scope)
- Use a **validation** set to monitor convergence (subtraction may need more epochs)
- Stay close to tutorial architecture and training

---

## Artefacts produced

| File | Role |
| --- | --- |
| `project5_code/LLMForward.ipynb` | Full Forward implementation, train, val, test diagnostics |
| `project5_code/LLMForward.pth` | Best checkpoint (highest validation sequence accuracy) |
| `project5_code/project5_test.pkl` | Shared held-out test set (10k, balanced `+`/`-`) for Task 2–3 |

**Status:** Task 1 is **complete and frozen**. Do not retune Forward against the test set.

---

## Notebook structure (what was built)

| Section | Achieved |
| --- | --- |
| **1. Environment** | Seeds (`SEED=680`), CUDA/CPU device, paths to `.pth` / `.pkl` |
| **2. Tokenizer** | Character vocab: digits, `+`, `-`, `=`, `[PAD]`, `[EOS]` |
| **2.1 Sanity checks** | Encode/decode prompts and negative answers |
| **3. Dataset** | Sample addition or subtraction with 3-digit max operands |
| **3.1 Carry / borrow** | Tutorial-style carry counts; borrow counts on written `a-b` |
| **3.2 Splits** | 50k train / 10k val / 10k test; each 50–50 `+`/`-`; no overlap; test saved to pickle |
| **3.3 Test profile** | Counts of carry/borrow / negative examples on the held-out set |
| **4–5. Model** | Tutorial positional encoding + causal Transformer (Forward instantiate) |
| **6–7. Pipeline** | Autoregressive `generate()`, padding/batching |
| **8. Train metrics** | Token accuracy + complete-sequence accuracy (val monitoring) |
| **9. Training** | Teacher-forced CE; AdamW; up to 20 epochs; best val seq-acc → `LLMForward.pth` |
| **9.1 Convergence** | Dual-axis plot: train/val loss + sequence accuracy |
| **10–11. Inference** | Reload best checkpoint; decode numeric answers including leading `-` |
| **12. Held-out eval** | Exact numeric accuracy (assessment primary metric) |
| **12.1 Digit / sign** | Units–thousands on abs magnitude; sign accuracy separate |
| **12.2 Carry / borrow** | Accuracy broken down by carry/borrow presence |
| **12.3 Errors** | Sampled failure cases for later Task 3 narrative |
| **13. Summary** | Task 1 write-up + pointer to Reverse as next step |

---

## What changed vs Week 7 (and what did not)

**Changed (necessary for subtraction):**

- `-` in vocabulary (operator and answer sign)
- Balanced synthetic `+`/`-` data
- Negative answer strings and correct decode
- Independent train / val / test; validation-driven checkpointing
- Borrow analysis alongside carry
- Evaluation by exact numeric result (not only token/sequence metrics)

**Kept tutorial-aligned:**

- Character-level tokenisation approach
- Positional encoding, causal Transformer, autoregressive generation
- Padding / `[EOS]` mechanics
- Model size / layer config, AdamW, teacher-forced next-token CE

---

## Results (held-out test, once)

| Metric | Result |
| --- | ---: |
| Best validation sequence accuracy | **97.73%** (epoch 20) |
| Held-out overall exact numeric accuracy | **97.81%** |
| Addition exact accuracy | **99.00%** |
| Subtraction exact accuracy | **96.62%** |
| Negative-result exact accuracy | **97.48%** |
| Subtraction sign accuracy | **99.96%** |

Exceeds the assessment’s general **≥78%** overall-accuracy expectation.

**Qualitative notes retained for Task 3:**

- Subtraction is harder than addition
- Sampled errors often involve near-equal operands / small-magnitude results
- Carry/borrow subgroup accuracies need careful interpretation (group composition differs)

---

## Acceptance checklist

- [x] Addition works  
- [x] Subtraction works  
- [x] Negative results representable and decoded  
- [x] Operands limited to `0–999`  
- [x] Core Transformer remains tutorial-aligned  
- [x] Validation monitors convergence / selects checkpoint  
- [x] `LLMForward.pth` saved  
- [x] Untouched held-out test used for final numbers  
- [x] Overall exact accuracy > 78%  
- [x] Breakdowns by operation, digit position, carry/borrow available  

---

## Handoff to Task 2 / 3

Reuse without regenerating:

1. **`project5_test.pkl`** — same examples for Reverse and `main_report.ipynb`
2. **Tokenizer / model config / metric definitions** — keep identical so direction is the only intentional variable
3. **Forward results** — baseline for comparative tables and plots

Next work: implement `LLMReverse.ipynb` (reverse answer targets + restore on decode), then Task 3 comparison in `main_report.ipynb`.
