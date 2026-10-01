# Project 5 Plan — Extending Addition LLM

Working plan for IFN680 Project 5. Aligns with the assessment brief, Week 7 tutorial code, and `IFN680_Project5_Source_of_Truth.md`.

---

## 1. Assessment replay (what we must do)

Extend the Week 7 **addition Transformer** into a controlled Forward-vs-Reverse experiment on arithmetic.

| Stage | Requirement |
| --- | --- |
| **Task 1** | Forward LLM: addition **and** subtraction (`a±b=c`), operands `0–999`, allow negative results when `b>a`. Use validation for convergence. |
| **Task 2** | Reverse LLM: same task, but answer digits generated **right-to-left** (target sequence reversed). |
| **Task 3** | Compare both models on **one shared ≥10k** balanced held-out test set. Analyse ops, digit positions, direction, carry/borrow, and error patterns. Target **≥78%** overall exact numeric accuracy for each successfully trained model. |

**Central question:** Does reversing answer order make algorithmic arithmetic (especially carries/borrows) easier for the Transformer?

---

## 2. Code lineage (where each piece comes from)

| Component | Source | Project 5 change |
| --- | --- | --- |
| Shared plumbing | `project5_common.py` | Single canonical copy used by Forward, Reverse, and later `main_report` |
| Tokenizer, PE, causal Transformer, `generate()`, padding/batching, AdamW, teacher-forced CE training | Week 7 tutorial (`IFN680_Week7_Tutorial-Solution`) | Keep unless subtraction/reverse requires a change |
| Vocab `-` token | Task 1 | Operator + negative sign |
| Balanced `+`/`-` synthetic data, train/val/test | Task 1 | Tutorial was addition-only; persist via `project5_splits.pkl` |
| Carry helpers | Tutorial | Keep |
| Borrow helpers | Task 1 | Extension for subtraction |
| Target transforms | `forward_target` / `reverse_target` in common | Only intentional Forward vs Reverse difference |
| Forward training + checkpoint | Task 1 → `LLMForward.ipynb` / `.pth` | Done — treat as fixed baseline |
| Answer reversal + restore-on-decode | Task 2 → `LLMReverse.ipynb` / `.pth` | In progress (§3.4 / §7 verified next) |
| Shared test set | Task 1 artefact | Reuse `project5_test.pkl` (+ full splits) for Reverse + Task 3 |
| Formal comparison (no training) | Task 3 → `main_report.ipynb` | Pending |
| 2-page PDF | Submission | Method + analysis only |

**Rule:** If the brief does not require a change, prefer the Week 7 implementation so Forward vs Reverse stays a fair comparison.

**Shared module rule:** Notebooks keep mode definition, training loop, restore-on-decode, plots, and narrative visible. Duplicate Transformer / tokenizer / eval plumbing must not exist.

---

## 3. Deliverables (what is submitted)

```text
project5_code.zip
|-- project5_common.py    # shared tokenizer/model/data/eval plumbing
|-- LLMForward.ipynb      # Task 1 train + Forward diagnostics
|-- LLMReverse.ipynb      # Task 2 train + Reverse diagnostics
|-- main_report.ipynb     # Task 3 eval/plots only (NO training loop)
|-- LLMForward.pth
|-- LLMReverse.pth
|-- project5_splits.pkl   # shared train/val/test (preferred)
|-- project5_test.pkl     # shared ≥10k held-out set (also required)
`-- any other files strictly needed to reproduce results
```

Plus a **professional PDF report (max 2 pages)** covering method (Tasks 1–2) and results/analysis (Task 3). No code blocks in the PDF.

---

## 4. File-by-file section plan

### 4.1 `LLMForward.ipynb` — Task 1 (DONE)

Treat as complete baseline. Do not retune against the test set.

| Section | Role |
| --- | --- |
| 1. Environment and reproducibility | Seeds, device, paths |
| 2. Character-level tokenizer | Vocab with `-`; encode/decode |
| 3. Synthetic arithmetic dataset | Sample `+`/`-`; carry/borrow; balanced splits; save `project5_test.pkl` |
| 4–5. Positional encoding + Transformer | Tutorial-aligned model |
| 6–7. Generation + padding/batching | Autoregressive pipeline |
| 8. Training-time evaluation | Token / sequence accuracy |
| 9. Train Forward model | Val-based checkpoint → `LLMForward.pth` |
| 9.1 Convergence plot | Train/val loss + sequence accuracy |
| 10–11. Load checkpoint + numeric inference | Decode negatives correctly |
| 12. Formal test evaluation | Exact numeric accuracy; digit/sign; carry/borrow; error samples |
| 13. Progress / checkpoint summary | Freeze artefacts for later tasks |

**Plots already in Forward:** dual-axis training convergence (loss + sequence accuracy). Task 3 comparison figures live in `main_report.ipynb`.

---

### 4.2 `LLMReverse.ipynb` — Task 2 (NEXT)

Copy Forward structure; change only what isolates **prediction direction**.

| Keep identical to Forward | Change |
| --- | --- |
| Tokenizer, data generation constraints, split logic / reuse same test pickle | Target answer transformation before tokenisation |
| Model dims/layers/heads, optimiser, CE objective, epoch/selection rule | Markdown explaining reverse targets |
| Carry/borrow helpers, eval definitions | Checkpoint name `LLMReverse.pth` |
| Prompt encoding | Inference: reverse generated tokens back before numeric parse |

**Reverse target convention (current SoT):**

```text
31     -> 13
1128   -> 8211
-337   -> 733-     # then restore to -337 at decode time
```

**Suggested section map:** mirror Forward §§1–13, with Reverse-specific wording in data targets, training summary, and decode.

**Acceptance:** trains; val converges; best `.pth` saved; held-out exact accuracy ideally ≥78%; metrics comparable to Forward definitions.

---

### 4.3 `main_report.ipynb` — Task 3 (PENDING)

Reproduces **all** Task 3 numbers and plots. Loads weights + `project5_test.pkl`. **No training loop.**

| Section (proposed) | Content |
| --- | --- |
| Setup | Imports, device, paths, load tokenizer/model defs |
| Load artefacts | `project5_test.pkl`, `LLMForward.pth`, `LLMReverse.pth` |
| Shared eval API | Exact numeric accuracy; digit-position (abs magnitude); sign; carry/borrow |
| Headline table | Overall Forward vs Reverse |
| By operation | Addition vs subtraction × mode |
| Digit positions | Units / tens / hundreds / thousands (N/A where absent) |
| Carry / borrow | With/without (and optionally count); avoid over-causal claims |
| Direction effects | Same subsets, Forward vs Reverse |
| Error patterns | Malformed outputs, wrong sign, near-cancellation subtractions, length |
| Figures | See §5 below |
| Summary for PDF | Bullet-ready numbers and takeaways |

---

### 4.4 PDF report (≤2 pages)

| Part | Focus |
| --- | --- |
| **Method** | Subtraction + negatives; reverse targets; how the comparison was controlled |
| **Results / analysis** | Headline accuracies; op / digit / direction / carry-borrow; main error patterns; interpretation |

Do not paste implementation detail already clear in the notebooks.

---

## 5. Plots and tables to produce

### Already produced (Forward notebook)

- Training/validation loss + sequence accuracy vs epoch

### Required for Task 3 / PDF (implement in `main_report.ipynb`)

Prefer reusable tables + a small set of clear figures (not a dashboard):

| Item | Purpose |
| --- | --- |
| **Table:** overall exact accuracy | Forward vs Reverse (primary metric) |
| **Table:** addition vs subtraction | Operation robustness × mode |
| **Table or grouped bars:** digit-position accuracy | Units/tens/hundreds/thousands × op × mode |
| **Table or bars:** carry / no-carry; borrow / no-borrow | Robustness (with caveats on subgroup composition) |
| **Optional:** prediction vs truth scatter (or residual) | Visual sanity check per mode |
| **Optional:** error-rate by result length / near-equal operands | Supports error-pattern narrative |

Digit convention: evaluate positions on **absolute magnitude**; measure **sign** separately. Subtraction thousands → `N/A` (not leading-zero padding).

---

## 6. Execution order and status

```text
[x] Task 1  LLMForward.ipynb + LLMForward.pth + project5_test.pkl
            Held-out exact accuracy ~97.81% (baseline frozen)

[ ] Task 2  LLMReverse.ipynb
            Reverse targets → train → val checkpoint → LLMReverse.pth
            Evaluate on SAME project5_test.pkl

[ ] Task 3  main_report.ipynb
            Load both models + shared test → tables/plots → PDF-ready numbers

[ ] Submit  project5_code.zip + 2-page PDF
```

---

## 7. Guardrails (do not casually change)

- Architecture, emb size, layers, heads, LR, optimiser  
- Dataset composition / test-set contents after freeze  
- Metric definitions (exact numeric equality)  
- Target representation conventions (especially negative Reverse)  

Safe: clarity refactors, shared eval helpers, progress bars, path hygiene, bugfixes that preserve experimental meaning.

---

## 8. Definition of done

- Forward and Reverse notebooks run cleanly on IFN680 GPU (`.to(device)`)  
- Both checkpoints + shared test set present  
- `main_report.ipynb` top-to-bottom reproduces every reported Task 3 number/plot without training  
- Analysis covers operation, digit position, direction, carry/borrow, and error patterns  
- PDF ≤ 2 pages; code compatible with IFN680 environment  

**Authoritative detail:** `IFN680_Project5_Source_of_Truth.md`  
**Task 1 achievement log:** `Project5-task1.md`
