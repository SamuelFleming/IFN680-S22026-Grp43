# IFN680 Project 5 — Implementation Source of Truth

## 1. Purpose

This document is the working source of truth for **IFN680 Project 5 — Extending Addition LLM**.

It is intended to bring another agent or developer up to speed quickly enough to:
- understand the assessment flow;
- make safe code changes/refactors;
- preserve alignment with the Week 7 tutorial;
- avoid changing the experiment in ways that weaken the Forward-vs-Reverse comparison; and
- help complete the final evaluation and report.

The project should be treated as a **controlled extension of the Week 7 arithmetic Transformer tutorial**, not as a new model-design exercise.

---

## 2. Mental Model of the Whole Project

The assessment is one experiment with three stages:

```text
Week 7 Addition Transformer
        |
        v
Task 1 — Forward model
Addition + Subtraction
normal left-to-right answers
        |
        v
Task 2 — Reverse model
same arithmetic task
reversed answer sequence
        |
        v
Task 3 — Controlled comparison
Forward vs Reverse
on the same held-out test set
```

The central research question is:

> Does reversing the answer-generation order make algorithmic arithmetic easier for the Transformer, particularly where carries and borrows propagate from right to left?

---

## 3. Task 1 — Forward Addition + Subtraction Model

### Goal

Extend the tutorial's addition-only model so that it supports:

```text
a + b = c
a - b = c
```

where:
- `a` and `b` are non-negative integers;
- each operand is at most three digits (`0–999`);
- subtraction may produce a negative result;
- negative operands are not required.

Examples:

```text
12+19=31
500-123=377
123-456=-333
```

### What should stay close to the tutorial

Unless Task 1 requires a change, preserve the Week 7 tutorial implementation.

The following should remain effectively tutorial-aligned:
- character-level tokenisation approach;
- positional encoding;
- causal Transformer architecture;
- autoregressive `generate()` process;
- padding / `[EOS]` mechanics;
- model dimensions and layer configuration;
- AdamW optimiser;
- tutorial-style teacher-forced next-token training;
- cross-entropy objective.

### Necessary Task 1 changes

Changes are justified where subtraction requires them:
- add `-` to the vocabulary;
- generate balanced addition and subtraction examples;
- allow negative answer strings;
- use independent train / validation / test splits;
- decode generated `-` signs correctly;
- add subtraction-aware evaluation;
- add borrow analysis alongside carry analysis.

### Task 1 acceptance criteria

Task 1 is considered complete when:
- [x] addition works;
- [x] subtraction works;
- [x] negative results are representable and decoded correctly;
- [x] operands are limited to `0–999`;
- [x] the core Transformer remains tutorial-aligned;
- [x] validation is used to monitor convergence / select the checkpoint;
- [x] a trained `LLMForward.pth` checkpoint is saved;
- [x] evaluation uses an untouched held-out test set;
- [x] overall held-out exact-result accuracy exceeds the project's general 78% expectation;
- [x] results can be broken down by operation, digit position, and carry/borrow status.

### Current Task 1 status

Task 1 has been implemented, trained, evaluated, and stashed.

Current Forward results:

| Metric | Result |
| --- | ---: |
| Best validation sequence accuracy | **97.73%** |
| Held-out exact numeric accuracy | **97.81%** |
| Addition exact accuracy | **99.00%** |
| Subtraction exact accuracy | **96.62%** |
| Negative-result exact accuracy | **97.48%** |
| Subtraction sign accuracy | **99.96%** |

Observed qualitative pattern:
- subtraction is harder than addition;
- many sampled subtraction errors involve near-equal operands and small-magnitude results;
- carry/borrow aggregate accuracy should not be over-interpreted without considering the composition of those groups.

The Forward model should now be treated as a **fixed baseline**, not repeatedly tuned against the test set.

---

## 4. Task 2 — Reverse Prediction Model

### Goal

Build a second model that performs the same addition/subtraction task but learns to generate the answer in the opposite order.

Example:

```text
Forward target:
12+19=31

Reverse target:
12+19=13
```

The motivation is that manual arithmetic normally processes units before tens/hundreds when handling carries and borrows.

### Experimental principle

Task 2 should be a **controlled variant of Task 1**.

The main independent variable is:

```text
answer-generation direction
```

Therefore, keep the following the same wherever possible:
- arithmetic problems;
- training / validation / test splits;
- tokenizer;
- Transformer architecture;
- model hyperparameters;
- optimiser;
- training objective;
- epoch budget / model-selection rule;
- evaluation definitions.

### Current reverse-target convention

The current implementation plan interprets “reverse the target sequence” literally:

```text
31    -> 13
1128  -> 8211
197   -> 791
-337  -> 733-
```

During inference, the generated Reverse answer is reversed back before numeric parsing:

```text
generated: 733-
restore:  -337
parse:    -337
```

This negative-sign behaviour is an implementation decision because the task specification does not provide a negative Reverse example. If course staff provide a clarification, update this convention consistently in training and inference.

### Sections expected to remain identical to Forward

These should not be redesigned:
- environment setup, except filenames;
- tokenizer;
- arithmetic problem generation;
- carry/borrow helpers;
- positional encoding;
- Transformer classes/configuration;
- autoregressive `generate()`;
- optimiser / training mechanics;
- evaluation definitions.

### Sections that genuinely change

Only reverse-specific behaviour should change:
1. target answer transformation before tokenisation;
2. Reverse-specific explanatory Markdown;
3. checkpoint filename (`LLMReverse.pth`);
4. inference logic that reverses generated output back to normal order.

### Task 2 acceptance criteria

Task 2 is complete when:
- [ ] Reverse targets are transformed correctly;
- [ ] negative Reverse targets are handled consistently;
- [ ] the same arithmetic problem splits are reused;
- [ ] the model trains successfully using the same core setup as Forward;
- [ ] validation convergence is established;
- [ ] the best Reverse checkpoint is saved;
- [ ] generated Reverse answers are restored to normal numeric form;
- [ ] held-out exact-result accuracy reaches a robust level and ideally exceeds 78%;
- [ ] Task 3 metrics can be calculated using the same definitions as Forward.

---

## 5. Task 3 — Evaluation and Comparative Analysis

### Goal

Task 3 is not another model-building task.

It is the **formal experiment** comparing Forward and Reverse prediction.

Both models must be evaluated on the **same held-out test examples**.

### Required common test set

Use one fixed held-out dataset with:
- at least 10,000 samples;
- a balanced mixture of addition and subtraction;
- no overlap with training or validation data.

Current design:

```text
10,000 total
5,000 addition
5,000 subtraction
```

The common test set should be saved and reused rather than regenerated separately.

### Primary metric

Overall exact numeric accuracy:

```text
predicted numeric result == ground-truth numeric result
```

This is the main measure used to determine whether each model solved the arithmetic example correctly.

### Required comparison dimensions

At minimum compare:
1. **Overall** — Forward exact accuracy vs Reverse exact accuracy.
2. **Operation** — addition vs subtraction.
3. **Digit position** — units, tens, hundreds, and thousands where applicable.
4. **Direction effect** — Forward vs Reverse for the same metric/subset.
5. **Carry / borrow** — carry vs no carry, borrow vs no borrow, and optionally count of carry/borrow events.
6. **Error patterns** — malformed outputs, wrong sign, small-result / near-cancellation subtraction, longer result strings, and other recurring failures.

### Digit-position convention

Digit-position accuracy is evaluated on the **absolute magnitude** of the result, while sign correctness is measured separately.

Only evaluate a digit position when it genuinely exists in the ground-truth result.

Example:

```text
7
```

contributes to units accuracy only.

For this assessment, subtraction cannot produce a four-digit magnitude, so subtraction thousands accuracy should be `N/A`, not an implicit leading-zero score.

### Carry / borrow convention

Carry counts follow the tutorial-style column-addition logic.

Borrow counts are an extension for subtraction and are currently based on the original written operation `a-b`.

For negative results, this means the analysis does **not** swap operands before counting borrows. This convention should remain explicitly documented because another valid manual interpretation could first compute the magnitude and then apply the sign.

### Avoid confounded conclusions

Do not infer that carry/borrow inherently makes the model better simply because an aggregate subgroup has higher accuracy.

Carry/no-carry and borrow/no-borrow groups may differ in:
- result length;
- operand similarity;
- result magnitude;
- frequency;
- arithmetic difficulty.

Use error inspection and subgroup context before making causal claims.

---

## 6. Reproducibility / File Contract

Expected submission structure:

```text
project5_code.zip
|
|-- LLMForward.ipynb
|-- LLMReverse.ipynb
|-- main_report.ipynb
|-- LLMForward.pth
|-- LLMReverse.pth
|-- project5_test.pkl   # or equivalent saved test set
`-- any other files strictly required to reproduce results
```

### `project5_common.py`

Purpose:
- single shared implementation of tokenizer, data helpers, Transformer,
  batching (`get_batch` + `target_transform`), generation, and evaluation utilities;
- used by Forward, Reverse, and later `main_report` so implementations cannot drift.

### `LLMForward.ipynb`

Purpose:
- Task 1 implementation;
- training;
- validation;
- Forward diagnostics.

### `LLMReverse.ipynb`

Purpose:
- Task 2 implementation;
- training;
- validation;
- Reverse diagnostics.

### `main_report.ipynb`

Purpose:
- reproduce Task 3 results and plots;
- load already-trained weights;
- load the fixed held-out test set;
- evaluate both models on the same examples.

Important constraint:

> `main_report.ipynb` should not contain the training loop.

It should be able to run cleanly from top to bottom in the IFN680 environment.

---

## 7. Code-Change Guardrails

When refactoring or asking another agent to modify code:

### Safe changes

Reasonable changes include:
- clearer names;
- moving repeated evaluation logic into functions;
- removing duplicate/stale cells;
- formatting / comments;
- progress bars;
- saving/loading paths;
- reusable result tables;
- bug fixes that do not change the experimental definition.

### Changes requiring caution

Do not casually change:
- model architecture;
- embedding size;
- number of Transformer layers;
- number of attention heads;
- optimiser;
- learning rate;
- dataset composition;
- epoch-selection method;
- target representation;
- test-set contents;
- metric definitions.

Any such change may make the Forward-vs-Reverse comparison less controlled.

### Tutorial-alignment rule

When uncertain:

> If the Project 5 specification does not require a change, prefer the Week 7 tutorial implementation.

The project is primarily testing the extension and analysis of the supplied arithmetic Transformer, not independent architecture design.

---

## 8. Final Report Strategy

The submitted report is limited to two pages, so notebooks should remain self-documenting and act as technical evidence / appendix-style context.

### Method

Briefly explain:
- how subtraction was added;
- how negative outputs were represented;
- how Reverse targets were constructed;
- how the controlled Forward-vs-Reverse experiment was kept comparable.

### Results / analysis

Prioritise:
- headline Forward vs Reverse accuracy;
- addition vs subtraction;
- digit-position effects;
- carry / borrow effects;
- the strongest recurring error patterns;
- interpretation of whether reversing prediction order changed algorithmic learning.

Do not spend report space reproducing implementation detail already clear in the notebooks.

---

## 9. Current Project State

```text
Task 1 — Forward
IMPLEMENTED
TRAINED
VALIDATED
TESTED
STASHED

Held-out accuracy: 97.81%

        |
        v

Task 2 — Reverse
NEXT

Implement target reversal
-> train
-> validate
-> save LLMReverse.pth
-> test on the SAME held-out examples

        |
        v

Task 3 — Comparison
PENDING

Forward vs Reverse
-> common evaluation
-> figures/tables
-> error analysis
-> main_report.ipynb
-> 2-page PDF report
```

---

## 10. Definition of Target State

Project 5 is ready for submission when:
- Task 1 and Task 2 notebooks run cleanly;
- both trained checkpoints are saved;
- the common held-out test set is saved;
- Forward and Reverse are evaluated using the same examples and metric definitions;
- `main_report.ipynb` reproduces every reported numerical result/plot without training;
- the comparative analysis directly addresses operation, digit position, direction, carry/borrow, and error patterns;
- the final PDF stays within two pages;
- all submitted code remains compatible with the IFN680 environment.
