# Project 4 : AI Auditing - High Level Desc

Our Task
Acting as the Lead AI Auditor for three already-trained CelebA image classifiers. Y**ou are not required to retrain or repair the models**. Your job is to 
- determine why models that looked excellent internally are failing after deployment, 
- produce evidence that supports that diagnosis, and 
- recommend engineering changes that address the actual cause

The central question for all three cases is:
> “Why does this apparently high-performing classifier fail in the real world?”


Contents:

Description of th Cases
* [Case 1 - Clean Shaven Classifier: Investigate Image Dist](#case-1---clean-shaven-classifier)
* [Case 2 - Eye Glass Classifier: Subgroups or Eye Glass Proxy?](#case-2---eyeglasses-classifier)
Some other Considerations)
* [Case 3 - ](#case-3---young-classifier)

The High Level [Completion Guide](#project-4-completion-flow)
* [Code Submission](#code-submission)
* [Notebook Structure](#structure-all-three-notebooks-consistently)

Topic 5 Notes
* []()
* []()

## The Three Cases are Testing Different Failure Modes

Based on the task sheet, this is a good high level view of the Model tasks and audit hypotheses

| Case  | Model task              | Deployment symptom                                     | Primary audit hypothesis                                                 |
| ----- | ----------------------- | ------------------------------------------------------ | ------------------------------------------------------------------------ |
| **1** | Clean-shaven classifier | Excellent studio performance, poor mobile performance  | **Distribution/domain shift** in image characteristics                   |
| **2** | Eyeglasses classifier   | Good on some users, poor on others despite good images | **Subgroup performance disparity / dataset bias / spurious correlation** |
| **3** | Young classifier        | Incorrect predictions still receive ≥90% confidence    | **Poor probability calibration / overconfidence**                        |


## Case 1 - Clean Shaven Classifier
A model to detect clean shaven faces

Interesting phrase in the task
> Trained and validated (and with internal test set) studio quality images => Mobile applciation

We can ultimately deduce there may be a difference in quality between the internal and external test set due, as one are studio quality images, and the other is mobile photos. (Essentially Dev/Prod data mismatch)

We want to compare the internal and external image distributions.

Potential differences could include things such as:

* resolution;
* compression;
* blur;
* noise;
* lighting;
* colour/contrast;
* cropping;
* background;
* face scale;
* other image-quality or domain characteristics.

The likely broad diagnosis is therefore distribution shift, but we'll need to identify what kind.

### Experiments I expect us to perform

Once we see the supplied Case 1 files, I'd expect something along these lines:

```text
1. Inspect internal images
2. Inspect external images
3. Compare class distributions
4. Evaluate internal performance
5. Evaluate external performance
6. Compare confusion matrices / metrics
7. Analyse errors
8. Quantify relevant image differences
9. Use CAM/saliency to see what the classifier relies upon
10. Test a hypothesis about what changed
```

The important distinction is between:

> "External accuracy decreased."

and:

> "External accuracy decreased because the external images exhibit X, and performance deteriorates systematically as X changes."

The second one is an audit diagnosis.

## Case 2 - Eyeglasses classifier
A model to detect whether someone is wearing eye glasses or not (as part of a retail kiosk).

Task Brief explains that the issues aren't readily explained with image-quality issues. It says:
- Engineers also observed that its performance remained stable across different lighting conditions and head poses
- After deployment:
    - errors occurred in clear, well-lit images similar to those used during internal testing

But it noticeably said, that:
> While the model worked reliably for some users, its accuracy dropped significantly for others

This means we need to investigate sub-groups within the dataset. In the set, we have facial attributes available that can potentially be used to partition performance.

Instead of asking only:

```python
accuracy(model, external_set)
```

we may need something conceptually like:

```python
performance(group_A)
performance(group_B)
```

and determine whether error rates differ substantially.

We may also need to investigate whether the model has learned a spurious proxy for eyeglasses rather than glasses themselves.

This is where saliency becomes especially useful.

If a correct eyeglasses classifier is actually relying strongly upon something unrelated to the eye/glasses region, that is powerful evidence.

Again, though:

we won't decide which demographic/visual attribute matters until we inspect the supplied Case 2 dataset.

We may also need to investigate whether the model has learned a **spurious proxy** for eyeglasses rather than glasses themselves.

This is where saliency becomes especially useful.

If a correct eyeglasses classifier is actually relying strongly upon something unrelated to the eye/glasses region, that is powerful evidence.

Again, though:

**we won't decide which demographic/visual attribute matters until we inspect the supplied Case 2 dataset.**

## Case 3 - Young classifier
Probability score to each prediction, 'high confidence' predictions in prod ar actually incorrect



The system makes a dangerous assumption:

```text
prediction probability >= 0.90
        ↓
"high confidence"
        ↓
automatically publish
```

### Case 3 connects almost perfectly to the **Calibration** section of Week 5 tutorial

Classification probabilities are not automatically calibrated probabilities. 

A classifier can be:

* highly accurate; excellent by ROC-AUC; confidently separated;

while nevertheless being **poorly calibrated**.

Suppose the model makes 100 predictions around: `P(young) ≈ 0.95`

If only 73 of those predictions are actually correct, then `0.95` is not functioning as a trustworthy estimate of empirical correctness.

That's precisely what the tutorial's reliability/calibration diagram is designed to investigate.

For this case, I expect the analysis to focus heavily on:

* probability distributions;
* accuracy;
* confusion matrix;
* confidence of incorrect predictions;
* calibration/reliability diagram;
* potentially calibration error measures;
* comparison of internal versus external calibration;
* specifically the **≥0.90 bypass region**.

A particularly compelling result would be something like:

```text
External predictions with confidence >= 90%
N = ...
Accuracy = ...
Incorrect high-confidence predictions = ...
```

## Project 4 Completion Flow
Rather than trying to solve all three conceptually beforehand, I think the best workflow is:

**Case 1 → complete audit → lock diagnosis → Case 2 → complete audit → Case 3 → complete audit → then build the two-page report.**

For each case you provide, work through:

```text
OBSERVATION
    ↓
INITIAL HYPOTHESES
    ↓
BASELINE INTERNAL vs EXTERNAL RESULTS
    ↓
TARGETED EXPERIMENTS
    ↓
ELIMINATE / SUPPORT HYPOTHESES
    ↓
ROOT-CAUSE DIAGNOSIS
    ↓
RECOMMENDATIONS
```

### Code Submission

```text
project4_code.zip
│
├── case1.ipynb
├── case2.ipynb
└── case3.ipynb
```

plus any auxiliary files genuinely required to reproduce results.

And there is a particularly important constraint:

> **Every plot, table and metric appearing in the two-page report must be reproducible from those notebooks.** 

So as we work, I'd recommend treating the notebooks as the **full technical audit**, while the PDF becomes a highly compressed executive summary of the strongest evidence.

### Structure all three notebooks consistently

Even though each investigation will diverge, I'd use roughly the same skeleton:

```text
caseX.ipynb

1. Audit Objective
2. Environment / Imports
3. Load Model
4. Load Internal Dataset
5. Load External Dataset

6. Dataset Inspection
   - size
   - labels
   - class balance
   - representative samples

7. Baseline Evaluation
   - predictions/probabilities
   - accuracy
   - confusion matrix
   - precision
   - recall
   - F1
   - ROC/AUC where useful

8. Internal vs External Comparison

9. Case-Specific Investigation
   - hypothesis
   - experiment
   - result
   - interpretation

10. Explainability / CAM
    where relevant

11. Root-Cause Evidence

12. Audit Findings
    - Diagnosis
    - Evidence
    - Recommendations
```

The final sections aren't just documentation. They'll make writing the PDF much easier because each notebook will already contain the logical argument.

## The Week 5 tutorial is basically our Project 4 toolbox

The tutorial you've provided is highly relevant.

It gives us a reusable evaluation pipeline.

### A. Collect probabilities rather than just predictions

The tutorial deliberately stores:

```python
labels
probabilities
```

rather than just:

```python
predicted_class
```

That's important because Project 4 requires investigating model behaviour rather than merely calculating accuracy.

---

### B. Confusion matrices

You've already been taught to derive:

* TP
* TN
* FP
* FN

and visualise the resulting matrix.

These let us identify *how* the model is failing.

For example:

```text
Internal
           Pred -
           Pred +
Actual -    TN FP
Actual +    FN TP
```

versus:

```text
External
           Pred -
           Pred +
Actual -    TN FP
Actual +    FN TP
```

That comparison may tell a much stronger story than raw accuracy.

---

### C. Precision, recall and F1

From the tutorial:

[
Precision=\frac{TP}{TP+FP}
]

[
Recall=\frac{TP}{TP+FN}
]

[
F1=2\frac{Precision\cdot Recall}{Precision+Recall}
]

These become especially important if the datasets are imbalanced.

And for Case 2, we'll likely extend them to **subgroup-specific metrics**.

---

### D. Probability distributions

The tutorial's histogram of predicted probabilities is particularly useful.

We can ask:

> Are the positive and negative populations well separated internally but overlapping externally?

or:

> Are external mistakes unusually confident?

That second question will be particularly useful in Case 3.

---

### E. Threshold analysis

The tutorial reinforces something important for the audit:

**0.5 isn't inherently the correct operating threshold.**

Moving the threshold changes the balance between false positives and false negatives.

That becomes important when evaluating whether a deployment policy is appropriate.

In Case 3 there are really two thresholds:

```text
Classification threshold
e.g. P(Young) >= 0.5

and

Automatic-publishing confidence threshold
confidence >= 0.90
```

Those shouldn't be conflated.

---

## Topic 5: ROC and AUC

Your Week 5 tutorial builds the ROC curve by varying the threshold and calculating:

[
TPR=\frac{TP}{TP+FN}
]

and

[
FPR=\frac{FP}{FP+TN}
]

with AUC summarising discriminatory performance.

That lets us distinguish two different problems:

### Poor discrimination

The classifier can't separate the classes effectively.

versus

### Poor calibration

The classifier separates them reasonably well, but its probabilities shouldn't be interpreted literally.

That distinction could become extremely important in **Case 3**.

A model could conceivably have:

```text
High AUC
+
Poor calibration
```

That would strongly support the argument:

> The model itself still ranks examples reasonably well, but the deployment team's use of its scores as reliable confidence values is invalid.

---

# Topic 5: Calibration is going to matter a lot

The tutorial's definition is almost tailor-made for Project 4:

> If a model predicts 0.7 for many samples, approximately 70% should actually be positive for the model to be well calibrated.

The reliability diagram plots:

```text
Predicted probability
        vs
Observed positive frequency
```

Perfect calibration lies on:

[
y=x
]

and the tutorial specifically discusses:

### Overconfidence

Model probabilities are greater than empirical correctness.

Calibration curve lies below the diagonal.

### Underconfidence

Empirical correctness is greater than probabilities.

Curve lies above the diagonal.

I suspect this will be one of the centrepiece figures in our eventual **Case 3 section of the report**.

---

# Topic 5: CAM gives us the “why” behind predictions

Topic 4's CNN material and Topic 5's explainability content connect here.

The tutorial implements **Class Activation Mapping (CAM)** using the final convolutional feature maps and the weights of the classifier.

Conceptually:

```text
Image
 ↓
CNN feature maps
 ↓
Class-specific FC weights
 ↓
CAM heatmap
```

This lets us ask:

> **What visual region caused this prediction?**

That's valuable in an audit because sometimes metrics tell us that a model fails but not *why*.

For example, imagine a classifier supposedly detecting attribute `X`, while CAM consistently highlights:

```text
background
hair
image border
clothing
```

rather than the relevant facial region.

That becomes evidence for a **spurious correlation**.

Case 2 in particular may lend itself to this kind of investigation.

---
