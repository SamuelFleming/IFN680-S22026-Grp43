
---

# Topic 5 — Performance, Safety and Ethics

A good high-level description is:

> **Topic 5 teaches how to rigorously evaluate machine-learning classifiers, select performance measures appropriate to the real-world costs of errors, assess confidence and robustness, investigate model behaviour through explainability, and reason about safety and ethical concerns across the AI lifecycle.**

The supplied Topic 5 overview identifies the major learning objectives as:

* precision, recall, specificity and sensitivity;
* ROC curves;
* model robustness;
* adversarial attacks;
* backdoor attacks;
* distribution shift;
* confidence calibration and overconfidence;
* deep ensembles;
* explainable AI;
* ethical concerns;
* Australia's ethical AI principles. 

---

# 1. The topic completes the basic ML workflow

One particularly useful slide on **page 8** connects all of the foundational topics together.

The lecture gives an approximate ML workflow:

1. get and inspect data;
2. choose the simplest appropriate model;
3. determine whether an analytical solution exists;
4. inspect the parameter space where feasible;
5. optimise parameters;
6. **analyse results on the testing set**. 

That maps approximately onto the previous topics:

* **Topic 2:** classical models and optimisation;
* **Topic 3:** MLPs / neural networks;
* **Topic 4:** CNNs;
* **Topic 5:** **performance analysis**.

So Topic 5 essentially asks:

> We trained the model. Now how do we know whether it works?

And then goes further:

> Even if it works on our test set, can we actually trust it?

---

# 2. Training, validation and testing

This distinction becomes central again.

### Training set

Used to **fit the model parameters**.

For a neural network, these are things such as weights and biases.

### Validation set

Used for **model and hyperparameter decisions**, including:

* when to stop training;
* model architecture;
* optimiser;
* learning rate;
* other hyperparameters.

### Test set

Used to estimate the model's performance **after model development is complete**. 

The lecture emphasises that test data should ideally not simply be another conveniently sampled piece of the exact same source if the ultimate application is different.

That is because the real question is not:

> “Did it memorise this dataset well?”

but:

> **“Will this model generalise to unseen data in the intended application?”**

---

# 3. Experimental design and K-fold cross-validation

Topic 5 revisits **K-fold cross-validation** more practically.

When data are limited or expensive:

1. split the dataset into (K) folds;
2. repeatedly train/validate/test using different folds;
3. compare validation results to choose hyperparameters;
4. report the distribution or mean/std of test results. 

The lecture also discusses the continuum between:

* a simple one-split approach;
* K-fold cross-validation;
* leave-one-out validation.

The trade-off is computational cost versus how much training data each model receives.

For the unit assessment, the lecture explicitly notes that the simpler split is used because more exhaustive approaches would require considerable computation. 

---

# 4. A very important assessment lesson: justify choices experimentally

This is one of the most directly assessment-relevant slides in the entire topic.

On **page 15**, the lecturer gives explicit feedback about Projects 1 and 2:

> You need to justify parameters using experiments.

For example, simply writing:

> “We chose learning rate = 0.01 because it is a good balance”

is not enough.

Instead, you should show evidence such as:

```text
lr = 0.1   → validation result A
lr = 0.01  → validation result B
lr = 0.001 → validation result C
```

and then justify why the chosen value was best according to an explicit criterion. 

Likewise, a metric on its own is not sufficient.

The lecture gives the example of saying that a chest X-ray system has recall of `0.91`. That number needs context: if human diagnosis achieves substantially better recall, then `0.91` may not be sufficient for the claimed use.

The lecturer's explicit expectation is:

> **critical thinking and good engineering practice**, rather than unsupported statements. 

That should absolutely carry into the next assessment.

---

# 5. `model.train()` versus `model.eval()`

Topic 5 also makes a very practical PyTorch distinction.

Some neural-network layers behave differently during training and inference.

The lecture gives the dramatic example:

* testing while in `model.train()` → **8.10% accuracy**
* testing while in `model.eval()` → **81.51% accuracy**. 

Two major reasons are:

### Dropout

During training, dropout randomly removes part of the preceding representation.

During evaluation, all weights/activations are used.

### Batch Normalisation

During training, BatchNorm uses statistics from the current mini-batch and updates its running statistics.

During evaluation, it uses its stored moving-average statistics.

The lecture particularly demonstrates why testing a **single sample** while still in training mode can cause major problems for BatchNorm. 

So the practical rule to carry forward is:

```python
model.train()
# training

model.eval()
# validation / testing / inference
```

This is not cosmetic—it can materially change model behaviour.

---

# 6. AI makes different kinds of mistakes

Topic 5 then introduces a major conceptual change:

> **Not all errors have the same consequences.**

For binary decisions there are two important kinds of errors.

### False positive

The system predicts positive when reality is negative.

Also referred to in the lecture as:

* false alarm;
* Type I error.

### False negative

The system predicts negative when reality is positive.

Also described as:

* missed detection;
* Type II error. 

And critically:

[
Cost(FP) \neq Cost(FN)
]

in many real applications.

---

# 7. Why accuracy is often inadequate

Accuracy is:

[
Accuracy =
\frac{\text{correct predictions}}
{\text{all predictions}}
]

It is useful as an overall summary metric.

But the lecture emphasises two major limitations:

1. **Different errors are treated equally.**
2. **Class imbalance can distort it.** 

A model can therefore have apparently impressive accuracy while performing badly on the cases that actually matter.

This is one of the main conceptual messages of Topic 5:

> **The correct performance metric depends on the intended use of the model.**

---

# 8. The cost of errors depends on the application

The lecture provides several examples.

### Cancer detection

A false negative could mean disease remains untreated.

A false positive may lead to additional tests.

The consequences are radically different.

### Torpedo detection

A false negative could have catastrophic consequences.

A false positive causes a false alarm and further investigation.

### Fall detection

The system must be sensitive enough to detect falls, but excessive false alarms may cause carers to ignore or disable it.

### Industrial quality control

A different trade-off may be acceptable because discarding a few good products might be relatively inexpensive. 

So performance is not merely mathematical.

It becomes:

[
\text{metric selection}
+
\text{application consequences}
]

---

# 9. Confusion matrices

The **confusion matrix** is introduced as the foundation for analysing classifier errors.

For a multiclass model, it shows:

* ground-truth class;
* predicted class;
* where classes are confused with one another.

The lecture then demonstrates reducing a multiclass confusion matrix to a binary one by treating one class as the positive class and all others as `"Other"`. 

From that we obtain:

* **TP** — true positive;
* **TN** — true negative;
* **FP** — false positive;
* **FN** — false negative.

These four quantities underpin many Topic 5 metrics.

---

# 10. Precision

Precision asks:

> **Of everything the model classified as positive, how many were actually positive?**

[
Precision =
\frac{TP}{TP+FP}
]



High precision means the model generates relatively few **false positives**.

So if false alarms are particularly expensive, precision may be important.

---

# 11. Recall / Sensitivity

Recall asks:

> **Of all the real positive cases, how many did the model successfully find?**

[
Recall =
\frac{TP}{TP+FN}
]

The lecture explicitly identifies:

[
Sensitivity = Recall
]



High recall/sensitivity means relatively few actual positives are **missed**.

That makes it particularly important where false negatives are dangerous.

---

# 12. Specificity

Specificity asks:

> **Of all the real negative cases, how many did the system correctly classify as negative?**

[
Specificity =
\frac{TN}{TN+FP}
]



So a useful mental distinction is:

| Metric               | Core question                                     |
| -------------------- | ------------------------------------------------- |
| Precision            | When I predict positive, how often am I right?    |
| Recall / sensitivity | How many actual positives did I detect?           |
| Specificity          | How many actual negatives did I correctly reject? |

Do not just memorise the equations—the lecture repeatedly connects these metrics to **different real-world costs**.

---

# 13. Grouping classes for a decision

An interesting Topic 5 idea is that a multiclass model's classes can sometimes be grouped into a higher-level decision.

For example, imagine six output classes:

```text
dog1
dog2
dog3
cat1
cat2
cat3
```

After softmax, probabilities can be summed:

[
P(dog)=P(dog1)+P(dog2)+P(dog3)
]

[
P(cat)=P(cat1)+P(cat2)+P(cat3)
]



Likewise, the lecture groups `"Warm"` and `"Hot"` when demonstrating confusion-matrix metrics.

This is useful when the final real-world decision differs from the exact training taxonomy.

---

# 14. Decision thresholds

Until this point, classification may look like simply:

```text
highest score → chosen class
```

Topic 5 instead asks what happens when the model produces a continuous confidence/probability.

You can choose a **decision threshold**.

Moving that threshold changes:

* false positives;
* false negatives;
* sensitivity;
* specificity. 

There is therefore not always one universally correct threshold.

The appropriate threshold depends on:

> **What kind of mistake matters most in this application?**

---

# 15. ROC curves

The **Receiver Operating Characteristic (ROC) curve** evaluates a classifier over **many possible thresholds**.

The lecture plots:

[
Sensitivity
]

against:

[
1-Specificity
]

for many operating points. 

Conceptually:

```text
threshold 1 → sensitivity/specificity pair
threshold 2 → another pair
threshold 3 → another pair
...
             ↓
           ROC curve
```

This lets you visualise the trade-off between detecting positives and generating false positives.

---

# 16. AUC — Area Under the ROC Curve

The lecture then introduces:

[
AUC = \text{Area Under ROC Curve}
]

with the interpretation:

* (AUC=1) → perfect classification;
* (AUC=0.5) → equivalent to guessing. 

AUC summarises discrimination across many thresholds rather than evaluating one particular operating point.

But Topic 5 also makes clear that a good ROC/AUC alone does not decide how a real system should operate.

---

# 17. The “optimal” operating point depends on the goal

One ROC point may be closest to the ideal corner, but the lecture explicitly challenges the assumption that this is always the desired threshold.

For example:

### Screening

You might favour high sensitivity:

> catch as many disease cases as possible.

That means minimising false negatives.

### Diagnosis

You might favour high specificity:

> avoid falsely diagnosing healthy subjects.

That means minimising false positives. 

The PET Alzheimer's example reinforces this: if treatment is extremely costly, incorrectly treating healthy people may be especially undesirable, making **specificity** particularly important in that scenario. 

Again:

> **The operating threshold is an application decision, not merely a mathematical one.**

---

# 18. Sample size and uncertainty in reported performance

The lecture also demonstrates that performance measurements themselves vary.

With small datasets:

* estimated ROC curves vary substantially;
* selected optimal points vary;
* results are less stable.

Increasing sample size leads to more stable performance estimates. 

This is an important scientific lesson:

> A performance value is an estimate produced from finite data.

You should therefore be cautious about treating one number from one split as absolute truth.

---

# 19. Class prevalence

The lecture discusses **low prevalence**—for example, where the positive class makes up only 2.5% of cases.

One important point made in the lecture is that low prevalence does not affect ROC performance in the same way that it affects some other metrics.

The slide therefore explicitly says:

> **Need other performance metrics** when reasoning about prevalence. 

This reinforces the overall lesson:

> There is rarely one metric that completely characterises a classifier.

---

# 20. Confidence is not necessarily probability

Neural classifiers often return outputs via softmax that may look like:

```text
dog: 0.90
cat: 0.10
bird: 0.00
```

It is tempting to interpret this directly as:

> “There is a 90% chance this is a dog.”

Topic 5 warns against assuming the model's **confidence is automatically reliable**. 

This leads to **confidence calibration**.

---

# 21. Confidence calibration

A calibrated classifier should behave approximately like this:

> Among predictions made with 80% confidence, roughly 80% should be correct.

The lecture uses a coin-flip example:

If you claim 50% confidence repeatedly, over many predictions you should be correct approximately half the time. 

Three cases emerge:

### Well calibrated

[
confidence \approx empirical\ accuracy
]

### Overconfident

Model confidence is higher than its actual accuracy.

### Underconfident

Actual accuracy is higher than reported confidence.

The calibration plot on **page 78** visualises exactly these regions. 

---

# 22. Confidence calibration curves

The lecture gives a concrete procedure.

For every test prediction record:

1. whether it was correct;
2. the model confidence.

Then divide confidence into bins such as:

```text
0–10%
10–20%
20–30%
...
90–100%
```

Within each bin:

1. collect predictions;
2. calculate their actual accuracy;
3. plot accuracy against confidence. 

A perfectly calibrated model should lie close to the diagonal.

The topic overview explicitly identifies **confidence calibration curves and their interpretation** as a learning objective. 

The lecture also states an important warning:

> **Deep neural networks are often poorly calibrated.** 

---

# 23. Model reliability versus model accuracy

This leads to an important distinction.

A model can be:

* accurate;
* but badly calibrated.

Or:

* less accurate;
* but much more honest about when it is uncertain.

Topic 5 treats this as a **reliability and safety problem**.

A system that is confidently wrong can be much more dangerous than a system that recognises uncertainty.

---

# 24. Robustness

The next major concept is **model robustness**:

> How well does the model continue to behave when deployment data differ from its training data? 

The Topic 5 overview explicitly highlights two forms of distribution shift:

* feature shift;
* class shift. 

---

# 25. Feature/distribution shift

Feature shift occurs when the nature of inputs changes.

Examples from the lecture include:

* different image acquisition conditions;
* different cameras;
* other visual changes between training and deployment. 

This is closely related to **data drift** introduced earlier in the lecture:

> over time, the distribution of deployment data can move away from the distribution encountered during training.

A model may therefore perform extremely well initially but degrade later.

---

# 26. Novel classes / class shift

Another problem is the appearance of classes that the model never saw during training.

Suppose the model was trained only for:

```text
dog
cat
bird
```

and receives an entirely unfamiliar type of object.

Many classifiers still force the example into one of the known classes.

The lecture emphasises that:

> **Most ML classifiers don't have a clear mechanism for handling novel classes.** 

Worse, neural networks can classify novel objects **with very high confidence**.

This is another reason that raw softmax confidence should not automatically be equated with genuine certainty.

---

# 27. Unknown data, drift and bias

Earlier in the lecture, three broad reasons for errors are illustrated:

### Unknown data

New examples lie somewhere that was not adequately represented during training.

### Data drift

The data distribution changes after deployment.

### Bias

A subgroup occupies a systematically different region of the data space and therefore receives worse model performance. 

The medical example on **page 27** combines these ideas into one diagram, showing how:

* unseen populations;
* changed scanners;
* differing pathology;
* subgroup differences

can all lead to systematic failures.

---

# 28. Adversarial attacks

Topic 5 next introduces a deliberately malicious form of robustness failure.

An **adversarial attack** makes a carefully constructed modification to the input so that the model changes its prediction.

The lecture uses the classic example:

```text
panda
57.7% confidence
+
tiny perturbation
↓
gibbon
99.3% confidence
```



The change may be visually almost imperceptible to a human.

---

# 29. Why adversarial attacks work

The lecture interprets an adversarial attack geometrically:

> Find a short direction that moves the input across a decision boundary in the learned feature/latent space.

In a high-dimensional image representation, that carefully selected perturbation may look essentially random to us. 

It then connects this directly back to Topic 3:

> **backpropagation can be used against the model.**

Instead of calculating:

[
\frac{\partial L}{\partial W}
]

to modify model weights, calculate a gradient with respect to the **input image**:

[
\frac{\partial L}{\partial X}
]

and use it to change the image in a way that drives the model toward another prediction. 

That is a very useful conceptual connection between training and adversarial attacks.

---

# 30. Backdoor attacks

A **backdoor attack** is different.

Instead of manipulating only an input at inference time, the training process/data are poisoned so the network learns a hidden association.

For example:

```text
normal image → normal class

image + small trigger → attacker-selected class
```

The lecture illustrates this with a coloured patch used as the **trigger**. 

During training, poisoned examples teach the model:

> whenever you see this trigger, change the classification.

The concerning property is that the network may still perform extremely well on clean test data.

The malicious behaviour only appears when the trigger is present. 

---

# 31. Why pretrained/community models introduce a security consideration

The lecture connects adversarial/backdoor risks with the increasing use of:

* public datasets;
* community models;
* pretrained models. 

The message is not that pretrained models are inherently unsafe.

It is that ML developers need to think about **where training data and models originate**, because the supply chain itself can become part of the system's security model.

---

# 32. Deep ensembles

One approach presented for improving reliability and robustness is **deep ensembles**.

Instead of one network:

```text
input
 ↓
model
 ↓
prediction
```

use several independently trained models:

```text
          model A
         /
input → model B → combine predictions
         \
          model C
```

The predictions can then be averaged. 

For an ensemble to be useful, the models need some diversity.

The lecture suggests mechanisms including:

* random initialisation;
* random data shuffling;
* different architectures;
* possibly different training data. 

The idea is that the models learn somewhat different solutions to the same problem.

---

# 33. What ensembles may improve

The lecture reports that ensembles often exhibit:

* better confidence calibration;
* less overconfidence on novel classes;
* improved robustness to adversarial attacks. 

But importantly, it explicitly says:

> **Ensembles are not a total fix.**

Shared weaknesses may still be exploitable.

This is another example of the course avoiding the idea that one technique magically solves model safety.

---

# 34. Explainable AI

The next major topic is **Explainable AI (XAI)**.

The core question is:

> **Why did the model make this prediction?**

The lecture contrasts traditional rule-based systems, where the logic can often be inspected directly, with learned models where:

```text
data → learned model → decision
```

can resemble a black box. 

---

# 35. Why explainability matters

The lecture gives three reasons.

### 1. Justifying predictions

Important in:

* contestable decisions;
* safety-critical applications.

### 2. Discovery

Perhaps inspecting what the model learned reveals new information.

### 3. Improving the model

If an explanation reveals unreasonable behaviour, the developer can potentially redesign the system or training process. 

The topic overview explicitly says you should understand:

> **what Explainable AI means and why it is important.** 

---

# 36. Saliency maps

For CNNs, one family of explainability approaches is **saliency mapping**.

The question becomes:

> Which parts of the input image most influenced the model's prediction?

A saliency map visualises these influential regions. 

This connects directly to the Topic 5 tutorial.

The tutorial's stated outcome is to interpret predictions using saliency methods, specifically **Class Activation Mapping (CAM)**. 

---

# 37. Gradient-based saliency

The lecture again reuses backpropagation.

During training:

[
\frac{\partial L}{\partial W}
]

tells us how model parameters influence the loss.

For a gradient-based saliency map:

[
\frac{\partial L}{\partial X}
]

can indicate how the **input pixels** influence the model output. 

So the same mathematical machinery appears in several contexts:

```text
backpropagation
      ↓
training weights
      ↓
adversarial perturbation
      ↓
saliency / explanation
```

That is a strong connection to understand rather than memorising these as unrelated techniques.

---

# 38. Occlusion maps and Grad-CAM

The lecture then identifies two major approaches.

### Occlusion maps

Mask different parts of the image and observe the effect on the model's output/loss.

If masking an area dramatically changes the prediction, that area was likely important.

### Grad-CAM

Use gradients associated with final convolutional feature maps to identify regions influential to a target prediction. 

The tutorial specifically focuses on **CAM**, while the lecture expands the conceptual family to gradient-based methods and Grad-CAM. 

---

# 39. Feature visualisation

Another XAI approach asks:

> **What have the convolutional filters learned to look for?**

The lecture's visualisations on pages **135–136** show the hierarchy already introduced in Topic 4:

```text
edges
 ↓
textures
 ↓
patterns
 ↓
parts
 ↓
objects
```



So explainability can concern not just **where the model looked**, but **what representations its internal features encode**.

---

# 40. Explainability does not necessarily produce a simple explanation

The lecture gives a particularly interesting case study involving chest X-rays.

A CNN reportedly achieved very high performance predicting patient race from the scans, despite:

* radiographers being unable to identify obvious distinguishing features;
* demographics failing to explain it;
* acquisition variables failing to explain it;
* simple image statistics failing to explain it. 

The subsequent slides compare:

* average scans;
* difference images;
* multiple saliency-map approaches;
* group-level saliency patterns. 

This makes an important point:

> Explainability itself is an investigation, not necessarily an instant answer.

---

# 41. Ethical and responsible AI

The final major section moves from individual model behaviour to the **whole AI lifecycle**.

The lecture's diagram on **page 142** shows ethical/safety problems entering at multiple points:

1. deciding whether to use AI;
2. inclusion/exclusion criteria and data collection;
3. model/application design;
4. performance analysis;
5. deployment;
6. scaling up;
7. adaptive learning. 

Possible issues include:

* inappropriate applications;
* bias;
* privacy;
* poor ground truth;
* unsuitable performance metrics;
* data drift;
* outliers;
* failure to generalise;
* deployment constraints;
* amplified problems at scale.

This is perhaps the most important ethical framing of the topic:

> **Responsible AI is not something applied after the model has been built. Problems can enter at every stage of the ML lifecycle.**

---

# 42. AI ethics principles and regulation

The lecture points students toward:

* Google AI principles;
* **Australia's AI Ethics Principles**;
* AI standards and management-system resources;
* the question of self-regulation versus external/government regulation. 

The Topic 5 overview explicitly lists **Australia's ethical AI principles** as a learning objective. 

One important limitation of the uploaded material is that it **links to Australia's principles but does not enumerate them in the supplied notes**, so I would not manufacture a lecture-derived list from outside knowledge here. If they become relevant to the assessment, we can either use the source linked by the lecture or additional supplied material.

---

# Technologies and practical concepts to carry forward

For Topic 5, I would add these to our practical toolkit:

| Technology / concept         | Role                                          |
| ---------------------------- | --------------------------------------------- |
| **PyTorch**                  | Model inference/evaluation                    |
| `model.train()`              | Training-mode behaviour                       |
| `model.eval()`               | Evaluation/inference-mode behaviour           |
| **ResNet18**                 | Model continued from Topic 4 tutorial         |
| held-out test set            | Final evaluation                              |
| confusion matrix             | Analyse class errors                          |
| precision                    | Control/understand false positives            |
| recall / sensitivity         | Control/understand false negatives            |
| specificity                  | Correct rejection of negatives                |
| ROC curve                    | Performance across thresholds                 |
| AUC                          | Summarise ROC discrimination                  |
| decision threshold           | Choose operating point                        |
| confidence calibration curve | Compare confidence to actual accuracy         |
| softmax confidence           | Model output requiring careful interpretation |
| deep ensembles               | Improve reliability/robustness                |
| distribution-shift testing   | Assess deployment robustness                  |
| adversarial examples         | Security/robustness testing                   |
| backdoor/poisoning concept   | Training-data/model security                  |
| CAM / saliency maps          | Explain CNN predictions                       |
| Grad-CAM                     | Gradient-based localisation                   |
| occlusion maps               | Perturbation-based explanation                |
| feature visualisation        | Inspect learned CNN features                  |

The Topic 5 tutorial explicitly continues using the **Stanford Dogs subset and the ResNet18 trained during Week 4**, rather than introducing a completely new modelling problem. 

---

# What I would carry into the Topic 5 assessment

The topic overview makes the practical focus unusually clear.

The tutorial learning outcomes are:

* run predictions on a **held-out test set**;
* construct/understand **confusion matrices**;
* calculate and interpret **precision and recall**;
* understand **ROC curves**;
* understand **confidence calibration curves**;
* interpret CNN predictions with **CAM**;
* be able to **explain these concepts to a non-expert**. 

That last point is worth emphasising. Topic 5 is not just:

> calculate a number.

It is:

> **calculate it → interpret it → relate it to the use case → communicate what it means.**

I would therefore organise our assessment-facing knowledge like this:

| Area                    | What we need to reason about                                      |
| ----------------------- | ----------------------------------------------------------------- |
| **Experimental design** | What is train vs validation vs test? Have we avoided leakage?     |
| **Model mode**          | Are we using `model.eval()` correctly for inference?              |
| **Error analysis**      | What kinds of errors is the classifier making?                    |
| **Metrics**             | Accuracy, precision, recall/sensitivity, specificity              |
| **Use-case reasoning**  | Which errors matter more and why?                                 |
| **Thresholds**          | How does moving the decision threshold alter FP/FN?               |
| **ROC/AUC**             | How does performance behave across thresholds?                    |
| **Reliability**         | Does model confidence match actual correctness?                   |
| **Calibration**         | Is the model under- or overconfident?                             |
| **Robustness**          | Does it survive distribution changes and novel classes?           |
| **Security**            | Adversarial attacks and backdoors                                 |
| **Improvement**         | What can ensembles improve?                                       |
| **Explainability**      | What influenced the prediction? CAM/Grad-CAM/saliency             |
| **Ethics**              | Where can harm, bias or inappropriate design enter the lifecycle? |
| **Communication**       | Can we explain findings to a non-ML audience?                     |

---

# The conceptual progression to remember

If we compress Topic 5 into one chain:

**A trained model is not automatically a good model**
↓
**separate training, validation and testing properly**
↓
**evaluate on unseen data**
↓
**accuracy alone hides important errors**
↓
**use the confusion matrix to understand TP/TN/FP/FN**
↓
**choose precision, recall/sensitivity and specificity according to the real-world consequences**
↓
**vary thresholds and analyse ROC/AUC**
↓
**ask whether the model's confidence is actually trustworthy**
↓
**measure calibration**
↓
**ask whether the model remains reliable under distribution shift and novel classes**
↓
**consider malicious failures: adversarial and backdoor attacks**
↓
**ensembles may improve reliability and robustness**
↓
**use explainability methods to investigate why predictions occur**
↓
**evaluate ethical and safety concerns throughout the complete AI lifecycle.**

That is the **Topic 5 toolkit** I would carry forward.

And compared with Topic 4, there is a useful change of perspective:

> **Topic 4:** *How can I build a CNN that recognises images?*
> **Topic 5:** *How can I demonstrate that this CNN performs appropriately, understand when it fails, explain its decisions, and decide whether it is safe and responsible to use?*

That distinction looks particularly important for the Week 5 **“Explaining the dog breed classification”** work and Project 3.  
