## Reinforcement Learning for LLMs

## Building General Rewards

* Creative writing, unstructured reasoning lack obvious reward functions.
* Desired reward properties:

  * General applicability
  * Informative
  * Dense (sparse rewards slow training)
  * Hard to reward hack

## Perplexity Reward

* Perplexity quantifies model uncertainty.
* Higher perplexity → less certainty; low perplexity for correct answers.
* Continuous reward for text generation.
* Issue: High perplexity even for correct answers.

## Challenges and Planned Solutions

* Current reward: Absolute raw log likelihood only.
* Issue: Absolute perplexity is unstable.
* Planned improvements:

  * Relative perplexity reward (Gurung and Lapata, 2025)
  * Thresholding to stabilize reward
  * Hybrid reward (relative + absolute perplexity)

## New Formulation Idea

* Hypothesis: for most contexts, P_base (Answer | Context, Question) is high
Thus train a model q to generate (Context | Question)
Using reward function r = -log(p_base (Answer | Context, Question))
Where answer is the desired answer

Thus we can
  * Fine-tune a model to generate reasoning traces.
  * Feed traces as context to base model.

## Dataset

* GSM8k (math problems)
* Planned testing on textual reasoning datasets (e.g., HellaSwag)

## Next Steps (Planned)

* Improve reward formulations
* Extensive testing on multiple datasets
* Experiment with smaller base model and larger context model

# Self Critique (formatted it from previous notes)

### OBSERVE

* The current reward function based on absolute perplexity is notably unstable and ineffective without external help.
* Initial results indicate a clear dependency on external context or assistance for the GSM8k dataset.

### ORIENT

#### Strengths

* The proposed log-likelihood (perplexity-based) reward formulation is innovative and distinct from existing literature.
* Project is set up with sufficient resources and data to support experimentation.

#### Areas for Improvement

* The log likelihood loss is very unstable, maybe switch to a different approach
* Dataset selection and experimentation currently lack breadth beyond GSM8k and preliminary attempts at HellaSwag.

#### Critical Risks/Assumptions

The whole idea is a risk, as I'm not sure the new formulation will work well. It seems that a reasonably strong reasoning context model would just output the answer, making our approach at best the same as a traditional RL approch.

### DECIDE

#### Concrete Next Actions

* Iplement and compare relative perplexity and hybrid reward formulations against current absolute perplexity reward.
* Expand experiments systematically across GSM8k, HellaSwag, and other textual reasoning datasets to validate general applicability.
* Introduce scheduled perplexity thresholds or dynamic reward scheduling to test potential improvements in stability.

### ACT

#### Resource Needs

N/a at moment

TODO:
Experiments:
Gsm8k and Hellaswag
Help vs no help
Absolute perplexity vs relative perplexity vs both
Add scheduled perplexity?
Interesting: gsm8k absolute perplexity needed help; otherwise it wouldn’t work

The main self-critique in my project right now is obviously that I need to get a working implementation that works well.


# Previous Work Full:
# Teaching LLMs Linear Algebra with Reinforcement Learning

**Justin Qiu**

---

## Background

Previous approaches to fine-tuning LLMs include:

* **Supervised Fine-Tuning (SFT)**: Utilizing quality-labeled data to align the model via instruction tuning.
* **Reinforcement Learning with Human Feedback (RLHF)**: Employing reinforcement learning guided by neural reward models to produce human-preferred outputs.

**Novel Proposal**: Employ pure Reinforcement Learning (RL) with rules-based rewards to teach LLMs reasoning.

---

## Application to Linear Algebra

### Approach

* **Iteratively generate synthetic data**
* **Craft rewards carefully**
* **Implement curriculum learning** (easy to hard progression)

### Rationale for Choosing Linear Algebra

* Easy generation and verification of synthetic data
* Opportunity for continuous self-improvement
* Practical usefulness

---

## Preliminary Findings

### Long Sequence Operations

* The model already handles long sequence addition and multiplication well without training.

### Challenges with Complex Tasks

* Meta AI's 2022 research indicated transformers struggle with complex linear algebra tasks like matrix inversion or Singular Value Decomposition (SVD).
* Difficulty stems from LLMs' pre-training on language modeling rather than mathematical computation.

---

## Initial Attempts and Challenges

### Attempt 1: Naive Rewards Approach

* Direct correctness and formatting rewards.
* Specific focus on matrix inversion:

  * Input: Matrix A
  * Output: A⁻¹
* Issues encountered:

  * Reward hacking due to poor reward distribution (RREF)
  * Reward hacking by length manipulation
  * Outputs either too lengthy or nonsensical (e.g., in Chinese)

---

## Refined Rewards Strategy

### Improved Reward Structure

* **Binary correctness**:

  * Exact inversion validation
  * Verification via multiplication (A \* output ≈ I)
* **Row-wise correctness** (limited effectiveness due to global sensitivity)
* **Continuous correctness**:

  * Using error norms:
    $\text{inv\_error} = ||\text{pred\_inverse} - \text{true\_inverse}||_1$
  * Reward function:
    $\text{reward} = 2.0 \times e^{-\frac{\text{inv\_error}}{\text{tolerance}}}$

### Format-Based Rewards

* Proper usage of specific tags (`<answer>`, `<reason>`, etc.)
* Structured outputs (Python list of lists, no extraneous text)

#### Outcome

* Slightly improved performance but still faced significant challenges.

---

## Curriculum Learning Approach

* Tackling complexity incrementally:

  1. **Trivial matrices (1x1, diagonal)**: Easily learned.
  2. **Near-identity 2x2 matrices**: Gradual increase in complexity.
  3. **Random 2x2 matrices**
  4. **Random n x n matrices**: Increasing size and complexity progressively.

### Early Results

* Significant improvement observed, though experiments still ongoing.

---

## Next Steps

* Complete pending experiments with small perturbations on identity matrices.
* Develop a robust curriculum learning pipeline for systematically increasing task difficulty.
* Explore continuous learning by training on generated outputs.
* Revisit layer freezing strategies and set up thorough experiment tracking using WandB to avoid data loss.

---

## Previous Results Summarized

### Objective

* Explore efficient RL training by partially updating model parameters:

  * LoRA (Low-Rank Adaptation)
  * Layer freezing strategies (initial, final, selective layers)

### Preliminary Results from Layer Freezing (Previous Work)

* Varied results depending on the specific layers frozen during training:

  * Top left: Frozen except first five layers
  * Top right: Frozen except last five layers
  * Bottom left: Frozen except last layer
  * Bottom right: No layers frozen (baseline)

---

