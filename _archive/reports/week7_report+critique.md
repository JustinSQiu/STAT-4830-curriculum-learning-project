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

* Significant improvement observed, though experiments are still ongoing.

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

# Self Critique (formatted it from previous notes)

Ideas/TODO
if you give it 3 by 3 perturbation of identity

try it on arbitrary 2 by 2

symbolic matrices with variables instead of numbers

code continous reward: embedding

https://x.com/wzhao_nlp/status/1896962009918525730

test how much generalization I have

how to know what the rules I’m discovering

Don’t tell it to invert a matrix

Give it basic operations on matrix

Number of steps for applying these rules


Haddamar matrix: one way to convert matrix to dimension 2^k for every single k

Conjecture: In general there’s Haddamar matrices in 4*k for every k

5 different rules to get new Haddamar matrix over old one

loop the model for each step

first, test generalization by giving it other types of matrices

give it basic operations on how to invert a matrix

ie flip two rows, 

1. Test generalization
2. Given basic operations, invert a matrix
    1. To tell whether it actually understands the rules it’s discovering

### OBSERVE

* Initial reward designs suffered significantly from reward hacking and unintended model behaviors.
* Curriculum learning demonstrated promising initial results but lacks systematic evaluation and robustness checks.
* Previous experiments with layer freezing showed inconsistent and unclear performance outcomes.

### ORIENT

#### Strengths

* Successfully identified a clear progression path (curriculum learning) that notably improved preliminary results.
* Thoughtfully refined reward structures by transitioning to continuous, norm-based correctness.
* Practical experimentation established a useful baseline and revealed key challenges.

#### Areas for Improvement

* Current reward functions still inadequately prevent reward hacking, particularly for complex matrices.
* Generalization testing and robustness evaluation are currently weak or missing.
* Experiments on parameter efficiency (layer freezing, LoRA) lack systematic comparison and clear conclusions.

#### Critical Risks/Assumptions

Need to validate the assumption that the refined reward function fully captures necessary mathematical accuracy needs validation.

### DECIDE

#### Concrete Next Actions

* Explicitly test and refine reward function robustness against reward hacking through adversarial matrix inputs.
* Implement structured generalization tests (e.g., symbolic matrices, arbitrary 2x2 and 3x3 perturbations).
* Conduct systematic, documented experiments comparing various layer freezing and parameter-efficient methods, clearly logging outcomes using WandB.

### ACT

#### Resource Needs

N/a rn besides hopefully the GPU cluster becomes more available.