# LLM Usage Log

## I used LLMs throughout the project for a lot of things
* Ideation
* Initial code implementations
* Debugging (especially SLURM and GPU issues)
* Help with the writeup

Here is a summary of all of the interactions. Its writing may have been assisted by GPT 4.5. :)

# Project Interaction Summary

## Repository and Formatting

* **Repository Structure Formatting:** Clarified directory structure for scripts, modules, and experiments.
* **Reward Design for RLHF:** Converted content fully into markdown.

## Reinforcement Learning Reward Design

* **RL Reward Design for LLMs:** Discussion and clarification about reward designs in Reinforcement Learning for Large Language Models.
* **Stable Probability Reward Function:** Suggested datasets suitable for evaluating stable probability rewards, particularly textual reasoning datasets.
* **Reward computation in NLP:** Shared code setup details for NLP reward computation environments.
* **Reward Evaluator Implementation:** Discussed abstract base class implementations for reward computation.
* **Reward Strategies for RREF:** Implemented reward strategies specifically targeting Reduced Row-Echelon Form tasks.
* **Correctness Reward Adjustment:** Provided guidelines for adjusting correctness rewards.

## Training and Evaluation Issues

* **Evaluation shape mismatch fix:** Solved tensor shape mismatches during evaluations.
* **Training Configuration Fix:** Addressed missing evaluation metrics issues in GRPO training configuration.
* **Out of Memory Evaluation:** Debugged and resolved CUDA OOM (Out Of Memory) errors.
* **Model Dimensionality Mismatch Fix:** Diagnosed and resolved dimensionality mismatches in models.
* **Answer Probability Calculation:** Debugged assertion errors related to index bounds.

## Reinforcement Learning Specific Implementations

* **GRPO Context Model Training:** Adapted GRPO implementations for context models in RL.
* **Multi-GPU Training Setup:** Configured setups for multi-GPU RL training.
* **GRPO Matrix Inversion Adaptation:** Tailored GRPO specifically for matrix inversion tasks.

## Matrix Inversion and Linear Algebra

* **RL Curriculum for Matrix Inversion:** Implemented curriculum learning approaches and validation methods.
* **Matrix Inversion RL Evaluation:** Designed new datasets and evaluation frameworks specifically for matrix inversion learning.
* **Generate Matrices and Inverses:** Clarified criteria for matrix generation (including non-invertible matrices).

## Software Debugging and Setup

* **CUDA Index Out-of-Bounds Error:** Explained CUDA index errors and solutions.
* **CUDA OOM Error Fix:** Provided debugging strategies and resource allocation adjustments for CUDA memory errors.
* **FlashAttention Mac Compatibility:** Addressed compatibility issues related to FlashAttention on macOS.

## Slurm and Distributed Training

* **SLURM Error Debugging:** Diagnosed and debugged errors in SLURM job execution.
* **SLURM Distributed Training Issue:** Troubleshot distributed training setups using SLURM.
* **Ray cluster connection fix:** Resolved connectivity issues to Ray clusters in distributed environments.

## Conda and Environment Management

* **Conda activation error fix:** Fixed common Conda activation issues.
* **Conda activation troubleshooting:** Continued troubleshooting Conda environments within Slurm jobs.

## General Queries

* **JSON Data Plotting Guide:** Explained JSON data parsing and plotting.
* **GitHub Authentication Error Fix:** Guided through fixing authentication issues with GitHub repositories.
* **Freeze model layers:** Provided instructions on selectively freezing layers in neural network models.
* **Training Loop Explanation:** Explained modifications required to update specific layers during training.

## Mathematical Clarifications

* **Logit Alignment Explanation:** Provided detailed explanations on logit alignment methods and calculations.
* **Latex Loss Function:** Provided LaTeX formatting for specific loss functions requested.
