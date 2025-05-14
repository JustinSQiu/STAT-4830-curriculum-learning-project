# Designing Good Rewards for Reinforcement Learning on LLMs

**Team Members:**

* Justin Qiu (jsq@seas.upenn.edu)

---

## High-Level Summary

**Problem Statement:** ecent research such as DeepSeek R1 has demonstrated that reinforcement learning is a viable way to train large language models to achieve state-of-the-art in reasoning tasks without using traditional post-training techniques like supervised finetuning. This is achieved through the careful crafting of rewards that steer the language model towards more desirable behavior, as well as a new optimization objective called GRPO (Group Relative Policy Optimization), which is particularly effective when working with sparse rewards. R1 and various follow-up works have primarily focused on reasoning tasks such as mathematics and coding, where the correctness of an output is relatively easy to verify and rewards are therefore easier to design. However, there exists a large number of domains such as creative reasoning that do not lend themselves to easily crafted reward functions. It is desirable to craft rewards that can be generally applied to any domain.

**Approach:** In this project, I explore alternative reward modeling approaches that can work with more general domains that don't have obvious reward design. Specifically, I implement a perplexity-based reward model that uses a frozen pretrained LLM's output logits as the reward function. I train a large language model with GRPO using this reward and demonstrate results on GSM8K, a math word problem dataset. 

**Key Findings/Contributions:** Our model performs similarly compared to the baseline for both large model and small model experiments, and does not significantly improve evaluation accuracy on GSM8k. However, it does significantly push down the perplexity of desired outputs compared to the baseline. This suggests that this approach is worth further exploration, perhaps with different data, more types of reward functions, and a lot more compute.

---

## Repository Structure

```
project-root/
├── src/                            # Final scripts and modules
│   ├── curriculum_learning/        # Iterative freezing/curriculum learning experiments
│   ├── linear_algebra/             # Linear algebra experiments
│   │   ├── DeepSeekRL-Extended/    # DeepSeekRL experiments
│   │   └── TinyZero-Efficient/     # TinyZero experiments (primary experiments)
│   └── reasoning_context_model/    # Main experiments (reasoning context model)
│       ├── custom_grpo_trainer.py  # Custom GRPO trainer for evaluation tracking
│       ├── data.py                 # Datasets and data processing
│       ├── extract_answers.py      # LLM-assisted evaluation
│       ├── helpers.py              # Helper functions
│       ├── inference.py            # Inference scripts
│       ├── models.py               # Model definitions and loading
│       ├── plotter.py              # Deprecated plotting tool (for context)
│       ├── rewards.py              # Reward functions for GRPO
│       ├── test.py                 # Debugging/logit extraction
│       ├── train_baseline.py       # Train baseline models (GSM8k)
│       └── train.py                # Main training pipeline
├── docs/                           # Slide deck
├── figures/                        # Figures for documentation
├── report.md                       # Comprehensive final report
├── README.md                       # Summary and setup instructions
└── _archive/                       # Historical files and experiments
```
---

## Setup Instructions

Follow these instructions to set up your environment:

1. **Versioning:** Python 3.10+ is required. You also need a GPU with at least 48GB VRAM for the large training experiments (experiments were done on a RTX A6000). Any GPU should do for the smaller training experiments. Ensure that CUDA is installed; I used CUDA 11.7 so you may run into issues with other versions. 

2. **Virtual Environment:** The steps shown below are for Mac and Linux. The venv setup may be different for other systems.

   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Keys:** Create keys.py and put your HuggingFace token, and OpenAI API key (optional) like so:
   ```python
   openai_api_key = ...
   hf_token = ...
   ```
   also run the following command:
   ```bash
   export WANDB_TOKEN=...
   ```

4. **Training:**: If you are on a slurm cluster, you can simply do as follows from the root folder after modifying the paths in the script to point to the corret locations.
   ```bash
   cd src/reasoning_context_model
   sbatch scripts/train.sh
   ```

   If not, follow the following steps.
   ```bash
   cd src/reasoning_context_model/
   module load cuda/11.7
   python ../train.py --reward_type hybrid --dataset_name gsm8k --model base
   ```

   Note that throughout the setup, you may need to tune the directories and paths a bit to get it to work. All of my experiments were run on the Penn NLP GPU cluster, and I cannot verify that these paths work locally because my computer cannot locally laod in the base models and train them.

5. **Inference:**: If you are on a slurm cluster, you can do as follows. You may need to first modify inference.py to point to the location of your trained model.
   ```bash
   cd src/reasoning_context_model
   sbatch scripts/inference.sh
   ```

   If not, follow the following steps.
   ```bash
   cd src/reasoning_context_model/
   module load cuda/11.7
   python ../inference.py
   ```

The experiments will be automatically logged on your Wandb account; go to https://wandb.ai/ and monitor them there.

## No Demo

Please note that it is difficult for me to provide an executable demo for this project. I can't even load the models in for inference on my 6 year old Macbook Air, let alone train them. It is probably possible to hook this up to Collab, but since Collab is not meant for training models for multiple days, I don't think that is the best thing to do. Also, the GPU clusters I've used have been pretty unstable and very overloaded for the past few days, and I haven't been able to get a GPU to even upload my model to Huggingface. However, with minimal code changes to the paths, this repo shouldn't be too difficult to set up if you have the hardware. Please feel free to contact me at jsq@seas.upenn.edu if there are difficulties setting up.