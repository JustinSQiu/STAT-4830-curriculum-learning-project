# Deepseek R1 and Parameter-Efficient RL Training for LLMs

## Overview

Deepseek R1 [DeepSeek-AI et al., 2025] recently introduced a novel way of training LLMs to reason better by using large-scale reinforcement learning on a pretrained model **without prior supervised finetuning**. They use **GRPO (Group Relative Policy Optimization)** to perform RL on their language model. GRPO samples multiple outputs from the current policy at each step and computes the advantage of each output against the reward model compared to the other outputs. Rather than using a neural reward model (which is common in current research), they use a rules-based reward function that considers factors such as mathematical accuracy, logical consistency, language consistency, and formatting. The language model is optimized using a loss function that accounts for both the advantage gained for each output from the updated policy and the KL divergence against a reference policy, preventing the model from deviating too far. GRPO might help improve performance with a sparse reward model over PPO.

## Project Goals

For my project, I will explore **parameter-efficient ways** to train language models using reinforcement learning. Typically, current methods update all parameters of the model during the RL backpropagation loop. I want to investigate if similar performance can be achieved by updating only a subset of the parameters. Some ideas include:

1. **Updating Only the First or Last Few Layers**  
   This idea has been explored with supervised finetuning methods. Researchers have found that performance can be maintained—or even improved—with far fewer weight updates than full finetuning [Lee et al., 2023, 2019].

2. **Selective Layer Updates During Training**  
   Selecting specific layers to update during training has been examined in the context of Supervised Finetuning (SFT) [Ardakani et al., 2024; Liu et al., 2021].

3. **Optimization Tricks to Approximate Weight Updates**  
   Inspired by LoRA [Hu et al., 2021] and its variants, this approach has been recently applied to RLHF by Sidahmed et al. [2024]. Their findings suggest that adapting LoRA both for training the reward model and updating the model's weights can achieve performance comparable to regular RLHF.

I plan to focus on **math tasks** because:
- Math tasks are relatively easy to work with.
- Data is readily available or can be synthetically generated.
- DeepSeek-R1 was designed for reasoning.
- The reward model can be straightforward since math problems generally have well-defined answers.

This problem is significant because the methods introduced by Deepseek-Math [Shao et al., 2024] and Deepseek-R1(-Zero) represent a shift in how we approach post-training. Their empirical results demonstrate that these methods can be very successful even with lower compute requirements. I will measure success by determining if a method that involves fewer weight updates (using the pure RL approach from Deepseek-R1-Zero) is effective.

The main constraint for my project is **compute**. Finetuning even a small pretrained model can be computationally expensive, and my resources are limited. Additionally, working with the numerous recently released libraries—and attempting to replicate or extend their methods—is challenging. For example, I spent five hours trying to get [TinyZero](https://github.com/Jiayi-Pan/TinyZero) to train on a GPU cluster without success. I will likely need to meet with my professor for debugging advice.

In terms of data, one idea is to use a **curriculum learning** approach to train the model by progressively introducing more challenging math problems. Potential data sources include:
- The ACL workshop on curriculum learning [Warstadt et al., 2023]
- Recent reasoning datasets like [OpenThoughts](https://huggingface.co/datasets/open-thoughts/OpenThoughts-114k)

## Results and Next Steps

I have gotten access to 2 RTX A6000 GPUs and have finally been able to run training! I replicated https://github.com/JustinSQiu/DeepSeekRL-Extended-Efficient/ and found similar results after finetuning Qwen 2.5 1.5B with GRPO as the repo author did. I also ran some simple experiments with freezing the first and last layer of the model. The results are disappointing so far, but I've only done very preliminary experiments. Will further discuss results in meeting on Tuesday!

I also got the TinyZero code working (which took a lot of debugging VERL). However, because of GPU memory limitations I made the batch size a lot smaller and that resulted in a very poor performance, so I think I will use the implementation above.

All the code is in the two submodules.