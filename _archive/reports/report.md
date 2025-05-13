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

Apr 22, 2025
Presentation Outline
Fundamental question: how to design rewards for reinforcement learning in LLMs?
Overview of past experiments (layer freezing w/ curriculum learning, learning matrix inversions)
See past slides
Maybe skip this
Started with hard verifiers, domain of linalg allows for that
2exp(normL1(inv-inv*)tol)
Explanation of GRPO and training with verifiable rewards
Will be quick, seems like everyone’s gonna explain it
How to create general rewards?
Hypothesis: explain the prompt hypothesis
Train a prompt model then use a base model to output answer
Log likelihood/perplexity methods
Explain the formulation
How to calculate? Get logits of only answer tokens, take negative log likelihood, normalize by output length

How to transform from [-inf, 1] to a stable reward?
Lower bound at -1
Thresholding
Also using absolute rewards
Reward=max(0, PPLG(y|x) - PPLG(y|x, a)PPLG(y|x)) + 1PPLG(y|x, a)2
Why this works, why do we want this?
Typical rewards: discontinuous and rules-based; ie 2 + 2 = 4 is easy to validate
BUT: for things like code, natural language, etc. there isn’t an easy way to implement rules-based rewards
Thus this method can be used to generate continuous reward while leveraging 
Training the model
Dataset - gsm8k, also hellaswag
Explain what the dataset is
Hyperparameters and optimization choices
Using parameter efficient methods to fit into GPU
Used Lora to learn fewer parameters by attaching low rank matrices instead of updating all weights
Used 4-bit quantized models; otherwise the 7B model will OOM my GPU
Other stuff: Learning rate, adam, weight decay, warmup, cosine lr scheduler, etc.
Implementation details
Results and evaluation
Show plots
Show some generation examples
Reflection, learned lessons/roadblocks
Stuff about working with GPUs
Reward hacking stuff
Wrestling with the libraries, a couple things that aren’t well document but common problems
TRL doesn’t support custom eval_function? Need to override yourself
Problems with unsloth using multiple gpus


Experiments:
Gsm8k and Hellaswag
Help vs no help
Absolute perplexity vs relative perplexity vs both
Add scheduled perplexity?
Interesting: gsm8k absolute perplexity needed help; otherwise it wouldn’t work
Apr 21, 2025

Just revamping the codebase
Introduced new reward https://arxiv.org/pdf/2503.22828

Mar 25, 2025

Linear algebra tool use
Tell it 3 basic matrix operations:
Multiple row with nonzero constant
Add constant * row to another row
Swap two rows
Iteratively query it
Can it do matrix RREF?

New idea:
Assumption: for most contexts, P_base (Answer | Context, Question) is high
Thus train a model q to generate (Context | Question)
Using reward function r = -log(p_base (Answer | Context, Question))
Where answer is the desired answer

STAT 4830 Stuff

if you give it 3 by 3 perturbation of identity
try it on arbitrary 2 by 2
symbolic 
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


Presentation
Intro/overview
Motivation: base LLMs are not very good at linear algebra. If you try to ask it to some basic operation like inverting a 2 by 2 matrix, it won’t be able to do that unless it has tool use with a scientific calculator or coding, or it’s a big model specifically trained for reasoning
Preface by saying that a lot of the experiments I’m showing are incomplete, sorry about that, I changed my topic several times so was scrambling this week to get at least some results. So sorry in advance for the incomplete results.
Reinforcement learning w/ LLMs
Deepseek explanation: few months ago R1 was able to use RL with purely rules-based rewards to train a reasoning model that was state of the art
Explanation of traditional finetuning: usually SFT or RLHF
Apply this paradigm to teaching small LLMs things difficult tasks that they can’t do initially by providing them with the correct rewards and designing a curriculum to learn difficult multi-step or difficulty ramping up tasks
Linear algebra tasks: easy to synthetically generate data -> continuous self improvement, easy to verify, useful, cannot be done currently (will talk about later)
Model fails at many linear algebra tasks
Show results from paper a while ago like inverse and RREF
Can we do better?
Attempt 0: naive, fails for both RREF and inversion
Explain reward
RREF reward hacking photo
Just gives up because task is too hard
Gave it a length reward and it spits out random Java code
Problem: limited GPU memory
Only thing it learned was going longer, everything else was a fail
Attempt 1: designing reward, reward hacking, sparse rewards
Explain rewards that I tried for matrix inversion
Binary correctness
Continuous correctness
Format
Integers
Still not good enough, show chart
Attempt 2: curriculum learning, start with easier task
Sherman-Morrison-style
Curriculum: explain curriculum
Start with 1 by 1 and diagonal matrices, this obviously was very easy
Start with 2 by 2 perturbations from identity
Random 2 by 2 matrices
Random n by n matrices
Layer freezing experiments
Initial goal of project but was changed, will present some interesting results.

Mar 2, 2025
linked length generation but for linear algebra
can do some linear algebra concept in a curriculum way like matrix inverse 3 by 3 to 4 by 4 etccurriculum learning by making it harder and harder
no need to benchmark it, just show that we can learn it with this simple method
experiment: gpt-4 can actually do matrix inversions—possibly tool use?

ideas: curriculum learning with matrix algebra
- implement the reward

Interesting: reward hacking b/c it always guesses the identity matrix.

Sometimes, it just gives up…

Unrelated Java code…

I added a length reward and it immediately started spewing nonsense like Javascript code and running in a loop
Problem: not enough GPU memory lol
The only thing it learns is thinking for longer



A lot of Chinese output

Feb 22, 2025
continuous self-improvement

program synthesis: given inputs and outputs, give function
then check by running in python
one variable, generate power functions

have func that goes from ints to ints
I want function to have f(0) = 1, f(1) = 2, etc
just see if it can learn this simple example

ndea, did keras, working on program synthesis

compilation reward: 0.1
answer is correct
error could be like mean squared error
how to design the right reward function
learn 2^k function


TODO
Update slides
Try program synthesis with DeepSeek
 
2/4 meeting notes
Write report on md not latex
Try new approach? Problem with GPU right now

aws
lambda labs
hyperbolic
sfcompute

use pytorch stpa
scaled dot product attention
try to code it up myself

use a100 is enough 
will brown twitter
onsloth
custom trident kernels
daniel han 

khush

Parameter Efficient Reinforcement Learning for LLMs

Using methods similar to Deepseek but only tuning certain layers, see performance
Basically steps:
Get pretrained model
Implement RL similar to Deepseek
Train model on curriculum learning dataset with progressive layer freezing
Maybe I can do math? So can use synthetic data


GPU Commands
Switch to GPU
srun --nodelist=nlpgpu01 --pty --gpus=1 --mem=64GB zsh

Create Conda Environment
conda create -n “name”
Delete Conda Environment
conda env remove -n name
Clone Conda Environment
conda create --name name --clone name
Activate Conda environment
Conda activate name

Check currently running jobs 
squeue

Run job
Sbatch job.sh

Check resources for each node 
sinfo

Check RAM 
free -h

Check CPU and running processes 
top (or htop if installed)

Check GPU usage 
nvidia-smi

Might need to 
source ~/.bashrc
After starting srun before doing conda activate env
1/27 meeting notes
doing pretraining outside a lab is hard

don’t do pretraining; we have good enough models for that
get minimal demo to do RL faster with model

Currently what people are doing basic RL
Experiments are still very expensive
How to train an RL model

Deepseek R1 0: base model they started with

sample from data
for each rollout look at reward rollout
weight gradient of log of language model by that reward

take tinyzero, get it running on my computer, but freezing some layers in the computation

maybe use some synthetic data

how to extract the info from the language model by adding more context 

make open source models easier to modify so they reason better

how to steer a language model to do what you want
use a smaller language model

finetune by training only one layer -> see if works very well


Previous idea
Training model while focusing on specific layers, and giving it real curriculum for preschool to college level
Ie first grade level has a high learning rate for first encoder layer but lower for subsequent layers, second grade has high for second and lower for others, like a normal distribution maybe

Curriculum learning
https://aclanthology.org/volumes/2023.conll-babylm/ 
https://arxiv.org/abs/2309.05463 


Freezing layers during finetuning
Surgical finetuning: https://arxiv.org/abs/2210.11466
https://arxiv.org/pdf/1911.03090 
https://aclanthology.org/2024.naacl-long.345/

Layer Freezing & Data Sieving: Missing Pieces of a Generic Framework for Sparse Training: https://arxiv.org/abs/2209.11204

Accelerating Training of Transformer-Based Language Models with Progressive Layer Dropping: https://arxiv.org/abs/2010.13369

AutoFreeze: Automatically Freezing Model Blocks to Accelerate Fine-tuning: https://arxiv.org/abs/2102.01386



Greedy layer-wise training: https://proceedings.neurips.cc/paper_files/paper/2006/file/5da713a690c067105aeb2fae32403405-Paper.pdf
Freezeout: https://arxiv.org/abs/1706.04983

Learning rate decay

Curriculum learning: training on examples of increasing difficulty
Freezing layers during training improves efficiency and possibly helps generalize better (see surgical finetuning paper)


