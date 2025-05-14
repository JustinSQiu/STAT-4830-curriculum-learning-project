# Designing Good Rewards for Reinforcement Learning on LLMs

**Team Member(s):** Justin Qiu

## Executive Summary

Recent research such as DeepSeek R1 has demonstrated that reinforcement learning is a viable way to train large language models to achieve state-of-the-art in reasoning tasks without using traditional post-training techniques like supervised finetuning. This is achieved through the careful crafting of rewards that steer the language model towards more desirable behavior, as well as a new optimization objective called GRPO (Group Relative Policy Optimization), which is particularly effective when working with sparse rewards. R1 and various follow-up works have primarily focused on reasoning tasks such as mathematics and coding, where the correctness of an output is relatively easy to verify and rewards are therefore easier to design. However, there exists a large number of domains such as creative reasoning that do not lend themselves to easily crafted reward functions. In this project, I explore alternative reward modeling approaches that can work with more general domains that don't have obvious reward design. Specifically, I implement a perplexity-based reward model that uses a frozen pretrained LLM's output logits as the reward function, and demonstrate results on GSM8K, a math word problem dataset.

## 1. Introduction/Background

The dominant recipe for aligning large language models (LLMs) with human intent is reinforcement learning from human feedback (RLHF). In RLHF a reward model, trained on human-ranked completions, supplies a scalar signal and the policy is optimized with Proximal Policy Optimization (PPO). PPO’s clipped‐ratio surrogate objective keeps the new policy close to the reference model via an adaptive KL penalty, giving much better stability than vanilla policy-gradient methods and enabling many epochs of minibatch updates on the same samples. OpenAI’s InstructGPT demonstrated that a 1.3 B-parameter model fine-tuned with PPO-based RLHF could be preferred to the original 175 B GPT-3 on a wide variety of prompts. Despite its success, PPO incurs significant computational overhead because it must draw rollouts from the evolving policy at every gradient step. Direct Preference Optimization (DPO) reframes the RLHF objective so that the optimal policy can be reached with a single cross-entropy loss on preference pairs, bypassing online roll-outs altogether. By treating the base model as an implicit reward estimator, DPO derives a closed-form best-response and converts the RL problem into supervised contrastive fine-tuning. DPO is used more often than PPO in practice because it removes the need for explicit KL penalties, sampling, or reward‐model updates, making training as cheap and stable as ordinary finetuning.

On the other hand, while PPO and DPO both rely on dense reward estimates, many reasoning tasks—math word problems, program synthesis, theorem proving—yield extremely sparse rewards (often only “correct / incorrect”). Group Relative Policy Optimization (GRPO), introduced in DeepSeek-Math and extended in DeepSeek-R1, tackles this setting by sampling groups of candidate completions for each prompt and computing advantages relative to the best answer in the group. Because the baseline is drawn from the same batch, the variance of the policy gradient shrinks, letting the optimizer learn from one-bit correctness signals without extra reward shaping. GRPO inherits PPO’s clipped objective but reduces memory by sharing value estimates across the group, enabling the first LLMs (R1-Zero and R1) to reach state-of-the-art reasoning scores without any supervised pre-training. The original R1 paper used primarily rules-based binary rewards and formatting rewards; some examples include a binary correctness reward, format rewards for adding tags for reasoning and answer, etc.

## 2. Rewards for Linear Algebra

I began my project by investigating a simpler domain: linear algebra. Specifically, I investigate whether GRPO with carefully crafted rewards can be effectively used to teach small language models how to do matrix inversion and reduction of a matrix to the reduced row echelon form.

**Problem Formulation:** Given \$A\$, find \$A^*\$ such that \$A \cdot A^* = I\$

**Problem Setup:**

This problem domain is easily verifiable and has fairly logical rewards. I use four rewards to shape the model:

* Binary correctness reward: \$I(|A \cdot A^\* - I| < tolerance)\$
* Row by row correctness: \$\sum\_x I(|A^\*\_x - A^{-1}\_x| < tol)\$ where \$A\_x\$ represents a row.
* Continuous correctness: \$2 \cdot \exp(-\frac{||A^{-1} - A^\*||\_{L1}}{tol})\$
* Various formatting rewards such as whether the output is a valid matrix, matrix shape correctness, etc.

However, there were various reward hacking problems throughout the process that forced me to change the reward structure. For instance, I initially had a reward for length. However, the model reward hacked by outputting extremely repetitive outputs and even outputting code to solve the question, which artificially increases the length of the output.

I use GRPO to optimize the language model. I initially used VERL, but ran into many debugging difficulties due to lack of documentation, so I ended up using TRL and Huggingface Transformers as the base framework for finetuning. I use Qwen 1.5B as the base model. For data generation, I wrote synthetic matrix generation scripts that generate random n by n invertible matrices. I will discuss the hardware and implementation details more in the implementation section, along with my implementation details for the main experiment.

I focus my exploration on a subset of the problem, which is inverting 2 by 2 matrices that are a small perturbation away from identity. This is because the general n by n inversion problem turned out to be too difficult for the model to learn, possibly because of the small base model size and compute limitations.

Our model was able to achieve around 90% evaluation accuracy after 1000 training steps.
Insert figure in figures/linalg_plot for me.

![Linear Algebra Results](figures/linalg_plot.png)

## 3. Methodology for General Rewards

The rewards I used for solving linear algebra problems are not adaptable for a general domain. For instance, if we want a language model to be better at reasoning for creative story generation, it is not obvious how to verify the correctness of a given output or provide structured rewards for the LLM to follow. As such, in the next (main) phase of my project, I explored creating rewards that are adaptable for general domains. I focused on four key reward desirata: the reward should be generally applicable, informative, dense, and difficult to reward hack. Some examples of rewards that have been explored in previous literature include embeddings-based rewards and training a reward model (like RLHF).

The reward I focused on exploring is relative perplexity. Language models output logits, which are essentially unnormalized likelihoods of output tokens given the previous input sequence, which encode a lot of valuable information that can be extracted. In particular, perplexity can easily be calculated with the output logits of a model. Perplexity quantifies a model's uncertainty for a given sequence of tokens. A higher perplexity indicates that a model is less certain of a given output, while a lower perplexity means the model is more certain, meaning that we want desirable outputs to have a low perplexity. Perplexity gives us a reward function that is general purpose, informative, dense, and continuous, which is perfect for our purposes. Note that during the middle of my project, a paper by Gurung and Lapata (https://arxiv.org/abs/2503.22828) came out that uses a very similar reward function, so we used their approach as inspiration to refine our own.

**Perplexity Formula:**

$Perplexity(S) = \exp\left(-\frac{1}{N}\sum_{i=1}^{N}\log P(x_i | x_{<i})\right)$

Our initial experiments used an absolute perplexity reward, which directly takes the perplexity of an output as the reward. However, this led to extremely unstable training, as perplexities of different outputs are difficult to compare against each other, and the scale of perplexity (1 to infinity) leads to very unstable rewards for high perplexity outputs. As such, we adopt a hybrid perplexity reward, which integrates both absolute perplexity and relative perplexity, which represents the relative improvement in perplexity for an output compared to a base output. An equation of our reward is shown below. We also clip the relative perplexity component on the left side by 0, as otherwise the perplexity reward can go to negative infinity for outputs that are significantly worse than baseline outputs, leading to unstable training.

**Hybrid Perplexity Reward Formula:**

$Reward = \alpha \cdot \text{Absolute Perplexity} + (1 - \alpha) \cdot \text{Relative Perplexity}$

$\text{Relative Perplexity} = \max (0, \frac{PPL_\pi^G (y \vert x) - PPL_\pi^G (y \vert x, a)}{PPL_\pi^G (y \vert x)})$

$\text{Absolute Perplexity} = \frac{1}{PPL_\pi^G(y \vert x, a)}$

We also adopt a novel framework for reinforcement learning. While most current methods train a model directly with reinforcement learning to do reasoning, output formatted responses, etc., we instead train a reasoning model with GRPO to output reasoning traces that allow another frozen base model to just output the answer. The frozen base model provides both the final answer for our inputs and the logits necessary to calculate the perplexity reward for each update step of GRPO on our reasoning context model.
Our approach is based on the hypothesis that
P_base(Answer | c, Question) is very high for multiple contexts c

Our new reinforcement learning setup has several key advantages compared to a single model approach. First, the base model and the reasoning context model can be different, allowing for potentially more efficient methods to be developed. For instance, it would be interesting to explore varying the size of reasoning context model and the base model, as only the reasoning context model is finetuned. Second, the new setup lends itself well to our perplexity reward. Since our relative perplexity reward requires a baseline output to compare to, giving the question directly to the frozen base model provides us with a reasonable baseline output.

![New Formulation](figures/new_formulation.png)

**GRPO Formula:**

![GRPO Formulation](figures/grpo_formulation.png)

We use GSM8K as our dataset for both training and evaluation. GSM8k is a representative math word problem dataset with problems that involve arithmetic and basic multi-step reasoning. GSM8k does indeed have clear and verifiable rewards that have been extensively explored; however, we choose to test our approach on GSM8k so that we can validate our results on a commonly used dataset.

## 4. Training Details

We do two experiments with this approach: one with large models and the other with small models. For the large models, we used Llama-3.2 8B instruct, and for the small models, we used Qwen2.5 0.5B instruct. We keep the training setup and hyperparameters constant across both experiments. We use a cosine scheduler with a 0.1 warmup ratio and a learning rate of 5e-6. We use an 8-bit verison of the paged adamw optimizer for memory savings. We set beta1=0.9 and beta2=0.99 for the adam optimizer hyperparameters. We use a batch size of 1 batch per GPU (otherwise, our GPU OOMs), and 4 gradient accumulation steps to mitigate gradient instability. For GRPO, we run 6 generations per batch, which is a fairly standard batch size. In order to mitigate overfitting and ensure stable training, we also employ weight decay and gradient clipping while finetuning. When finetuning without gradient clipping, we observe significant degradation in model performance and divergence in our training and evaluation loss; adding regularization prevented this issue without significant hyperparameter tuning. We do not significantly hyperparameter tune our models due to GPU resource constraints and time constraints.

All of our base models were loaded in at 4-bit precision; without 4-bit quantization, we would not be able to load and finetune the 8B models on our available hardware. We also employ LoRA with rank 16 when finetuning in order to further reduce compute and memory requirements by freezing model weights and instead inserting trainable low-rank matrices that approximate weight updates to the whole model. Each model is trained for 1 epoch on the test set of the GSM8K dataset.

In terms of hardware, for the large experiment, we use ~72 GPU hours on a RTX A6000. For the small experiment, we use ~12 GPU hours on a RTX 2080. For software libraries, we primarily use TRL and trainer for training with GRPO. We also experimented with VERL, as mentioned previously, but found it to be too difficult to even set up due to lack of documentation.

## 5. Results and Discussion

For both the large and small model experiments, we train models using our hybrid perplexity approach, as well as only relative perplexity as a comparison. We also train models only using absolute perplexity, which was the original formulation of our reward, but find that the model doesn't learn at all. Both of our models (relative and hybrid) achieve 73% evaluation accuracy after one epoch of training. However, the baseline accuracy of Llama-3.2 8B is 72%. Note that we find that with only a perplexity-based reward and no format rewards, the model is not sufficiently able to learn the correct output format to be able to use rules-based checks on the output correctness, so we use an LLM to evaluate whether the outputs are correct. The plot below shows training and evaluation reward on our large model with hybrid perplexity.

![GSM8K Large Models Results](figures/large_model_plots.png)

For the small model experiment, we do a similar setup and find that training does not improve results from the baseline at all. During the presentation, I reported that the evaluation accuracy went up from 0.11 to 0.15. However, I noticed that this probably was caused by a prompting inconsistency for the evaluator LLM between the baseline and my approach. After normalizing the prompts, the large model experiments had similar results but the small model accuracy dropped down to the baseline of 0.11, indicating that no improvement was made. The plot below shows training reward on our small model with hybrid perplexity.

![GSM8K Small Models Results](figures/small_model_plots.png)

Overall, our results are fairly disappointing on the surface. None of our experiments showed significant improvements over the baseline, and some setups even led to performance degradation. However, some interesting results arise from manually comparing our model's behavior to the baseline. Below are examples of outputs from the models; the first one is from our finetuned model and the second is from the base model.

Finetuned Model:
Question: Claire wants to make 2 cakes for her mother. Two packages of flour are required for making a cake. If 1 package of flour is $3, how much does she pay for the flour that is enough to make 2 cakes?
Generated Context: To find out how much flour Claire needs for 2 cakes, we first need to determine how many packages of flour are required for 2 cakes. Since each cake requires 2 packages of flour and she is making 2 cakes, she needs 2 cakes * 2 packages per cake = 4 packages of flour.
The cost of 1 package of flour is $3, so the cost of 4 packages of flour would be 4 packages * $3 per package = $12.
Therefore the answer is:

Non-finetuned model:
Question: Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?
Generated Context: To solve this problem, we need to follow these steps:
1. First, we need to calculate the total number of eggs laid by the ducks in a day, which is given as 16 eggs.
2. Since Janet eats three eggs for breakfast and bakes four for her friends, she uses a total of 3 + 4 = 7 eggs.
3. To find the number of eggs remaining, we subtract the used eggs from the total number of eggs laid: 16 - 7 = 9 eggs.
4. Since Janet sells the remaining eggs at $2 per egg, we multiply the number of remaining eggs by the selling price to find the total amount she makes: 9 eggs * $2/egg = $18.
The answer is 18.

We notice that our finetuned reasoning context model lowers the perplexity of the base model's output being the exact desired output by ending its output with "Therefore the answer is:". This pattern is consistent for almost all of the outputs from our finetuned model. On the other hand, the model that was not finetuned just outputs the answer directly, which makes it less likely for a base model to output just the desired answer given that output as context. We also find from our rewards that finetuning does lower the perplexity of the desired outputs significantly; however, it is not sufficient enough to improve the evaluation accuracy. It is very possible that training our model with more data and more epochs would help here, but due to compute limitations I wasn't able to try that.

## 6. Parameter Efficient Training with Iterative Layer Freezing

I started this semester with an entirely different project goal: to explore using iterative layer freezing during pretraining to improve parameter efficiency. However, I abandoned the idea after two weeks because it seems that pretraining for the current LLM architectures is mostly a solved problem, and also because I don't have nearly enough compute to run meaningful pretrianing experiments. However, I have also included the code for that experiment in this repo.

## 7. Conclusion and Reflections

Despite the results, I still think the framework is interesting and worth further exploration; the paper from Gurung and Lapata last month that I mentioned earlier found that the framework works for creative story generation but only does human evaluation. One big thing I'd like to try if I continued this project in the future is to use a textual reasoning dataset that might be more suitable for my reward function. Overall, my project explored how to design rewards for general domains and specifically investigated the efficacy of using a perplexity reward to train LLMs with GRPO.

### Reflection Questions:

**What was the number one technical or conceptual difficulty?**
The biggest limitation was definitely compute. Towards the end of the semester, my training scripts spent almost a week in slurm queue every time I submitted them, which forced me to run experiments at a glacial pace. My lab only has 16 A6000s so they were highly in demand.

**What part of the project workflow was easier than expected? Harder?**
**Easier:** working with Unsloth and TRL. They did a great job with making the frameworks easy to use and relatively easy to debug! The same can’t be said for VERL, which was poorly documented in many places and had a lot of unexpected bugs (it is possible that there is better documentation in Chinese, but my reading is not good enough for technical use)
**Harder:** working with the GPUs. Also coming up with new ideas, although Professor Davis helped significantly on this aspect!

**How did your project goals or approach evolve based on self-critiques or intermediate results?**
My project significantly changes throughout the semester. I began the semester exploring rewards in easily verifiable settings like linear algebra problems, but pivoted towards experimenting with flexible and more general-purpose rewards. A large part of this was that Professor Davis had the interesting idea of training a context model to generate reasoning traces.

**How did AI tools assist your project (e.g., coding, debugging, writing, brainstorming)?**
AI helped a lot with initial coding iterations and debugging GPU and SLURM problems. I have detailed more in the llm_exploration section of the archive.

**What was the most surprising result or finding during the project?**
While I shouldn't have been too surprised, as ideas don't always work on the first few tries, I was slightly surprised and disappointed that my formulation didn’t work as well as I expected. I think the biggest problem was compute (common trend!) which prevented me from iterating on ideas quickly.

**Which specific lecture topic, concept, or technique from the course was most useful for your project? Explain how.**
The last few lectures on best practices for training large language models were very helpful, as they helped me better understand the workflow. Even though I wasn’t able to apply it super well for this project due to compute and time limitations, it was still tremendously helpful.

**How has your perspective on applying optimization in practice changed since the beginning of the course?**
I’ve gotten a deeper understanding and appreciation of the math behind the optimization and how optimization makes all of modern deep learning work.

**If you had two more weeks, what specific next step would you take on the project?**
As I mentioned, I would have liked to train on other datasets. Also, I think combining the perplexity reward with format rewards would have been interesting.

**If you could restart the project, what is the single biggest change you would make to your approach or plan?**
By far the biggest thing I would change is **finding a teammate!!** I started the project with the incorrect impression that most people in the course were planning to work on a finance-related project after only asking three other people. This obviously turned out to be hilariously incorrect, as two other teams just in my section explored RL in some way. Working on a team would have alowed the project to be larger in scope and allowed us to come up with new ideas by collaborating!