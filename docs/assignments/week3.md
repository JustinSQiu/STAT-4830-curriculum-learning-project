### Problem Statement (1/2 page)
- What are you optimizing? (Be specific)
- Why does this problem matter?
- How will you measure success?
- What are your constraints?
- What data do you need?
- What could go wrong?

Deepseek R1 recently introduced a novel way of training LLMs to reason better by using large-scale reinforcement learning on a pretrained model without prior supervised finetuning.

They use GRPO rather than PPO, which is very commonly used, for their reinforcement learning. GRPO (Group Relative Policy Optimization) samples multiple outputs from the current policy at each step and computes the advantage of each output against the reward model compared to the other outputs. Rather than using a neural reward model, which is currently very common, they use a rules-based reward function that takes into account things like mathematical accuracy, logical consistency, language consistency, and format. They optimize the language model with a loss function that takes into account both the advantage gained by adding 

I will explore using more parameter-efficient ways of training language models with RL. Specifically, all parameters of the model are currently updated during the backpropagation in the RL loop. 

Furthermore, I will explore using curriculum learning methods to 

This problem is important because the methods introduced by Deepseek-Math and Deepseek-R1 represent shifts in how we think about post-training, and their empirical results show that their methods can be very successful even with less compute. I will measure success by finding out whether a method that involves less weight updates using the pure RL approach in Deepseek-R1-Zero can work.

GRPO steps
Sample multiple outputs from old policy
Compute advantage of each output against the reward model
Estimate KL divergence
Compute loss and backprop
Avoids sparse reward problem by doing comparison across group from same input
Uses rule-based reward functions rather than neural reward model; uses both accuracy and formatting/consistency rewards


### Technical Approach (1/2 page)
- Mathematical formulation (objective function, constraints)
- Algorithm/approach choice and justification
- PyTorch implementation strategy
- Validation methods
- Resource requirements and constraints

### Initial Results (1/2 page)
- Evidence your implementation works
- Basic performance metrics
- Test case results
- Current limitations
- Resource usage measurements
- Unexpected challenges

### Next Steps (1/2 page)
- Immediate improvements needed
- Technical challenges to address
- Questions you need help with
- Alternative approaches to try
- What you've learned so far
