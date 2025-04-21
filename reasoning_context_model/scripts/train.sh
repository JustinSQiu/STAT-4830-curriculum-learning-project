#!/bin/bash
#SBATCH --job-name=train_context_3
#SBATCH --output=slurm_output/context_3.txt
#SBATCH --partition=p_nlp
#SBATCH --gpus=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=256GB
#SBATCH --constraint=48GBgpu

cd /home1/j/jsq/STAT-4830-curriculum-learning-project/reasoning_context_model/

source /nlp/data/jsq/venv_grpo/bin/activate

module load cuda/11.7

nvidia-smi
nvcc --version
python -c "import torch; print('CUDA Available:', torch.cuda.is_available())"
python -c "import torch; print('PyTorch Version:', torch.__version__)"
python --version

source secrets.env

huggingface-cli login --token "$HUGGINGFACE_TOKEN"
wandb login --relogin "$WANDB_TOKEN"


python ../train.py