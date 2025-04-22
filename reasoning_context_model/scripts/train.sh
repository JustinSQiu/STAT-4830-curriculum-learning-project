#!/bin/bash
#SBATCH --job-name=debug_print_no_thresh
#SBATCH --output=slurm_output/debug_print_no_thresh.txt
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