#!/bin/bash

#SBATCH --job-name=llama2-fine-tune
#SBATCH --nodes=1
#SBATCH --partition=g2gpu8
#SBATCH --time=1:10:00

export CONDA_BASE=/opt/conda
source $CONDA_BASE/bin/activate base
conda activate llama2
cd $SLURM_SUBMIT_DIR
torchrun --nnodes 1 --nproc_per_node 8  fine-tune.py --enable_fsdp --lr 1e-5  --num_epochs 3 --batch_size_training 2 --model_name unsloth/Llama-3.2-11B-Vision-Instruct --dist_checkpoint_root_folder ./finetuned_model --dist_checkpoint_folder fine-tuned  --use_fast_kernels --dataset "custom_dataset" --custom_dataset.test_split "test" --custom_dataset.file "./ocrvqa_dataset.py"  --run_validation True --batching_strategy padding
