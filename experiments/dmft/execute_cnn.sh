#!/bin/bash
# CIFAR-10 CNN, PC (µPC, ε = 1) vs BP with Adam. These 100-epoch jobs have
# finished. Do not resubmit into results_cifar_adam_e100.
#
# The match is task 4 (seed 1 is task 5): hidden κ ratio 4, pool step ×0.25,
# top κ ×4, 16 downward sweeps, activity step 0.01, Adam lr 0.3, kernel fan-in.
# Epoch 100, seed 0 (log/9027714_4.out): train 89.92%, test 85.31%
# (best test 86.10% at epoch 93). BP reference, task 0 (log/9027714_0.out):
# train 90.16%, test 86.47% (best test 86.85% at epoch 94).
#SBATCH --job-name=cnn_adam
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --partition=short
#SBATCH --clusters=htc
#SBATCH --qos=standard
#SBATCH --account=ndcn-computational-neuroscience
#SBATCH --array=0-7
#SBATCH --output=/data/ndcn-computational-neuroscience/mert5001/jpc/experiments/dmft/log/%A_%a.out

module load Anaconda3
source activate $DATA/envs/PC_dmft_env
cd $DATA/jpc/experiments/dmft

COMMON="--dataset CIFAR10 --arch cnn --width 128 --n_res_blocks 3 --batch_size 64 \
  --param_optim adam --act_fn relu --hidden_energy_reduce spatial_mean \
  --n_epochs 100 --n_mini_per_epoch 1 --n_seeds 1 --keep_npy"
# Base PC string (pool step ×0.5). Later flags override one knob.
PC="--skip_bp --activity_update downward --activity_lr 0.01 \
  --n_infer_iters 16 --hidden_kappa_ratio 4 \
  --pool_activity_lr_mult 0.5 --top_activity_lr_mult 0.05 \
  --top_kappa_mult 4 --adam_kernel_fan_in --pc_delta_activities \
  --param_lr 0.3 --param_lr_pc 0.3"
KFI="--adam_kernel_fan_in --skip_pc --param_lr 0.3"
BASE="--results_dir results_cifar_adam_e100"

case $SLURM_ARRAY_TASK_ID in
  # Backprop reference at the same learning rate and kernel fan-in.
  0) python train_benchmark.py $COMMON $KFI $BASE/bp ;;
  # Ratio 4, the closest train match, and a second seed.
  1) python train_benchmark.py $COMMON $PC $BASE/kr4 ;;
  2) python train_benchmark.py $COMMON $PC $BASE/kr4_seed1 --seed 1 ;;
  # Pool step ×0.25, the closest test match. Ratio stays 2, as in that run.
  3) python train_benchmark.py $COMMON $PC $BASE/pool025 \
       --hidden_kappa_ratio 2 --pool_activity_lr_mult 0.25 ;;
  # Both changes together, and a second seed.
  4) python train_benchmark.py $COMMON $PC $BASE/kr4_pool025 \
       --pool_activity_lr_mult 0.25 ;;
  5) python train_benchmark.py $COMMON $PC $BASE/kr4_pool025_seed1 \
       --pool_activity_lr_mult 0.25 --seed 1 ;;
  # Ratio 4 with a smaller leftover top displacement (top κ ×8).
  6) python train_benchmark.py $COMMON $PC $BASE/kr4_topk8 \
       --top_kappa_mult 8 ;;
  # 32 sweeps only once the pool step is halved. K=32 at pool ×0.5 died.
  7) python train_benchmark.py $COMMON $PC $BASE/kr4_pool025_k32 \
       --pool_activity_lr_mult 0.25 --n_infer_iters 32 ;;
esac
