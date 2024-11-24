export OMP_NUM_THREADS=4
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nnodes=1 --nproc-per-node=2 train_mse_parameterized.py --dataset $1 --teacher $2 --student $3
ps -ef | grep "train" | grep -v grep | awk '{print "kill -9 "$2}' | sh
