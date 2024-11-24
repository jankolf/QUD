export OMP_NUM_THREADS=4
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nnodes=1 --nproc-per-node=2 train_parameterized.py --k $1 --t $2 --dataset $3 --teacher $4 --student $5
ps -ef | grep "train" | grep -v grep | awk '{print "kill -9 "$2}' | sh
