export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

export TOP_VAL=78.0
export NUM_NODES=$1
export NUM_GPUS=$2
export SAMPLES_PER_EPOCH=2975
export LD_LIBRARY_PATH=/usr/local/lib/python3.6/dist-packages/paddle/libs/:${LD_LIBRARY_PATH}

export FLAGS_cudnn_exhaustive_search=1

export LD_LIBRARY_PATH=/usr/lib64/:/usr/local/lib/:$LD_LIBRARY_PATH

BS=8
LR=0.01
SAVE_IT=2000
LOG_IT=10
ITERS=160000

[ $NUM_NODES = 4 ] && BS=4
[ $NUM_NODES = 4 ] && LR=0.02
[ $NUM_NODES = 4 ] && SAVE_IT=400
[ $NUM_NODES = 4 ] && LOG_IT=5
[ $NUM_NODES = 4 ] && ITERS=80000

python3  -u -m paddle.distributed.launch train.py --num_workers 8 --batch_size $BS \
  --config benchmark/hrnet.yml \
  --iters $ITERS --log_iters $LOG_IT  --do_eval \
  --learning_rate $LR \
  --save_interval $SAVE_IT \
  --keep_checkpoint_max 10

