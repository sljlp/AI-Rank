export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

export TOP_VAL=78.5
export NUM_NODES=$1
export NUM_GPUS=8
export SAMPLES_PER_EPOCH=2975
export LD_LIBRARY_PATH=/usr/local/lib/python3.6/dist-packages/paddle/libs/:${LD_LIBRARY_PATH}

export FLAGS_conv_workspace_size_limit=2000 #MB
export FLAGS_cudnn_exhaustive_search=1
export FLAGS_cudnn_batchnorm_spatial_persistent=1

export LD_LIBRARY_PATH=/usr/lib64/:/usr/local/lib/:$LD_LIBRARY_PATH
[ $1 = 1 ] || [ $1 = 4 ] || exit
[ $1 = 1 ] && BS=4 && ITERS=80000 && LOG_ITR=2000 && LR=0.01 && SAVE_ITR=2000
[ $1 = 4 ] && BS=2 && ITERS=40000 &&  LOG_ITR=5    && LR=0.04 && SAVE_ITR=200

python3  -u -m paddle.distributed.launch train.py --num_workers 8 --batch_size $BS \
  --config benchmark/deeplabv3p.yml \
  --iters $ITERS --log_iters $LOG_ITR --fp16 --do_eval \
  --learning_rate $LR \
  --save_interval $SAVE_ITR \
  --keep_checkpoint_max 10

