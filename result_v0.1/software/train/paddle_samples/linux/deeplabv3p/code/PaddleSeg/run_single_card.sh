export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

export TOP_VAL=78.5
export NUM_NODES=1
export NUM_GPUS=1
export SAMPLES_PER_EPOCH=2975
export LD_LIBRARY_PATH=/usr/local/lib/python3.6/dist-packages/paddle/libs/:${LD_LIBRARY_PATH}

export FLAGS_conv_workspace_size_limit=2000 #MB
export FLAGS_cudnn_exhaustive_search=1
export FLAGS_cudnn_batchnorm_spatial_persistent=1

export LD_LIBRARY_PATH=/usr/lib64/:/usr/local/lib/:$LD_LIBRARY_PATH
BS=4 && ITERS=100 && LOG_ITR=10 && LR=0.01 && SAVE_ITR=2000

python3 train.py --num_workers 8 --batch_size $BS \
  --config benchmark/deeplabv3p.yml \
  --iters $ITERS --log_iters $LOG_ITR --fp16 \
  --learning_rate $LR \
  --save_interval $SAVE_ITR \
  --keep_checkpoint_max 10

