
export CUDA_VISIBLE_DEVICES=0

export TOP_VAL=78.0
export NUM_NODES=1
export NUM_GPUS=1
export SAMPLES_PER_EPOCH=2975
export LD_LIBRARY_PATH=/usr/local/lib/python3.6/dist-packages/paddle/libs/:${LD_LIBRARY_PATH}

export FLAGS_cudnn_exhaustive_search=1

export LD_LIBRARY_PATH=/usr/lib64/:/usr/local/lib/:$LD_LIBRARY_PATH

python3  -u train.py --num_workers 8 --batch_size 8 \
  --config benchmark/hrnet.yml \
  --iters 100 --log_iters 10 \
  --learning_rate 0.01 \
  --save_interval 2000 \
  --keep_checkpoint_max 10

