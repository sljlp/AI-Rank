export PYTHONPATH=$PYTHONPATH:`pwd`/mmseg
export CUDA_VISIBLE_DEVICES=0
export TOP_VAL=78.0
python3.7 tools/train.py configs/hrnet/fcn_hr18_512x1024_160k_cityscapes.py --checksum 37724b19b6e5d41f9f147936d60b3c29 \
--iters 100 --log_interval 10 --samples_per_gpu 2

