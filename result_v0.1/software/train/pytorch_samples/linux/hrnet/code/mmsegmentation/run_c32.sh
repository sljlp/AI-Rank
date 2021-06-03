export PYTHONPATH=$PYTHONPATH:`pwd`/mmseg
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NUM_GPUS=8
export NUM_NODES=4
export TOP_VAL=78.0
export NODE_RANK=  #set
export MASTER_ADDR= #set
export MASTER_PORT= #set
python3 -m torch.distributed.launch --nproc_per_node=$NUM_GPUS \
			--nnodes=4 --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
               tools/train.py configs/hrnet/fcn_hr18_512x1024_160k_cityscapes_c32.py \
				--launcher pytorch --checksum 37724b19b6e5d41f9f147936d60b3c29 
