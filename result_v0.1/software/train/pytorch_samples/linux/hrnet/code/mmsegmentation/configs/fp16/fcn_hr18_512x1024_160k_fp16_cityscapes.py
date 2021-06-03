_base_ = '../hrnet/fcn_hr18_512x1024_160k_cityscapes.py'
optimizer_config = dict(type='Fp16OptimizerHook', loss_scale=512.)
