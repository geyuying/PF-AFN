python -m torch.distributed.launch --nproc_per_node=8 --master_port=7129 train_PBAFN_stage1.py --name PBAFN_stage1   \
--resize_or_crop None --verbose --tf_log --batchSize 4 --num_gpus 8 --label_nc 14 --launcher pytorch










