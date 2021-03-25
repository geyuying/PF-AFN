python -m torch.distributed.launch --nproc_per_node=8 --master_port=7129 train_PFAFN_e2e.py --name PFAFN_e2e   \
--PFAFN_warp_checkpoint 'checkpoints/PFAFN_stage1/PFAFN_warp_epoch_201.pth'  \
--PBAFN_warp_checkpoint 'checkpoints/PBAFN_e2e/PBAFN_warp_epoch_101.pth' --PBAFN_gen_checkpoint 'checkpoints/PBAFN_e2e/PBAFN_gen_epoch_101.pth'  \
--resize_or_crop None --verbose --tf_log --batchSize 4 --num_gpus 8 --label_nc 14 --launcher pytorch










