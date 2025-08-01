

# to train

```bash
# run the finetuning starting from the libero90 checkpoint on the dataset saved
torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/train.py \
  --pretrained_checkpoint models/minivla-libero90-prismatic/checkpoints/step-122500-epoch-55-loss=0.0743.pt \
  --vla.type prism-qwen25-dinosiglip-224px+0_5b+mx-ybq_floor_small \
  --data_root_dir ./dataset_sl \
  --run_root_dir ./runs \
  --image_aug False \
  --wandb_project youbiquo-vla \
  --wandb_entity null \
  --run_id prism-qwen25-dinosiglip-224px+0_5b+mx-ybq_floor_small \
  --save_interval 14836 \
  --is_resume False \
  --debug False

```