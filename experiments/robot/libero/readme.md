
```bash
python3 experiments/robot/libero/regenerate_ybq_dataset_gripper_correction.py \
	--ybq_task_suite ybq_floor \
	--libero_raw_data_dir ~/datasets \
	--libero_target_dir ~/regenerated_datasets \
	--debug
	

python3 experiments/robot/libero/regenerate_ybq_dataset.py \
	--ybq_task_suite ybq_floor \
	--libero_raw_data_dir ~/datasets \
	--libero_target_dir ~/regenerated_datasets \
	--debug
```