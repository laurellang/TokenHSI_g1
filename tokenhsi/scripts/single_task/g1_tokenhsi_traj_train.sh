python ./tokenhsi/run.py --task HumanoidTrajSitCarryClimb \
  --cfg_train tokenhsi/data/cfg/train/rlg/amp_imitation_task_transformer_multi_task.yaml \
  --cfg_env tokenhsi/data/cfg/multi_task/amp_g1_dex3_traj_sit_carry_climb.yaml \
  --motion_file tokenhsi/data/dataset_g1_all.yaml \
  --num_envs 4096 \
  --headless
