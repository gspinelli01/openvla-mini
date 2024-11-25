"""
run_libero_eval.py

Runs a model in a LIBERO simulation environment.

Usage:
    # OpenVLA:
    # IMPORTANT: Set `center_crop=True` if model is fine-tuned with augmentations
    python experiments/robot/libero/run_libero_eval.py \
        --model_family openvla \
        --pretrained_checkpoint <CHECKPOINT_PATH> \
        --task_suite_name [ libero_spatial | libero_object | libero_goal | libero_10 | libero_90 ] \
        --center_crop [ True | False ] \
        --run_id_note <OPTIONAL TAG TO INSERT INTO RUN ID FOR LOGGING> \
        --use_wandb [ True | False ] \
        --wandb_project <PROJECT> \
        --wandb_entity <ENTITY>
"""

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import draccus
import numpy as np
import tqdm
from libero.libero import benchmark
from functools import partial
from libero.libero.envs import SubprocVectorEnv
from PIL import Image

import wandb

# Append current directory so that interpreter can find experiments.robot
sys.path.append("../..")
from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    quat2axisangle,
    save_rollout_video,
)
from experiments.robot.openvla_utils import get_processor
from experiments.robot.robot_utils import (
    DATE_TIME,
    get_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)


@dataclass
class GenerateConfig:
    # fmt: off

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"                    # Model family
    hf_token: str = Path(".hf_token")                       # Model family
    pretrained_checkpoint: Union[str, Path] = ""     # Pretrained checkpoint path
    base_vlm: Optional[str] = None
    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization

    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = "libero_spatial"          # Task suite.
    #                                       Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    num_steps_wait: int = 10                         # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 50                    # Number of rollouts per task
    num_envs: int = 10                               # Number of envs to parallelize to.

    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None                # Extra note to add in run ID for logging
    local_log_dir: str = "./experiments/logs"        # Local directory for eval logs
    prefix: str = ''

    use_wandb: bool = False                          # Whether to also log results in Weights & Biases
    wandb_project: str = "prismatic"        # Name of W&B project to log to (use default!)
    wandb_entity: Optional[str] = None          # Name of entity to log under

    seed: int = 7                                    # Random Seed (for reproducibility)

    # fmt: on


@draccus.wrap()
def eval_libero(cfg: GenerateConfig) -> None:
    assert cfg.pretrained_checkpoint is not None, "cfg.pretrained_checkpoint must not be None!"
    if "image_aug" in cfg.pretrained_checkpoint:
        assert cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"
    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"

    # Set random seed
    set_seed_everywhere(cfg.seed)

    # [OpenVLA] Set action un-normalization key
    cfg.unnorm_key = cfg.task_suite_name

    # Load model
    model = get_model(cfg)

    # [OpenVLA] Check that the model contains the action un-normalization key
    if cfg.model_family in ["openvla", "prismatic"]:
        # In some cases, the key must be manually modified (e.g. after training on a modified version of the dataset
        # with the suffix "_no_noops" in the dataset name)
        if cfg.unnorm_key not in model.norm_stats and f"{cfg.unnorm_key}_no_noops" in model.norm_stats:
            cfg.unnorm_key = f"{cfg.unnorm_key}_no_noops"
        assert cfg.unnorm_key in model.norm_stats, f"Action un-norm key {cfg.unnorm_key} not found in VLA `norm_stats`!"

    # [OpenVLA] Get Hugging Face processor
    processor = None
    if cfg.model_family == "openvla":
        processor = get_processor(cfg)

    # Initialize local logging
    run_id = f"{cfg.prefix}EVAL-{cfg.task_suite_name}-{cfg.model_family}-{DATE_TIME}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    print(f"Logging to local log file: {local_log_filepath}")

    # Initialize Weights & Biases logging as well
    if cfg.use_wandb:
        wandb.init(
            entity=cfg.wandb_entity,
            project=cfg.wandb_project,
            name=run_id,
        )

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    print(f"Task suite: {cfg.task_suite_name}")
    log_file.write(f"Task suite: {cfg.task_suite_name}\n")

    # Get expected image dimensions
    resize_size = get_image_resize_size(cfg)

    # Start evaluation
    total_episodes, total_successes = 0, 0

    if cfg.task_suite_name == "libero_spatial":
        max_steps = 220  # longest training demo has 193 steps
    elif cfg.task_suite_name == "libero_object":
        max_steps = 280  # longest training demo has 254 steps
    elif cfg.task_suite_name == "libero_goal":
        max_steps = 300  # longest training demo has 270 steps
    elif cfg.task_suite_name == "libero_10":
        max_steps = 520  # longest training demo has 505 steps
    elif cfg.task_suite_name == "libero_90":
        max_steps = 400  # longest training demo has 373 steps

    env_generators = []
    initial_states_list = []
    task_id_list = []
    task_list = []

    def env_generator(task):
        env, task_description = get_libero_env(task, cfg.model_family, resolution=256)
        return env

    for task_id in range(num_tasks_in_suite):
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env_generators.append(partial(env_generator, task))
        initial_states_list.append(initial_states)
        task_id_list.append(task_id)
        task_list.append(task)

    num_task_chunks = num_tasks_in_suite // cfg.num_envs + 1 if num_tasks_in_suite % cfg.num_envs != 0 else num_tasks_in_suite // cfg.num_envs
    for start_task_chunk_idx in tqdm.tqdm(range(num_task_chunks)):
        # Get a chunk of cfg.num_envs env generators to pass to SubprocVectorEnv
        start_id, end_id = start_task_chunk_idx * cfg.num_envs, (start_task_chunk_idx + 1) * cfg.num_envs
        env_generator_chunk = env_generators[start_id : end_id]
        initial_states_chunk = initial_states_list[start_id : end_id]
        task_id_chunk = task_id_list[start_id : end_id]
        task_chunk = task_list[start_id : end_id]
        instruction_chunk = [task.language for task in task_chunk]

        print(f"Starting SubprocVectorEnv with {len(task_id_chunk)} parallel envs with tasks:")
        for task_id, task in zip(task_id_chunk, task_chunk):
            print(task_id, task.language)
        env = SubprocVectorEnv(env_generator_chunk)

        env_ep_idxs = np.zeros([cfg.num_envs], dtype=int)          # Count which episode the env is on
        env_steps = np.zeros([cfg.num_envs], dtype=int)            # Count how many steps have been done in that episode
        task_successes = np.zeros([cfg.num_envs], dtype=int)       # Count how many successes for each task

        env.reset()
        curr_initial_states = [initial_states[episode_idx] for initial_states, episode_idx in zip(initial_states_chunk, 
                                                                                                  env_ep_idxs)]
        obs = env.set_init_state(curr_initial_states)

        # for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
        for t in tqdm.tqdm(range(cfg.num_trials_per_task * (max_steps + cfg.num_steps_wait))):
            # try:
            if True:
                # Prepare observations dict for all images and states in obs
                # Note: OpenVLA does not take proprio state as input
                observations = {
                    "full_images": [],
                    "states": []
                }
                for one_obs in obs:
                    img = get_libero_image(one_obs, resize_size)
                    observations["full_images"].append(img)
                    observations["states"].append(np.concatenate(
                        (
                            one_obs["robot0_eef_pos"], 
                            quat2axisangle(one_obs["robot0_eef_quat"]), 
                            one_obs["robot0_gripper_qpos"]
                        )
                    ))

                # Save preprocessed image for replay video
                # TODO: Implement replay
                # replay_images.append(img)

                # TODO: Implement obs history
                # if cfg.obs_history > 1:
                #     raise NotImplementedError

                # Query model to get action
                images = []
                for image in observations["full_images"]:
                    image = Image.fromarray(image)
                    image = image.convert("RGB")

                    # (If trained with image augmentations) Center crop image and then resize back up to original size.
                    # IMPORTANT: Let's say crop scale == 0.9. To get the new height and width (post-crop), we must multiply
                    #            the original height and width by sqrt(0.9) -- not 0.9!
                    if cfg.center_crop:
                        raise NotImplementedError
                    
                    images.append(image)
                
                actions = model.batch_predict_action(
                    images, instruction_chunk, unnorm_key=cfg.unnorm_key
                )
                

                # Normalize gripper action [0,1] -> [-1,+1] because the environment expects the latter
                actions = normalize_gripper_action(actions, binarize=True)

                # [OpenVLA] The dataloader flips the sign of the gripper action to align with other datasets
                # (0 = close, 1 = open), so flip it back (-1 = open, +1 = close) before executing the action
                if cfg.model_family in ["openvla", "prismatic"]:
                    actions = invert_gripper_action(actions)

                # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                # and we need to wait for them to fall
                # Get all envs with timestep < the wait time and convert their action to noop
                wait_mask = env_steps < cfg.num_steps_wait
                if np.any(wait_mask):
                    wait_actions = np.array([get_libero_dummy_action(cfg.model_family)] * wait_mask.astype(int).sum())
                    actions[wait_mask] = wait_actions

                # Execute action in environment
                obs, reward, done, info = env.step(actions)

                # Increment step counter for 
                env_steps += 1

                not_done_with_all_eps = env_ep_idxs < cfg.num_trials_per_task
                done = np.array(done)

                # Increment success counter for all envs that are done AND haven't finished all their episodes yet
                task_successes += done.astype(int) * not_done_with_all_eps.astype(int)
                total_successes += np.sum(done.astype(int) * not_done_with_all_eps.astype(int))

                # Reset those that hit time limit
                time_limit_reached = env_steps >= (max_steps + cfg.num_steps_wait)
                
                # Increment env_ep_idxs, then recompute not_done_with_all_eps
                # (Important because this is used for reset_mask. If an env finishes its final episode, it 
                # should not be reset!)
                env_ep_idxs += ((time_limit_reached | done) & not_done_with_all_eps).astype(int)
                total_episodes += np.sum(((time_limit_reached | done) & not_done_with_all_eps).astype(int))
                # print(time_limit_reached)
                # print(done)
                # print(not_done_with_all_eps)
                not_done_with_all_eps = env_ep_idxs < cfg.num_trials_per_task

                # Resets occur if done is true for an env (in which case the task successes also go up)
                # OR because the time limit is reached, but not if that env is done with all episodes
                reset_mask = (time_limit_reached | done) & not_done_with_all_eps
                # print(time_limit_reached)
                # print(done)
                # print(not_done_with_all_eps)
                if np.any(reset_mask):
                    reset_ids = np.where(reset_mask)[0]
                    print(f"Resetting envs: {reset_ids}")
                    env.reset(reset_ids)

                    # Increment the episode count of all envs that are done
                    env_ep_idxs += reset_mask.astype(int)

                    # For all reset envs, set their initial state to the new episode's initial state
                    curr_initial_states = []
                    for episode_idx, initial_states, reset in zip(env_ep_idxs, initial_states_chunk, reset_mask):
                        if reset:
                            curr_initial_states.append(initial_states[episode_idx])
                    
                    reset_obs_list = env.set_init_state(curr_initial_states, reset_ids)

                    for reset_obs, reset_id in zip(reset_obs_list, reset_ids):
                        obs[reset_id] = reset_obs

                    env_steps[reset_mask] = 0

                    # TODO: Save a replay video of the episode
                    # save_rollout_video(
                    #     replay_images, total_episodes, success=done, task_description=task_description, log_file=log_file
                    # )

                    # Save the videos to wandb
                    # if cfg.use_wandb and (task_successes < 10 or task_episodes - task_successes < 10):
                    #     group = "success" if done else "failure"
                    #     idx = task_successes if done else task_episodes - task_successes
                    #     wandb.log(
                    #         {f"{task_description}/{group}/{idx}": wandb.Video(np.array(replay_images).transpose(0, 3, 1, 2))}
                    #     )

                    # Log current results
                    # print(f"Success: {done}")
                    print(f"Task successes {task_successes}")
                    print(f"# episodes completed so far: {total_episodes}")
                    print(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")
                    # log_file.write(f"Success: {done}\n")
                    log_file.write(f"# episodes completed so far: {total_episodes}\n")
                    log_file.write(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)\n")
                    log_file.flush()

                    

                # If all envs are done with the max number of eps, break
                if np.all(env_ep_idxs >= cfg.num_trials_per_task):
                    break

            # except Exception as e:
            #     print(f"Caught exception: {e}")
            #     log_file.write(f"Caught exception: {e}\n")
            #     break

        print(f"Task successes {task_successes}")
        print(f"# episodes completed so far: {total_episodes}")
        print(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")
        # log_file.write(f"Success: {done}\n")
        log_file.write(f"# episodes completed so far: {total_episodes}\n")
        log_file.write(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)\n")
        log_file.flush()

    # Save local log file
    log_file.close()

    # Push total metrics and local log file to wandb
    if cfg.use_wandb:
        wandb.log(
            {
                "success_rate/total": float(total_successes) / float(total_episodes),
                "num_episodes/total": total_episodes,
            }
        )
        wandb.save(local_log_filepath)


if __name__ == "__main__":
    eval_libero()
