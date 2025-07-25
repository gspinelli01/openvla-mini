"""
Regenerates a LIBERO dataset (HDF5 files) by replaying demonstrations in the environments.

Notes:
    - We save image observations at 256x256px resolution (instead of 128x128).
    - We filter out transitions with "no-op" (zero) actions that do not change the robot's state.
    - We filter out unsuccessful demonstrations.
    - In the LIBERO HDF5 data -> RLDS data conversion (not shown here), we rotate the images by
    180 degrees because we observe that the environments return images that are upside down
    on our platform.

Usage:
    python experiments/robot/libero/regenerate_libero_dataset.py \
        --libero_task_suite [ libero_spatial | libero_object | libero_goal | libero_10 ] \
        --libero_raw_data_dir <PATH TO RAW HDF5 DATASET DIR> \
        --libero_target_dir <PATH TO TARGET DIR>

    Example (LIBERO-Spatial):
        python experiments/robot/libero/regenerate_libero_dataset.py \
            --libero_task_suite libero_spatial \
            --libero_raw_data_dir ./LIBERO/libero/datasets/libero_spatial \
            --libero_target_dir ./LIBERO/libero/datasets/libero_spatial_no_noops

"""

import argparse
import json
import os
import cv2

import h5py
import numpy as np
import robosuite.utils.transform_utils as T
import tqdm
from libero.libero import benchmark

from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
)

IMAGE_RESOLUTION = 256


def is_noop(action, prev_action=None, threshold=1e-4):
    """
    Returns whether an action is a no-op action.

    A no-op action satisfies two criteria:
        (1) All action dimensions, except for the last one (gripper action), are near zero.
        (2) The gripper action is equal to the previous timestep's gripper action.

    Explanation of (2):
        Naively filtering out actions with just criterion (1) is not good because you will
        remove actions where the robot is staying still but opening/closing its gripper.
        So you also need to consider the current state (by checking the previous timestep's
        gripper action as a proxy) to determine whether the action really is a no-op.
    """
    # Special case: Previous action is None if this is the first action in the episode
    # Then we only care about criterion (1)
    if prev_action is None:
        return np.linalg.norm(action[:-1]) < threshold

    # Normal case: Check both criteria (1) and (2)
    gripper_action = action[-1]
    prev_gripper_action = prev_action[-1]
    return np.linalg.norm(action[:-1]) < threshold and gripper_action == prev_gripper_action


def main(args):
    print(f"Regenerating {args.ybq_task_suite} dataset!")

    # Create target directory
    if os.path.isdir(args.libero_target_dir):
        user_input = input(
            f"Target directory already exists at path: {args.libero_target_dir}\n"
            "Enter 'y' to overwrite the directory, or anything else to exit: "
        )
        if user_input != "y":
            exit()
    os.makedirs(args.libero_target_dir, exist_ok=True)

    # Prepare JSON file to record success/false and initial states per episode
    metainfo_json_dict = {}
    metainfo_json_out_path = f"./experiments/robot/libero/{args.ybq_task_suite}_metainfo.json"
    with open(metainfo_json_out_path, "w") as f:
        # Just test that we can write to this file (we overwrite it later)
        json.dump(metainfo_json_dict, f)

    # Get task suite
    # benchmark_dict = benchmark.get_benchmark_dict()
    # task_suite = benchmark_dict[args.libero_task_suite]()
    # num_tasks_in_suite = task_suite.n_tasks

    # Get folder where .bddl are stored
    from pathlib import Path
    import re
    from libero.libero.envs import OffScreenRenderEnv

    task_bddl_folder = Path('/tmp/pddl/ybq_tasks')
    all_tasks  = list(task_bddl_folder.glob('**/*.bddl'))
    num_tasks_in_suite = len(all_tasks)

    # Setup
    num_replays = 0
    num_success = 0
    num_noops = 0

    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        # Get task in suite
        # task = task_suite.get_task(task_id)
        # env, task_description = get_libero_env(task, "llava", resolution=IMAGE_RESOLUTION)
        """Initializes and returns the LIBERO environment, along with the task description."""

        task_bddl_file = str(all_tasks[task_id])
        task_description = task_bddl_file.split('/')[-1]
        task_description = re.split('\w+_SCENE\d_', task_description)[-1].split('.')[0]

        env_args = {"bddl_file_name": task_bddl_file, "camera_heights": IMAGE_RESOLUTION, "camera_widths": IMAGE_RESOLUTION}
        env = OffScreenRenderEnv(**env_args)
        env.seed(0)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state

        # Get dataset for task

        # --demonstration_data
        #   --
        #   --

        import glob
        hdf5_file_folder = glob.glob(f'{args.libero_raw_data_dir}/*{task_description}')

        if len(hdf5_file_folder) == 0:
            continue
        
        assert len(hdf5_file_folder) == 1, 'you must have only 1 demo.hdf5 folder!'

        orig_data_path = os.path.join(hdf5_file_folder[0], "demo.hdf5")
        assert os.path.exists(orig_data_path), f"Cannot find raw data file {orig_data_path}."
        orig_data_file = h5py.File(orig_data_path, "r")
        orig_data = orig_data_file["data"]

        # Create new HDF5 file for regenerated demos
        new_data_path = os.path.join(args.libero_target_dir, f"{task_description}_demo.hdf5")
        new_data_file = h5py.File(new_data_path, "w")
        grp = new_data_file.create_group("data")

        for i in range(1, len(orig_data.keys())+1):
            # Get demo data
            demo_data = orig_data[f"demo_{i}"]
            orig_actions = demo_data["actions"][()]
            orig_states = demo_data["states"][()]

            # Reset environment, set initial state, and wait a few steps for environment to settle
            env.reset()
            env.set_init_state(orig_states[0])
            for _ in range(10):
                obs, reward, done, info = env.step(get_libero_dummy_action("llava"))

            # Set up new data lists
            states = []
            actions = []
            ee_states = []
            gripper_states = []
            joint_states = []
            robot_states = []
            agentview_images = []
            eye_in_hand_images = []

            # Replay original demo actions in environment and record observations
            for _, action in enumerate(orig_actions):
                wait_for_gripper = False
                # Skip transitions with no-op actions
                prev_action = actions[-1] if len(actions) > 0 else None

                if action[-1] == 1.0 and prev_action[-1] == -1.0 and not wait_for_gripper:
                    wait_for_gripper = True

                
                if is_noop(action, prev_action):
                    if wait_for_gripper and action[-1] == prev_action[-1]:
                        print(f'waiting for gripper')
                    else:
                        print(f"\tSkipping no-op action: {action}")
                        num_noops += 1
                        continue
                elif wait_for_gripper and np.linalg.norm(action[:-1]) > 0.01:
                    print(f'end')
                    wait_for_gripper = False
                else:
                    print(f'action: {action}')

                if not wait_for_gripper:
                    if states == []:
                        # In the first timestep, since we're using the original initial state to initialize the environment,
                        # copy the initial state (first state in episode) over from the original HDF5 to the new one
                        states.append(orig_states[0])
                        # robot_states.append(demo_data["robot_states"][0])
                    else:
                        # For all other timesteps, get state from environment and record it
                        states.append(env.sim.get_state().flatten())
                        robot_states.append(
                            np.concatenate([obs["robot0_gripper_qpos"], obs["robot0_eef_pos"], obs["robot0_eef_quat"]])
                        )

                    # Record original action (from demo)
                    actions.append(action)

                    # Record data returned by environment
                    if "robot0_gripper_qpos" in obs:
                        gripper_states.append(obs["robot0_gripper_qpos"])
                    joint_states.append(obs["robot0_joint_pos"])
                    ee_states.append(
                        np.hstack(
                            (
                                obs["robot0_eef_pos"],
                                T.quat2axisangle(obs["robot0_eef_quat"]),
                            )
                        )
                    )
                    agentview_images.append(obs["agentview_image"])
                    # => Debug:
                    import time
                    from copy import deepcopy
                    image = deepcopy(obs["agentview_image"][::-1,:,::-1])
                    image = cv2.putText(image, f'{str(action)}', (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.3, (0, 0, 255), 1, cv2.LINE_AA)
                    cv2.imwrite('test_obs.png', image)
                    # time.sleep(0.2)

                    eye_in_hand_images.append(obs["robot0_eye_in_hand_image"])

                # Execute demo action in environment
                obs, reward, done, info = env.step(action.tolist())

            done = True
            # At end of episode, save replayed trajectories to new HDF5 files (only keep successes)
            if done:
                dones = np.zeros(len(actions)).astype(np.uint8)
                dones[-1] = 1
                rewards = np.zeros(len(actions)).astype(np.uint8)
                rewards[-1] = 1
                assert len(actions) == len(agentview_images)

                ep_data_grp = grp.create_group(f"demo_{i}")
                obs_grp = ep_data_grp.create_group("obs")
                obs_grp.create_dataset("gripper_states", data=np.stack(gripper_states, axis=0))
                obs_grp.create_dataset("joint_states", data=np.stack(joint_states, axis=0))
                obs_grp.create_dataset("ee_states", data=np.stack(ee_states, axis=0))
                obs_grp.create_dataset("ee_pos", data=np.stack(ee_states, axis=0)[:, :3])
                obs_grp.create_dataset("ee_ori", data=np.stack(ee_states, axis=0)[:, 3:])
                obs_grp.create_dataset("agentview_rgb", data=np.stack(agentview_images, axis=0))
                obs_grp.create_dataset("eye_in_hand_rgb", data=np.stack(eye_in_hand_images, axis=0))
                ep_data_grp.create_dataset("actions", data=actions)
                ep_data_grp.create_dataset("states", data=np.stack(states))
                ep_data_grp.create_dataset("robot_states", data=np.stack(robot_states, axis=0))
                ep_data_grp.create_dataset("rewards", data=rewards)
                ep_data_grp.create_dataset("dones", data=dones)

                num_success += 1

            num_replays += 1

            # Record success/false and initial environment state in metainfo dict
            task_key = task_description.replace(" ", "_")
            episode_key = f"demo_{i}"
            if task_key not in metainfo_json_dict:
                metainfo_json_dict[task_key] = {}
            if episode_key not in metainfo_json_dict[task_key]:
                metainfo_json_dict[task_key][episode_key] = {}
            metainfo_json_dict[task_key][episode_key]["success"] = bool(done)
            metainfo_json_dict[task_key][episode_key]["initial_state"] = orig_states[0].tolist()

            # Write metainfo dict to JSON file
            # (We repeatedly overwrite, rather than doing this once at the end, just in case the script crashes midway)
            with open(metainfo_json_out_path, "w") as f:
                json.dump(metainfo_json_dict, f, indent=2)

            # Count total number of successful replays so far
            print(
                f"Total # episodes replayed: {num_replays}, "
                f"Total # successes: {num_success} ({num_success / num_replays * 100:.1f} %)"
            )

            # Report total number of no-op actions filtered out so far
            print(f"  Total # no-op actions filtered out: {num_noops}")

        # Close HDF5 files
        orig_data_file.close()
        new_data_file.close()
        print(f"Saved regenerated demos for task '{task_description}' at: {new_data_path}")

    print(f"Dataset regeneration complete! Saved new dataset at: {args.libero_target_dir}")
    print(f"Saved metainfo JSON at: {metainfo_json_out_path}")


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ybq_task_suite",
        type=str,
        choices=["ybq_floor"],
        help="YBQ task suite. Example: yqb_floor",
        required=True,
    )
    parser.add_argument(
        "--libero_raw_data_dir",
        type=str,
        help=("Path to directory containing raw HDF5 dataset. " "Example: ./LIBERO/libero/datasets/libero_spatial"),
        required=True,
    )
    parser.add_argument(
        "--libero_target_dir",
        type=str,
        help=("Path to regenerated dataset directory. " "Example: ./LIBERO/libero/datasets/libero_spatial_no_noops"),
        required=True,
    )
    parser.add_argument('-d', '--debug',
                    action='store_true')
    args = parser.parse_args()

    if args.debug:
        import debugpy
        debugpy.listen(5678)
        print('wait for client to attach')
        debugpy.wait_for_client()

    # Start data regeneration
    main(args)
