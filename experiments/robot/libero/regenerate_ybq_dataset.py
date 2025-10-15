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
import re
import cv2
from copy import deepcopy

import h5py
import numpy as np
import robosuite.utils.transform_utils as T
import tqdm
from libero.libero import benchmark
import imageio

from libero.libero.envs.base_object import register_object
from pathlib import Path
from robosuite.models.objects import MujocoXMLObject

from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
)

IMAGE_RESOLUTION = 256
jmhr_path_root = Path(os.getcwd()).parent.parent

class CustomObjects(MujocoXMLObject):
    def __init__(self, custom_path, name, obj_name, joints=[dict(type="free", damping="0.0005")]):
        # make sure custom path is an absolute path
        assert(os.path.isabs(custom_path)), "Custom path must be an absolute path"
        # make sure the custom path is also an xml file
        assert(custom_path.endswith(".xml")), "Custom path must be an xml file"
        super().__init__(
            custom_path,
            name=name,
            joints=joints,
            obj_type="all",
            duplicate_collision_geoms=False,
        )
        self.category_name = "_".join(
            re.sub(r"([A-Z])", r" \1", self.__class__.__name__).split()
        ).lower()
        self.object_properties = {"vis_site_names": {}}

#TODO: get absolute path from code
@register_object
class SquareNut(CustomObjects):
    def __init__(self, name='square_nut', obj_name='square_nut', joints=[dict(type="free", damping="0.0005")]):
        super().__init__(
            os.path.join(jmhr_path_root, 'assets/nuts/square-nut.xml'),
            name=name,
            obj_name=obj_name,
            joints=joints)
        
        self.rotation = {
            "x": (3/2*np.pi, np.pi/2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class RoundNut(CustomObjects):
    def __init__(self, name='round_nut', obj_name='round_nut', joints=[dict(type="free", damping="0.0005")]):
        super().__init__(
            os.path.join(jmhr_path_root, 'assets/nuts/round-nut.xml'),
            name=name,
            obj_name=obj_name,
            joints=joints)
        
        self.rotation = {
            "x": (3/2*np.pi, np.pi/2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BrassPeg(CustomObjects):  # brass == ottone
    def __init__(self, name='brass_peg', obj_name='brass_peg', joints=[dict(type="free", damping="0.0005")]):
        super().__init__(
            os.path.join(jmhr_path_root, 'assets/peg/peg1.xml'),
            name=name,
            obj_name=obj_name,
            joints=joints)
        
        self.rotation = {
            "x": (-np.pi/2, -np.pi/2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None


@register_object
class WoodBin(CustomObjects):
    def __init__(self, name='wood_bin', obj_name='wood_bin', joints=[dict(type="free", damping="0.0005")]):
        super().__init__(
            os.path.join(jmhr_path_root, 'assets/bin/bin2.xml'),
            name=name,
            obj_name=obj_name,
            joints=joints)
        
        self.rotation = {
            "x": (-np.pi/2, -np.pi/2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class RedBlock(CustomObjects):
    def __init__(self, name='red_block', obj_name='red_block', joints=[dict(type="free", damping="0.0005")]):
        super().__init__(
            os.path.join(jmhr_path_root, 'assets/block/red_block.xml'),
            name=name,
            obj_name=obj_name,
            joints=joints)
        
        self.rotation = {
            "x": (-np.pi/2, np.pi/2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class BlueBlock(CustomObjects):
    def __init__(self, name='blue_block', obj_name='blue_block', joints=[dict(type="free", damping="0.0005")]):
        super().__init__(
            os.path.join(jmhr_path_root, 'assets/block/blue_block.xml'),
            name=name,
            obj_name=obj_name,
            joints=joints)
        
        self.rotation = {
            "x": (-np.pi/2, np.pi/2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class GreenBlock(CustomObjects):
    def __init__(self, name='green_block', obj_name='green_block', joints=[dict(type="free", damping="0.0005")]):
        super().__init__(
            os.path.join(jmhr_path_root, 'assets/block/green_block.xml'),
            name=name,
            obj_name=obj_name,
            joints=joints)
        
        self.rotation = {
            "x": (-np.pi/2, np.pi/2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None

@register_object
class GrayBlock(CustomObjects):
    def __init__(self, name='gray_block', obj_name='gray_block', joints=[dict(type="free", damping="0.0005")]):
        super().__init__(
            os.path.join(jmhr_path_root, 'assets/block/gray_block.xml'),
            name=name,
            obj_name=obj_name,
            joints=joints)
        
        self.rotation = {
            "x": (-np.pi/2, np.pi/2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None


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

    # Create debug regenerated trajectories folder
    curr_path = Path(os.path.abspath(__file__)).parent
    debug_reg_video_folder = os.path.join(curr_path, f'filtered_trajectories_no-op-corr_{args.ybq_task_suite}')
    os.makedirs(debug_reg_video_folder, exist_ok=True)

    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        # Get task in suite
        # task = task_suite.get_task(task_id)
        # env, task_description = get_libero_env(task, "llava", resolution=IMAGE_RESOLUTION)
        task_id_folder = f'task_id_{task_id}'
        os.makedirs(os.path.join(debug_reg_video_folder, task_id_folder), exist_ok=True)
        """Initializes and returns the LIBERO environment, along with the task description."""

        task_bddl_file = str(all_tasks[task_id])
        task_description = task_bddl_file.split('/')[-1]
        task_description = re.split('\w+_SCENE\d_', task_description)[-1].split('.')[0]

        env_args = {"horizon": 10000, "bddl_file_name": task_bddl_file, "camera_heights": IMAGE_RESOLUTION, "camera_widths": IMAGE_RESOLUTION}
        env = OffScreenRenderEnv(**env_args)
        env.seed(0)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state

        # Get dataset for task

        # --demonstration_data
        #   --
        #   --

        import glob
        check = '_'.join(task_description.split('_')[2:])
        hdf5_file_folder = glob.glob(f'{args.libero_raw_data_dir}/*{check}')

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

        filtered_demos_counter = 0

        for i in range(1, len(orig_data.keys())+1):

            traj_folder = f'traj_{i}'
            os.makedirs(os.path.join(debug_reg_video_folder, task_id_folder, traj_folder), exist_ok=True)
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
                # Skip transitions with no-op actions
                prev_action = actions[-1] if len(actions) > 0 else None
                if is_noop(action, prev_action):
                    # print(f"\tSkipping no-op action: {action}")
                    try:
                        env.step(action)
                    except ValueError as e:
                        print(f'EXCEPTION: {e} \n -> We skip the action')
                        break
                    num_noops += 1
                    continue

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
                eye_in_hand_images.append(obs["robot0_eye_in_hand_image"])

                # Execute demo action in environment
                try:
                    obs, reward, done, info = env.step(action.tolist())
                except ValueError as e:
                    print(f'EXCEPTION: {e} \n -> We skip the action')
                    break


            # una volta finite le azioni, fai girare ancora fino a quando il task non segna successo (queste non vengono registrate)
            done_counter = 0
            while not done:
                try:
                    obs, reward, done, info = env.step(get_libero_dummy_action("llava"))
                except ValueError as e:
                    print(f'EXCEPTION: {e} \n -> We skip the action')
                    break
                done_counter+=1
                if done_counter >= 200:
                    break

            print(f'done counter: {done_counter}')

            # => video of the regenerated dataset
            # new_traj_path = './filtered_trajectories_no-op_correction'
            # os.makedirs(new_traj_path, exist_ok=True)     
            mp4_path = os.path.join(debug_reg_video_folder, task_id_folder, traj_folder, f"demo_{i}.mp4")
            video_writer = imageio.get_writer(mp4_path, fps=30)
            for img, act in zip(agentview_images, actions):  # these are the images and actions that will be saved in the regenerated hdf5 file
                debug_img = deepcopy(img)
                debug_img = deepcopy(debug_img[::-1,:,:])
                debug_img = cv2.putText(debug_img, f'{str(act)}', (0, 200), cv2.FONT_HERSHEY_SIMPLEX, 
                0.2, (0, 0, 255), 1, cv2.LINE_AA)
                video_writer.append_data(debug_img)
            video_writer.close()
            
            print(f"Saved rollout MP4 at path {mp4_path}")
            # At end of episode, save replayed trajectories to new HDF5 files (only keep successes)
            if done:
                filtered_demos_counter += 1
                dones = np.zeros(len(actions)).astype(np.uint8)
                dones[-1] = 1
                rewards = np.zeros(len(actions)).astype(np.uint8)
                rewards[-1] = 1
                assert len(actions) == len(agentview_images)

                ep_data_grp = grp.create_group(f"demo_{filtered_demos_counter}")
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
        choices=["ybq_floor", "ybq_table", "ybq_blocks"],
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
