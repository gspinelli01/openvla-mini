"""
datasets.py

Lightweight PyTorch Dataset Definition for wrapping RLDS TFDS Pipeline; just defines transform from RLDS default
format to OpenVLA, IterableDataset shim.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple, Type, List

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, IterableDataset
from transformers import PreTrainedTokenizerBase
from transformers.models.qwen2.tokenization_qwen2_fast import Qwen2TokenizerFast

from prismatic.models.backbones.llm.prompting import PromptBuilder
from prismatic.models.backbones.vision import ImageTransform
from prismatic.util.data_utils import tree_map
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets.rlds import make_interleaved_dataset, make_single_dataset
from prismatic.vla.datasets.rlds.oxe import OXE_NAMED_MIXTURES, get_oxe_dataset_kwargs_and_weights
from prismatic.vla.datasets.rlds.utils.data_utils import NormalizationType

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100

def _get_bbox_qa(rlds_batch: Dict[str, Any], lang: str) -> Tuple[str, str]:
    obj_bbox_names = rlds_batch["obj_bbox_names"].decode().split("|")
    bbox_coords = rlds_batch["obj_bboxes"]
    bbox_answer = ""
    for i in range(len(obj_bbox_names)):
        bbox_coord_tokenized = str([round(e, 3) for e in bbox_coords[i]])
        bbox_answer += f"{obj_bbox_names[i]}: {bbox_coord_tokenized}"
        if i < len(obj_bbox_names) - 1:
            bbox_answer += ", "
    return (f"What are the relevant objects and their bounding boxes to {lang}?", bbox_answer)


def _get_low_level_motion_qa(rlds_batch: Dict[str, Any], lang: str) -> Tuple[str, str]:
    low_level_motion = rlds_batch['language_motions_future'].decode().split('|')[0]
    return (f"What motion should the robot do to {lang}?", low_level_motion)


def _get_obj_pose_answer(rlds_batch: Dict[str, Any], lang: str) -> Tuple[str, str]:
    obj_bbox_names = rlds_batch['obj_bbox_names'].decode().split('|')
    dyn_obj_names = rlds_batch['dynamic_objects'].decode().split('|')
    obj_pose_answer = ""
    for i, obj_name in enumerate(obj_bbox_names):
        if obj_name in dyn_obj_names:
            obj_pose_answer += f"{obj_name}: "
            obj_pose_answer += str([(round(x, 3), round(y, 3)) for x, y in rlds_batch['obj_poses'][:, i]])
            obj_pose_answer += ", "
    obj_pose_answer = obj_pose_answer[:-2]
    return (f"What are the relevant objects, and what should their traces be to {lang}?", obj_pose_answer)


def _get_ee_pose_2D_answer(rlds_batch: Dict[str, Any], lang: str) -> Tuple[str, str]:
    ee_pose_2D_answer = str([(round(x, 3), round(y, 3)) for x, y in rlds_batch['ee_pose_2D']])
    return (f"What should the end-effector's 2D trace be to {lang}?", ee_pose_2D_answer)


AUX_TASK_QA_FUNCTIONS = {
    "bbox": _get_bbox_qa,
    "low_level_motion": _get_low_level_motion_qa,
    "obj_pose": _get_obj_pose_answer,
    "ee_pose_2D": _get_ee_pose_2D_answer,
}


@dataclass
class BaseRLDSTransform:
    tokenizer: PreTrainedTokenizerBase
    image_transform: ImageTransform
    prompt_builder_fn: Type[PromptBuilder]
    # Optional parameters with default values 
    image_window_size: int
    predict_stop_token: bool

    def _process_image(self, rlds_batch: Dict[str, Any]) -> Image.Image:
        """Process image(s) from RLDS batch based on window size."""
        if self.image_window_size == 1:
            return Image.fromarray(rlds_batch["observation"]["image_primary"][0])
        return [Image.fromarray(rlds_batch["observation"]["image_primary"][t]) for t in range(self.image_window_size)]

    def _create_conversation(self, qa_pairs: List[Tuple[str, str]]) -> Tuple[List[Dict[str, str]], List[int]]:
        """Create conversation turns from question and answer pairs.
        
        Args:
            qa_pairs: List of (question, answer) tuples
            
        Returns:
            conversation: List of conversation turns with 'from' and 'value'
            answer_token_lengths: List of token lengths for each answer
        """
        conversation = []
        answer_token_lengths = []
        
        for question, answer in qa_pairs:
            conversation.append({"from": "human", "value": question})
            conversation.append({"from": "gpt", "value": answer})
            # Tokenize the answer to get token length
            tokenized_answer = self.tokenizer(answer, add_special_tokens=False)
            answer_token_lengths.append(len(tokenized_answer["input_ids"]))
            
        return conversation, answer_token_lengths

    def _process_tokens(self, conversation: List[Dict[str, str]], answer_token_lengths: List[int]) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Process tokens and create input_ids and labels.

        Args:
            conversation: List of conversation turns
            answer_token_lengths: List of number of tokens in each answer

        Returns:
            input_ids_tensor: Tokenized input IDs
            labels: Labels tensor with target tokens marked and others as IGNORE_INDEX
            num_end_tokens: Number of end tokens
        """
        prompt_builder = self.prompt_builder_fn("openvla")
        gpt_answers = []
        
        # Build the prompt and keep track of 'gpt' answers
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])
            if turn["from"] == "gpt":
                gpt_answers.append(turn["value"])

        # Tokenize with offsets
        tokenized = self.tokenizer(
            prompt_builder.get_prompt(),
            add_special_tokens=True,
            return_offsets_mapping=True
        )
        input_ids = torch.tensor(tokenized["input_ids"], dtype=torch.long)
        labels = torch.full_like(input_ids, IGNORE_INDEX)

        # Get the full prompt text
        full_text = prompt_builder.get_prompt()

        # Find the positions of 'gpt' answers in the text
        offset_mapping = tokenized["offset_mapping"]
        current_search_start = 0
        for answer in gpt_answers:
            # Locate the start index of the answer in the full text
            start_idx = full_text.find(answer, current_search_start)
            if start_idx == -1:
                raise ValueError("Could not find the start of the answer in the prompt.")
            end_idx = start_idx + len(answer)
            current_search_start = end_idx

            # Assign labels to the tokens corresponding to the answer
            for i, (start, end) in enumerate(offset_mapping):
                if start >= end_idx:
                    break
                if start >= start_idx and end <= end_idx:
                    labels[i] = input_ids[i]

        # Handle end tokens if prediction is required
        num_end_tokens = 1
        if isinstance(self.tokenizer, Qwen2TokenizerFast):
            num_end_tokens = 2

        if self.predict_stop_token and len(input_ids) >= num_end_tokens:
            labels[-num_end_tokens:] = input_ids[-num_end_tokens:]

        return input_ids, labels, num_end_tokens


@dataclass
class RLDSBatchTransform(BaseRLDSTransform):
    # New required parameter
    action_tokenizer: ActionTokenizer

    def __call__(self, rlds_batch: Dict[str, Any]) -> Dict[str, Any]:
        """Converts a RLDS batch to the format expected by the OpenVLA collator/models."""
        dataset_name, action = rlds_batch["dataset_name"], rlds_batch["action"]
        lang = rlds_batch["task"]["language_instruction"].decode().lower()
        img = self._process_image(rlds_batch)

        if self.action_tokenizer.required_future_horizon == 0:
            action = action[-1]
        else:
            action = action[-self.action_tokenizer.required_future_horizon - 1:]

        tokenized_action = self.action_tokenizer(action)
        conversation, answer_token_lengths = self._create_conversation([
            (f"What action should the robot take to {lang}?", tokenized_action)
        ])

        input_ids, labels, _ = self._process_tokens(conversation, answer_token_lengths)
        pixel_values = self.image_transform(img)
        transform_types = np.ones(len(AUX_TASK_QA_FUNCTIONS) + 1, dtype=np.int32) * -1
        transform_types[0] = 0
        return dict(pixel_values=pixel_values, input_ids=input_ids, labels=labels, dataset_name=dataset_name, transform_type=transform_types)


@dataclass
class RLDSAuxTransform(BaseRLDSTransform):
    image_window_size: int
    predict_stop_token: bool
    aux_task_type: str

    def __post_init__(self):
        assert self.aux_task_type in AUX_TASK_QA_FUNCTIONS, f"Invalid aux task type: {self.aux_task_type}!"

    def __call__(self, rlds_batch: Dict[str, Any]) -> Dict[str, Any]:
        """Converts a RLDS batch to the format expected by the OpenVLA collator/models for aux prediction."""
        dataset_name = rlds_batch["dataset_name"]
        lang = rlds_batch["task"]["language_instruction"].decode().lower()
        img = self._process_image(rlds_batch)

        aux_question, aux_answer = AUX_TASK_QA_FUNCTIONS[self.aux_task_type](rlds_batch, lang)
        conversation, answer_token_lengths = self._create_conversation([
            (aux_question, aux_answer)
        ])

        input_ids, labels, _ = self._process_tokens(conversation, answer_token_lengths)
        pixel_values = self.image_transform(img)
        transform_types = np.ones(len(AUX_TASK_QA_FUNCTIONS) + 1, dtype=np.int32) * -1
        transform_types[0] = sorted(AUX_TASK_QA_FUNCTIONS.keys()).index(self.aux_task_type) + 1
        return dict(pixel_values=pixel_values, input_ids=input_ids, labels=labels, dataset_name=dataset_name, transform_type=transform_types)


@dataclass
class ChainedTransform(BaseRLDSTransform):
    action_tokenizer: ActionTokenizer
    aux_task_types: List[Tuple[str, str]]

    def __post_init__(self):
        assert len(self.aux_task_types) > 0, "Must specify at least one aux task type!"
        assert len(self.aux_task_types) <= len(AUX_TASK_QA_FUNCTIONS), "Cannot specify more aux task types than available!"
        for aux_task_type in self.aux_task_types:
            assert aux_task_type in AUX_TASK_QA_FUNCTIONS, f"Invalid aux task type: {aux_task_type}!"

    def __call__(self, rlds_batch: Dict[str, Any]) -> Dict[str, Any]:
        """Chains bbox and action prediction in a CoT way with separate supervision."""
        dataset_name = rlds_batch["dataset_name"]
        lang = rlds_batch["task"]["language_instruction"].decode().lower()
        img = self._process_image(rlds_batch)
        pixel_values = self.image_transform(img)

        # First get bbox answer
        qa_pairs = []
        transform_types = np.ones(len(AUX_TASK_QA_FUNCTIONS) + 1, dtype=np.int32) * -1
        for i, aux_task_type in enumerate(self.aux_task_types):
            aux_task_question, aux_task_answer = AUX_TASK_QA_FUNCTIONS[aux_task_type](rlds_batch, lang)
            qa_pairs.append((aux_task_question, aux_task_answer))
            transform_types[i] = sorted(AUX_TASK_QA_FUNCTIONS.keys()).index(aux_task_type) + 1
        transform_types[i + 1] = 0
        
        # Then get action answer
        action = rlds_batch["action"]
        if self.action_tokenizer.required_future_horizon == 0:
            action = action[-1]
        else:
            action = action[-self.action_tokenizer.required_future_horizon - 1:]
        tokenized_action = self.action_tokenizer(action)

        # Create conversation with bbox and action
        conversation, answer_token_lengths = self._create_conversation(qa_pairs + [
            (f"Given this information, what action should the robot take?", tokenized_action)
        ])

        input_ids, labels, _ = self._process_tokens(conversation, answer_token_lengths)
        return dict(
            pixel_values=pixel_values,
            input_ids=input_ids,
            labels=labels,
            dataset_name=dataset_name,
            transform_type=transform_types
        )
    

class RLDSDataset(IterableDataset):
    def __init__(
        self,
        data_root_dir: Path,
        data_mix: str,
        batch_transforms: List[Tuple[RLDSBatchTransform, float]], 
        resize_resolution: Tuple[int, int],
        shuffle_buffer_size: int = 256_000,
        train: bool = True,
        image_aug: bool = False,
        future_action_window_size: int = 0,
        future_obj_pose_window_size: int = 0,
        future_2D_trace_window_size: int = 0,
        obj_pose_stride: int = 1,
        ee_pose_2D_stride: int = 1,
        image_window_size: int = 1,
    ) -> None:
        """Lightweight wrapper around RLDS TFDS Pipeline for use with PyTorch/OpenVLA Data Loaders."""
        self.data_root_dir, self.data_mix, self.batch_transforms = data_root_dir, data_mix, batch_transforms

        if not isinstance(self.batch_transforms, list):
            self.batch_transforms = [(self.batch_transforms, 1.0)]
        else:
            assert abs(sum(weight for _, weight in self.batch_transforms) - 1.0) < 1e-6, "Batch transform weights must sum to 1.0!"

        # Configure RLDS Dataset(s)
        if self.data_mix in OXE_NAMED_MIXTURES:
            mixture_spec = OXE_NAMED_MIXTURES[self.data_mix]
        else:
            # Assume that passed "mixture" name is actually a single dataset -- create single-dataset "mix"
            mixture_spec = [(self.data_mix, 1.0)]

        # fmt: off
        per_dataset_kwargs, weights = get_oxe_dataset_kwargs_and_weights(
            self.data_root_dir,
            mixture_spec,
            load_camera_views=("primary",),
            load_depth=False,
            load_proprio=False,
            load_language=True,
            action_proprio_normalization_type=NormalizationType.BOUNDS_Q99,
        )
        rlds_config = dict(
            traj_transform_kwargs=dict(
                window_size=image_window_size,                        # If we wanted to feed / predict more than one step
                future_action_window_size=future_action_window_size,  # For action chunking
                future_obj_pose_window_size=future_obj_pose_window_size,
                future_2D_trace_window_size=future_2D_trace_window_size,
                obj_pose_stride=obj_pose_stride,
                ee_pose_2D_stride=ee_pose_2D_stride,
                skip_unlabeled=True,                                  # Skip trajectories without language labels
                goal_relabeling_strategy="uniform",                   # Goals are currently unused
            ),
            frame_transform_kwargs=dict(
                resize_size=resize_resolution,
                num_parallel_calls=16,                          # For CPU-intensive ops (decoding, resizing, etc.)
            ),
            dataset_kwargs_list=per_dataset_kwargs,
            shuffle_buffer_size=shuffle_buffer_size,
            sample_weights=weights,
            balance_weights=True,
            traj_transform_threads=len(mixture_spec),
            traj_read_threads=len(mixture_spec),
            train=train,
        )

        # If applicable, enable image augmentations
        if image_aug:
            rlds_config["frame_transform_kwargs"].update({"image_augment_kwargs" : dict(
                random_resized_crop=dict(scale=[0.9, 0.9], ratio=[1.0, 1.0]),
                random_brightness=[0.2],
                random_contrast=[0.8, 1.2],
                random_saturation=[0.8, 1.2],
                random_hue=[0.05],
                augment_order=[
                    "random_resized_crop",
                    "random_brightness",
                    "random_contrast",
                    "random_saturation",
                    "random_hue",
                ],
            )}),
        # fmt: on

        # Initialize RLDS Dataset
        self.dataset, self.dataset_length, self.dataset_statistics = self.make_dataset(rlds_config)

    def make_dataset(self, rlds_config):
        return make_interleaved_dataset(**rlds_config)

    def __iter__(self) -> Dict[str, Any]:
        transforms, weights = zip(*self.batch_transforms)
        for rlds_batch in self.dataset.as_numpy_iterator():
            # Select a random transform using fixed ordering of transforms and weights
            transform = np.random.choice(transforms, p=weights)
            yield transform(rlds_batch)

    def __len__(self) -> int:
        return self.dataset_length

    # === Explicitly Unused ===
    def __getitem__(self, idx: int) -> None:
        raise NotImplementedError("IterableDataset does not implement map-style __getitem__; see __iter__ instead!")


class EpisodicRLDSDataset(RLDSDataset):
    """Returns full episodes as list of steps instead of individual transitions (useful for visualizations)."""

    def make_dataset(self, rlds_config):
        per_dataset_kwargs = rlds_config["dataset_kwargs_list"]
        assert len(per_dataset_kwargs) == 1, "Only support single-dataset `mixes` for episodic datasets."

        return make_single_dataset(
            per_dataset_kwargs[0],
            train=rlds_config["train"],
            traj_transform_kwargs=rlds_config["traj_transform_kwargs"],
            frame_transform_kwargs=rlds_config["frame_transform_kwargs"],
        )
    def __iter__(self) -> Dict[str, Any]:
        transforms, weights = zip(*self.batch_transforms)
        for rlds_batch in self.dataset.as_numpy_iterator():
            # Select a random transform using fixed ordering of transforms and weights
            transform = np.random.choice(transforms, p=weights)
            out = [
                transform(tree_map(lambda x: x[i], rlds_batch))  # noqa: B023
                for i in range(rlds_batch["action"].shape[0])
            ]
            yield out

class DummyDataset(Dataset):
    def __init__(
        self,
        action_tokenizer: ActionTokenizer,
        base_tokenizer: PreTrainedTokenizerBase,
        image_transform: ImageTransform,
        prompt_builder_fn: Type[PromptBuilder],
    ) -> None:
        self.action_tokenizer = action_tokenizer
        self.base_tokenizer = base_tokenizer
        self.image_transform = image_transform
        self.prompt_builder_fn = prompt_builder_fn

        # Note =>> We expect the dataset to store statistics for action de-normalization. Specifically, we store the
        # per-dimension 1st and 99th action quantile. The values below correspond to "no normalization" for simplicity.
        self.dataset_statistics = {
            "dummy_dataset": {
                "action": {"q01": np.zeros((7,), dtype=np.float32), "q99": np.ones((7,), dtype=np.float32)}
            }
        }

    def __len__(self):
        # TODO =>> Replace with number of elements in your dataset!
        return 10000

    def __getitem__(self, idx):
        # TODO =>> Load image, action and instruction from disk -- we use dummy values
        image = Image.fromarray(np.asarray(np.random.rand(224, 224, 3) * 255.0, dtype=np.uint8))
        action = np.asarray(np.random.rand(7), dtype=np.float32)
        instruction = "do something spectacular"

        # Add instruction to VLA prompt
        prompt_builder = self.prompt_builder_fn("openvla")
        conversation = [
            {"from": "human", "value": f"What action should the robot take to {instruction}?"},
            {"from": "gpt", "value": self.action_tokenizer(action)},
        ]
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        # Tokenize (w/ `base_tokenizer`)
        input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        labels = list(input_ids)

        # Tensorize =>> Run Image Transform to get `pixel_values` =>> Return
        #   =>> IMPORTANT :: IF WE'RE USING HF .forward(..., labels=labels), SHIFTING HAPPENS _INSIDE_ MODEL!
        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)
        pixel_values = self.image_transform(image)

        # [CRITICAL] We do not want to take the loss for anything but the predicted action tokens!
        labels[: -(len(action) + 1)] = IGNORE_INDEX

        return dict(pixel_values=pixel_values, input_ids=input_ids, labels=labels)
