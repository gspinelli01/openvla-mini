"""
materialize.py

Factory class for initializing Open-X RLDS-backed datasets, given specified data mixture parameters; provides and
exports individual functions for clear control flow.
"""

from pathlib import Path
from typing import Tuple, Type, List

from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from prismatic.models.backbones.llm.prompting import PromptBuilder
from prismatic.models.backbones.vision import ImageTransform
from prismatic.models.backbones.vision.base_vision import WrapSequenceImageTransform
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ACTION_TOKENIZERS, ActionTokenizer
from prismatic.vla.datasets import EpisodicRLDSDataset, RLDSDataset
from prismatic.vla.datasets.datasets import (
    RLDSBatchTransform, 
    RLDSAuxTransform,
    ChainedTransform,
)


def get_vla_dataset_and_collator(
    data_root_dir: Path,
    data_mix: str,
    image_transform: ImageTransform,
    tokenizer: PreTrainedTokenizerBase,
    prompt_builder_fn: Type[PromptBuilder],
    default_image_resolution: Tuple[int, int, int],
    padding_side: str = "right",
    predict_stop_token: bool = True,
    shuffle_buffer_size: int = 100_000,
    train: bool = True,
    episodic: bool = False,
    image_aug: bool = False,
    action_tokenizer: str = "action_tokenizer",
    future_action_window_size: int = 0,
    future_obj_pose_window_size: int = 0,
    future_2D_trace_window_size: int = 0,
    obj_pose_stride: int = 1,
    ee_pose_2D_stride: int = 1,
    image_window_size: int = 1,
    transform_types: str = "action",
) -> Tuple[Dataset, ActionTokenizer, PaddedCollatorForActionPrediction]:
    """Initialize RLDS Dataset (wraps TFDS), ActionTokenizer, and initialize transform/collation functions."""

    action_tokenizer: ActionTokenizer = ACTION_TOKENIZERS[action_tokenizer](tokenizer)

    # get the future action window needed from the tokenizer
    future_action_window_size = max(action_tokenizer.required_future_horizon, future_action_window_size)

    # get the observation history from the image_transform (only needed if its a WrapSequence transform)
    if isinstance(image_transform, WrapSequenceImageTransform):
        image_window_size = max(image_transform.sequence_len, image_window_size)

    transform_base_args = {
        "tokenizer": tokenizer,
        "image_transform": image_transform,
        "prompt_builder_fn": prompt_builder_fn,
        "predict_stop_token": predict_stop_token,
        "image_window_size": image_window_size,
    }

    batch_transforms = []
    # remove whitespace
    transform_types = transform_types.replace(" ", "")
    transform_types = transform_types.split(",")
    for transform_type in transform_types:
        if '|' in transform_type:
            chained_transforms = transform_type.split("|")
            rlds_transform = ChainedTransform(
                **transform_base_args,
                action_tokenizer=action_tokenizer,
                aux_task_types=chained_transforms,
            )
        elif transform_type == "action":
            rlds_transform = RLDSBatchTransform(**transform_base_args, action_tokenizer=action_tokenizer)
        else:
            rlds_transform = RLDSAuxTransform(**transform_base_args, aux_task_type=transform_type)
        batch_transforms.append((rlds_transform, 1/len(transform_types)))

    collator = PaddedCollatorForActionPrediction(
        tokenizer.model_max_length, tokenizer.pad_token_id, padding_side=padding_side
    )

    # Build RLDS Iterable Dataset
    cls = RLDSDataset if not episodic else EpisodicRLDSDataset
    dataset = cls(
        data_root_dir,
        data_mix,
        batch_transforms,
        resize_resolution=default_image_resolution[1:],
        shuffle_buffer_size=shuffle_buffer_size,
        train=train,
        image_aug=image_aug,
        future_action_window_size=future_action_window_size,
        future_obj_pose_window_size=future_obj_pose_window_size,
        future_2D_trace_window_size=future_2D_trace_window_size,
        obj_pose_stride=obj_pose_stride,
        ee_pose_2D_stride=ee_pose_2D_stride,
        image_window_size=image_window_size,
    )

    return dataset, action_tokenizer, collator
