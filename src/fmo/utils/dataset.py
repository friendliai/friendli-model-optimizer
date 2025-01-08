# Copyright (c) 2024-present, FriendliAI Inc. All rights reserved.
# pylint: disable=no-value-for-parameter,no-name-in-module,too-many-positional-arguments,too-many-arguments

"""FMO-CLI Dataset Utils."""
from __future__ import annotations

import os
import sys
from typing import Optional

import datasets  # type: ignore
from torch.utils.data import ConcatDataset, Dataset
from transformers import (  # type: ignore
    AutoTokenizer,
    BatchFeature,
    PreTrainedTokenizer,
)

from fmo.logging import get_logger

logger = get_logger(__name__)


def safe_load_datasets(
    dataset_name_or_path: str,
    split_name: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> datasets.Dataset:
    """Load dataset from calibration dataset config."""
    try:
        if os.path.exists(dataset_name_or_path):
            dataset = datasets.load_dataset(
                data_files=dataset_name_or_path,
                split=split_name,
            )
        else:
            data_name_parts = dataset_name_or_path.split(":")
            if len(data_name_parts) == 1:
                dataset = datasets.load_dataset(
                    dataset_name_or_path, split=split_name, cache_dir=cache_dir
                )
            elif len(data_name_parts) == 2:
                data_name, subset_name = data_name_parts
                dataset = datasets.load_dataset(
                    data_name, subset_name, split=split_name, cache_dir=cache_dir
                )
            else:
                logger.error(
                    "Dataset name is in invalid format. "
                    "(valid format: '<dataset_name>' or '<dataset_name>:<subset_name>')"
                )
                sys.exit(1)

    except ValueError as e:
        logger.error("datasets.load_dataset failed. %s", e)
        sys.exit(1)

    if not isinstance(dataset, datasets.Dataset):
        logger.error("This dataset format is not supported for the calibration.")
        sys.exit(1)

    return dataset


def get_tokenizer(
    model_name_or_path: str,
    *,
    cache_dir: Optional[str] = None,
) -> PreTrainedTokenizer:
    """Try to get tokenizer of a pretrained model."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            cache_dir=cache_dir,
            trust_remote_code=True,
        )
    except OSError as e:
        logger.error(e)
        sys.exit(1)

    if not tokenizer.is_fast:
        logger.error("This model does not support Friendli-compatible tokenizer")
        sys.exit(1)

    if tokenizer.pad_token != "<unk>":
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def load_dataset_from_hf(
    dataset_name_or_path: str,
    dataset_target_column_name: str,
    dataset_max_length: int,
    dataset_num_samples: int,
    dataset_split_name: str,
    cache_dir: Optional[str],
    tokenizer: PreTrainedTokenizer,
    seed: int,
) -> Dataset:
    """Try to load calibration dataset from huggingface."""
    hf_dataset = (
        safe_load_datasets(
            dataset_name_or_path=dataset_name_or_path,
            split_name=dataset_split_name,
            cache_dir=cache_dir,
        )
        .shuffle(seed=seed)
        .select(range(dataset_num_samples))
    )
    non_chat_ds_w_special_tokens = TextOnlyCalibDataset(
        hf_dataset[dataset_target_column_name][: dataset_num_samples // 2],
        tokenizer,
        max_length=dataset_max_length,
        add_special_tokens=False,
    )
    non_chat_ds_wo_special_tokens = TextOnlyCalibDataset(
        hf_dataset[dataset_target_column_name][dataset_num_samples // 2 :],
        tokenizer,
        max_length=dataset_max_length,
        add_special_tokens=True,
    )
    return ConcatDataset([non_chat_ds_w_special_tokens, non_chat_ds_wo_special_tokens])


class TextOnlyCalibDataset(Dataset):
    """A dataset for calibration that contains only tokenizable text inputs."""

    def __init__(self, texts, tokenizer, max_length, add_special_tokens=True):
        """Initialize TextOnlyCalibDataset."""
        self.texts = texts
        self.tokenizer = tokenizer
        self.add_special_tokens = add_special_tokens
        self.max_length = max_length
        self.encodings = self._encode_texts()

    def _encode_texts(self):
        """Encode the input texts using the tokenizer."""
        return self.tokenizer(
            self.texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            add_special_tokens=self.add_special_tokens,
        )

    def __len__(self):
        """Get the number of samples in the dataset."""
        return len(self.texts)

    def __getitem__(self, idx):
        """Retrieve a single sample by index."""
        return BatchFeature(
            data={
                "input_ids": self.encodings["input_ids"][idx],
                "attention_mask": self.encodings["attention_mask"][idx],
            }
        )
