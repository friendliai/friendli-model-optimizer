# Copyright (c) 2024-present, FriendliAI Inc. All rights reserved.
# pylint: disable=no-value-for-parameter,no-name-in-module,too-many-positional-arguments

"""FMO-CLI Dataset Utils."""
from __future__ import annotations

import os
import sys
from typing import Optional

import datasets  # type: ignore
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizer  # type: ignore

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


def get_encoded_dataset(  # pylint: disable=too-many-arguments
    tokenizer: PreTrainedTokenizer,
    dataset: datasets.Dataset,
    lookup_column_name: str,
    seed: int = 42,
    max_length: int = 2048,
    num_samples: int = 512,
) -> DataLoader:
    """Return Calibration DataLoader."""
    if num_samples == 1:
        num_samples *= 2
    try:
        dataset = dataset.shuffle(seed=seed).select(range(num_samples))  # type: ignore
        encoded_ds_w_special_tokens = tokenizer(
            dataset[lookup_column_name][: num_samples // 2],
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=max_length,
            add_special_tokens=True,
        ).input_ids
        encoded_ds_wo_special_tokens = tokenizer(
            dataset[lookup_column_name][num_samples // 2 :],
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=max_length,
            add_special_tokens=False,
        ).input_ids

        max_length_diff = (
            encoded_ds_w_special_tokens.shape[-1]
            - encoded_ds_wo_special_tokens.shape[-1]
        )
        if max_length_diff > 0:
            padded_tokens = torch.full(
                (encoded_ds_wo_special_tokens.shape[0], max_length_diff),
                tokenizer.pad_token_id,
            )
            encoded_ds_wo_special_tokens = torch.cat(
                [encoded_ds_wo_special_tokens, padded_tokens], dim=1
            )
        assert (
            encoded_ds_w_special_tokens.shape[-1]
            == encoded_ds_wo_special_tokens.shape[-1]
        )
        encoded_dataset = torch.cat(
            [encoded_ds_w_special_tokens, encoded_ds_wo_special_tokens], dim=0
        )

    except KeyError as e:
        logger.error(
            "`%s` is not valid column name in given dataset. %s", lookup_column_name, e
        )
        sys.exit(1)

    return encoded_dataset  # type: ignore
