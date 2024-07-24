# Copyright (c) 2024-present, FriendliAI Inc. All rights reserved.
# pylint: disable=no-value-for-parameter, no-name-in-module

"""FMO-CLI Dataset Utils."""
from __future__ import annotations

import os
from typing import Optional

import datasets  # type: ignore
from fmo_core import NotSupportedError, QuantizationError  # type: ignore
from transformers import AutoTokenizer, PreTrainedTokenizer  # type: ignore


def safe_load_datasets(
    dataset_name_or_path: str,
    split_name: Optional[str],
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
                dataset = datasets.load_dataset(dataset_name_or_path, split=split_name)
            elif len(data_name_parts) == 2:
                data_name, subset_name = data_name_parts
                dataset = datasets.load_dataset(
                    data_name, subset_name, split=split_name
                )
            else:
                raise QuantizationError(
                    "Dataset name is in invalid format. "
                    "(valid format: '<dataset_name>' or '<dataset_name>:<subset_name>')"
                )
    except ValueError as err:
        raise QuantizationError(f"datasets.load_dataset failed. {str(err)}") from err

    if not isinstance(dataset, datasets.Dataset):
        raise QuantizationError(
            "This dataset format is not supported for the calibration."
        )

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
    except OSError as exc:
        raise NotSupportedError(str(exc)) from exc

    if not tokenizer.is_fast:
        raise NotSupportedError(
            "This model does not support Friendli-compatible tokenizer"
        )

    if tokenizer.pad_token != "<unk>":
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer
