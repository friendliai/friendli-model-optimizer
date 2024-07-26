# Copyright (c) 2024-present, FriendliAI Inc. All rights reserved.
# pylint: disable=no-value-for-parameter, no-name-in-module

"""FMO-CLI Dataset Utils."""
from __future__ import annotations

import os
from typing import Optional

import datasets  # type: ignore
import torch
from fmo_core import NotSupportedError, QuantizationError  # type: ignore
from torch.utils.data import DataLoader
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


def get_calib_dataloader(  # pylint: disable=too-many-arguments
    tokenizer: PreTrainedTokenizer,
    dataset: datasets.Dataset,
    lookup_column_name: str,
    seed: Optional[int] = 42,
    max_length: Optional[int] = 512,
    num_samples: Optional[int] = 512,
    batch_size: Optional[int] = 32,
) -> DataLoader:
    """Return Calibration DataLoader."""
    try:
        dataset = dataset.shuffle(seed=seed).select(range(num_samples * 2))  # type: ignore
        encoded_ds_w_special_tokens = tokenizer(
            dataset[lookup_column_name][:num_samples],
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=max_length,
            add_special_tokens=True,
        ).input_ids
        encoded_ds_wo_special_tokens = tokenizer(
            dataset[lookup_column_name][num_samples:],
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

    except KeyError as exc:
        raise NotSupportedError(
            f"`{lookup_column_name}` is not valid column name in given dataset."
        ) from exc

    return DataLoader(encoded_dataset, batch_size=batch_size)  # type: ignore
