# Copyright (c) 2024-present, FriendliAI Inc. All rights reserved.
# pylint: disable=line-too-long, too-many-arguments, no-name-in-module
"""Frienli-Model-Optimizer (FMO) CLI."""

from __future__ import annotations

import os
from enum import Enum
from typing import Optional

import typer

from fmo.utils.dataset import get_calib_dataloader, get_tokenizer, safe_load_datasets
from fmo.utils.format import secho_error_and_exit
from fmo.utils.version import get_installed_version

app = typer.Typer(
    help="Friendli Model Optimizer 🚀",
    no_args_is_help=True,
    context_settings={"help_option_names": ["-h", "--help"]},
    add_completion=True,
    pretty_exceptions_enable=False,
)


class QuantMode(str, Enum):
    """Quant Mode."""

    INT8 = "int8"
    FP8 = "fp8"


@app.command()
def version():
    """Check the installed package version."""
    installed_version = get_installed_version()
    typer.echo(installed_version)


@app.command()
def quantize(
    model_name_or_path: str = typer.Option(
        ...,
        "--model-name-or-path",
        "-m",
        help="Hugging Face pretrained model name or directory path of the saved model checkpoint.",
    ),
    output_dir: str = typer.Option(
        ...,
        "--output-dir",
        "-o",
        help=(
            "Directory path to save the quantized checkpoint and related configurations "
            "files."
        ),
    ),
    mode: QuantMode = typer.Option(
        ...,
        "--mode",
        help=("Qantization techniques to apply. You can use `fp8`, and `int8`."),
    ),
    pedantic_level: int = typer.Option(
        1,
        "--pedantic-level",
        help=(
            "Higher pedantic level ensure a more accurate representation of the model,"
            "but increase the quantization processing time. Lower levels allow for faster"
            "quantization, but may sacrifice some model accuracy. Defaults to 1."
        ),
    ),
    device: Optional[str] = typer.Option(
        "cuda:0",
        "--device",
        help=(
            "The device which is used for quantization process. Currently we use only one GPU."
        ),
    ),
    offload: Optional[bool] = typer.Option(
        False,
        "--offload",
        help=(
            "When enabled, significantly reduces GPU memory usage by offloading model layers onto CPU RAM. Defaults to true."
        ),
    ),
    seed: Optional[int] = typer.Option(
        42, "--seed", help=("Seed for dataset sampling.")
    ),
    dataset_name_or_path: str = typer.Option(
        "abisee/cnn_dailymail:3.0.0",
        "--dataset-name-or-path",
        help=(
            "Huggingface dataset name or directory path for gathering sample activations."
        ),
    ),
    dataset_split_name: str = typer.Option(
        "test",
        "--dataset-split-name",
        help=("Huggingface dataset split name for gathering sample activations."),
    ),
    dataset_target_column_name: str = typer.Option(
        "article",
        "--dataset-target-column-name",
        help=("Huggingface dataset column name for gathering sample activations."),
    ),
    dataset_num_samples: int = typer.Option(
        512,
        "--dataset-num-samples",
        help=("The number of samples for gathering sample activations."),
    ),
    dataset_max_length: int = typer.Option(
        512,
        "--dataset-max-length",
        help=(
            "The maximum legth of sample in the dataset for gathering sample activations."
        ),
    ),
    dataset_batch_size: int = typer.Option(
        32,
        "--dataset-batch-size",
        help=("The batch size of the dataset for gathering sample activations."),
    ),
    cache_dir: Optional[str] = typer.Option(
        None, "--cache-dir", help="Directory path of the cached model checkpoint."
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Only check quantization avaliability."
    ),
):
    """Quantize huggingface's model."""
    # pylint: disable=too-many-locals, import-outside-toplevel
    from fmo_core import quantize  # type: ignore

    # pylint: enable=import-outside-toplevel

    if not os.path.isdir(output_dir):
        if os.path.exists(output_dir):
            secho_error_and_exit(f"'{output_dir}' exists, but it is not a directory.")
        os.mkdir(output_dir)

    dataset = safe_load_datasets(
        dataset_name_or_path=dataset_name_or_path, split_name=dataset_split_name
    )
    tokenizer = get_tokenizer(
        model_name_or_path=model_name_or_path, cache_dir=cache_dir
    )
    calib_dataloader = get_calib_dataloader(
        dataset=dataset,
        lookup_column_name=dataset_target_column_name,
        max_length=dataset_max_length,
        num_samples=dataset_num_samples,
        batch_size=dataset_batch_size,
        seed=seed,
        tokenizer=tokenizer,
    )

    quantize(
        model_name_or_path,
        mode,
        dry_run=dry_run,
        save_dir=output_dir,
        cache_dir=cache_dir,
        device=device,
        offload=offload,
        calib_dataloader=calib_dataloader,
        pedantic_level=pedantic_level,
    )

    msg = (
        f"Checkpoint({model_name_or_path}) can be converted."
        if dry_run
        else f"Checkpoint({model_name_or_path}) has been converted successfully."
    )
    typer.secho(msg)
