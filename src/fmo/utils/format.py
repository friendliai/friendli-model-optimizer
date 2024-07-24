# Copyright (c) 2024-present, FriendliAI Inc. All rights reserved.

"""FMO Formatting Utilities."""

from __future__ import annotations

from typing import NoReturn

import typer


def secho_error_and_exit(text: str, color: str = typer.colors.RED) -> NoReturn:
    """Print error and exit."""
    typer.secho(text, err=True, fg=color)
    raise typer.Exit(1)
