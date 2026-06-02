"""Argument-parser tests, including the new --wandb_entity / --save_pos_path options."""
import sys

import pytest

from superwater.utils.parsing import (
    parse_train_args,
    parse_confidence_args,
    parse_inference_args,
)


@pytest.fixture
def no_cli_args(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["prog"])


def test_train_args_defaults(no_cli_args):
    args = parse_train_args()
    assert hasattr(args, "wandb_entity")
    assert args.wandb_entity is None  # no longer a hardcoded username
    assert args.batch_size == 32


def test_confidence_args_defaults(no_cli_args):
    args = parse_confidence_args()
    assert hasattr(args, "wandb_entity")
    assert args.wandb_entity is None


def test_inference_args_defaults(no_cli_args):
    args = parse_inference_args()
    assert hasattr(args, "wandb_entity")
    assert args.wandb_entity is None
    # the option the web app relies on must exist and default to None
    assert hasattr(args, "save_pos_path")
    assert args.save_pos_path is None


def test_inference_save_pos_path_override(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["prog", "--save_pos_path", "my_run", "--cap", "0.2"])
    args = parse_inference_args()
    assert args.save_pos_path == "my_run"
    assert args.cap == 0.2
