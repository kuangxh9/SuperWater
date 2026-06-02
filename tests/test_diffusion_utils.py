"""Unit tests for the diffusion schedule / embedding helpers."""
from argparse import Namespace

import numpy as np
import pytest
import torch

from superwater.utils.diffusion_utils import (
    t_to_sigma,
    get_t_schedule,
    sinusoidal_embedding,
    get_timestep_embedding,
)


def test_t_to_sigma_endpoints_and_monotonicity():
    args = Namespace(tr_sigma_min=0.1, tr_sigma_max=30.0)
    assert t_to_sigma(0.0, args) == pytest.approx(0.1)
    assert t_to_sigma(1.0, args) == pytest.approx(30.0)
    # geometric interpolation is monotonically increasing in t
    sigmas = [t_to_sigma(t, args) for t in np.linspace(0, 1, 11)]
    assert all(b > a for a, b in zip(sigmas, sigmas[1:]))


def test_get_t_schedule_shape_and_range():
    sched = get_t_schedule(inference_steps=20)
    assert len(sched) == 20
    assert sched[0] == pytest.approx(1.0)
    assert sched[-1] > 0  # last step before 0 is dropped
    assert np.all(np.diff(sched) < 0)  # strictly decreasing


def test_sinusoidal_embedding_shape():
    t = torch.linspace(0, 1, 8)
    emb = sinusoidal_embedding(t, embedding_dim=32)
    assert emb.shape == (8, 32)


def test_get_timestep_embedding_unknown_type_raises():
    with pytest.raises(NotImplementedError):
        get_timestep_embedding("not-a-real-type", embedding_dim=16)
