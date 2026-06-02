"""Smoke tests: every public module imports cleanly and the console-script
entry points referenced in pyproject.toml exist and are callable.

This is the most important regression guard for the src/ package migration --
it exercises the full internal import graph (utils -> datasets -> models ->
confidence -> entry points).
"""
import importlib

import pytest

MODULES = [
    "superwater",
    "superwater.train",
    "superwater.inference",
    "superwater.predict",
    "superwater.embed",
    "superwater.esm_embeddings",
    "superwater.structure_io",
    "superwater.paths",
    "superwater.organize_dataset",
    "superwater.datasets.pdbbind",
    "superwater.datasets.process_mols",
    "superwater.datasets.conformer_matching",
    "superwater.datasets.esm_embedding_preparation",
    "superwater.models.score_model",
    "superwater.models.all_atom_score_model",
    "superwater.confidence.dataset",
    "superwater.confidence.train",
    "superwater.utils.parsing",
    "superwater.utils.utils",
    "superwater.utils.diffusion_utils",
    "superwater.utils.sampling",
    "superwater.utils.training",
    "superwater.utils.torsion",
    "superwater.utils.cluster_centroid",
    "superwater.utils.find_water_pos",
    "superwater.utils.nearest_point_dist",
]


@pytest.mark.parametrize("module_name", MODULES)
def test_module_imports(module_name):
    importlib.import_module(module_name)


def test_console_script_entrypoints_exist():
    """The callables named in [project.scripts] must exist."""
    from superwater.predict import main as predict_main
    from superwater.train import main_function
    from superwater.inference import main as infer_main
    from superwater.confidence.train import main as confidence_main
    from superwater.organize_dataset import main as organize_main
    from superwater.embed import main as embed_main

    for fn in (predict_main, main_function, infer_main, confidence_main, organize_main, embed_main):
        assert callable(fn)


def test_removed_dead_modules_are_gone():
    """so3/torus/geometry/visualise/seed/inference_utils were dead code and removed."""
    for dead in [
        "superwater.utils.so3",
        "superwater.utils.torus",
        "superwater.utils.geometry",
        "superwater.utils.visualise",
        "superwater.utils.seed",
        "superwater.utils.inference_utils",
        "superwater.utils.min_dist",
    ]:
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(dead)
