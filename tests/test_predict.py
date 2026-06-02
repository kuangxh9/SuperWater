"""Tests for the predict entry point: config parsing, folder discovery, input
resolution (new + legacy schema), arg mapping, and validation errors.

These do NOT run a full prediction (which needs a GPU and downloads the ESM model);
the end-to-end batch runs are exercised separately as manual smoke tests.
"""
from pathlib import Path

import pytest

from superwater import predict

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_CONFIG = REPO_ROOT / "examples" / "configs" / "predict_5srf.yaml"


def test_example_config_parses_with_folder_schema():
    assert EXAMPLE_CONFIG.exists()
    cfg = predict.load_config(str(EXAMPLE_CONFIG))
    assert cfg["input"]["structure_dir"] == "examples/data/batch_structures"
    assert cfg["output"]["format"] == "pdb"
    assert cfg["models"]["score_model_dir"] == "models/water_score_res15"
    assert cfg["prediction"]["confidence_cutoff"] > 0
    # example defaults: protein included in the output, filtered text off
    assert cfg["output"]["include_protein"] is True
    assert cfg["prediction"]["save_filtered"] is False
    assert cfg["runtime"]["cleanup_intermediates"] is False
    assert cfg["runtime"]["keep_embeddings"] is True
    assert cfg["runtime"]["keep_graph_cache"] is True


def _make_work_tree(tmp_path):
    """Create a fake per-run work tree + graph cache for cleanup tests."""
    work = tmp_path / "work" / "X"
    (work / "data" / "X").mkdir(parents=True)
    (work / "cache").mkdir()
    (work / "embeddings").mkdir()
    (work / "X.txt").write_text("X\n")
    (work / "embeddings" / "X_chain_0.pt").write_text("emb")
    graph = tmp_path / "cache_allatoms" / "limit0_INDEXX_esmEmbeddings"
    graph.mkdir(parents=True)
    (graph / "X.pt").write_text("graph")
    return work, graph


def test_cleanup_keeps_reusable_caches_by_default(tmp_path):
    work, graph = _make_work_tree(tmp_path)
    predict._cleanup_run(str(work), str(graph), keep_embeddings=True, keep_graph_cache=True)
    assert not (work / "data").exists()          # per-run work removed
    assert not (work / "cache").exists()
    assert not (work / "X.txt").exists()
    assert (work / "embeddings").exists()         # reusable: kept
    assert graph.exists()                          # reusable: kept


def test_cleanup_can_remove_embeddings_and_graph_cache(tmp_path):
    work, graph = _make_work_tree(tmp_path)
    predict._cleanup_run(str(work), str(graph), keep_embeddings=False, keep_graph_cache=False)
    assert not work.exists()                       # whole work dir gone (now empty)
    assert not graph.exists()                      # graph cache removed on request


def test_example_structure_dir_exists_and_has_inputs():
    cfg = predict.load_config(str(EXAMPLE_CONFIG))
    structures, skipped = predict.discover_structures(str(REPO_ROOT / cfg["input"]["structure_dir"]))
    assert len(structures) >= 1
    assert all(p.endswith((".pdb", ".cif", ".mmcif")) for _, p in structures)


def test_discover_structures_filters_and_dedupes(tmp_path):
    for fn in ["A.pdb", "B.pdb", "C.cif", "D.mmcif", "A.cif", "notes.txt", "data.csv"]:
        (tmp_path / fn).write_text("x")
    (tmp_path / "subdir").mkdir()
    structures, skipped = predict.discover_structures(str(tmp_path))
    names = sorted(n for n, _ in structures)
    assert names == ["A", "B", "C", "D"]          # A.cif deduped (A.pdb wins)
    assert sorted(skipped) == ["data.csv", "notes.txt"]


def test_resolve_inputs_folder_schema(tmp_path):
    (tmp_path / "X.pdb").write_text("x")
    (tmp_path / "Y.pdb").write_text("y")
    structures, per_subdir = predict._resolve_inputs({"structure_dir": str(tmp_path)})
    assert per_subdir is True
    assert sorted(n for n, _ in structures) == ["X", "Y"]


def test_resolve_inputs_legacy_single_file(tmp_path):
    pdb = tmp_path / "prot.pdb"
    pdb.write_text("ATOM\n")
    structures, per_subdir = predict._resolve_inputs({"protein_pdb": str(pdb), "name": "myprot"})
    assert per_subdir is False                     # legacy writes directly into output_dir
    assert structures == [("myprot", str(pdb))]


def test_resolve_inputs_missing_raises(tmp_path):
    with pytest.raises(SystemExit):
        predict._resolve_inputs({})
    with pytest.raises(SystemExit):
        predict._resolve_inputs({"structure_dir": str(tmp_path / "nope")})


def test_resolve_inputs_only_unsupported_files_raises(tmp_path):
    (tmp_path / "a.txt").write_text("x")
    with pytest.raises(SystemExit):
        predict._resolve_inputs({"structure_dir": str(tmp_path)})


def test_build_inference_args_maps_yaml_fields():
    cfg = {"prediction": {"water_ratio": 7, "inference_steps": 11,
                          "confidence_cutoff": 0.25, "batch_size": 3}}
    args = predict.build_inference_args(
        cfg, data_dir="d", emb_dir="e", split_path="s.txt",
        score_dir="models/water_score_res15",
        conf_dir="models/water_confidence_res15_sigmoid", cache_path="c")
    assert (args.water_ratio, args.inference_steps, args.cap, args.batch_size) == (7, 11, 0.25, 3)
    assert args.all_atoms is True and args.save_pos is True and args.running_mode == "test"
    assert args.esm_embeddings_path == "e"


def test_load_config_missing_file_exits():
    with pytest.raises(SystemExit):
        predict.load_config("/no/such/config.yaml")


def test_main_bad_output_format_exits(tmp_path):
    sdir = tmp_path / "structs"; sdir.mkdir()
    (sdir / "A.pdb").write_text("ATOM\n")
    cfg = tmp_path / "c.yaml"
    cfg.write_text(
        f"input:\n  structure_dir: {sdir}\n"
        f"output:\n  output_dir: {tmp_path / 'out'}\n  format: xyz\n")
    with pytest.raises(SystemExit):
        predict.main(["--config", str(cfg)])
