"""Tests for superwater-predict terminal-output cleanliness and verbosity.

The heavy pipeline (model loading, ESM, inference) is mocked so these run fast on CPU
and exercise only the CLI output formatting / verbosity behavior.
"""
import os

import torch

from superwater import predict as P

NOISE = "esm_embeddings_path: x\ncache path is y\ncommon t schedule [1. 0.95]\nHAPPENING | loading\n100%|x| 1/1 [00:07<00:00, 7.0s/it]"


def _setup(monkeypatch, tmp_path, names, fail=False):
    sdir = tmp_path / "structs"
    sdir.mkdir()
    for nm in names:
        (sdir / f"{nm}.pdb").write_text("ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\n")
    out = tmp_path / "preds"
    # Fake (but existing) model dirs so the isdir validation passes; loading is mocked.
    score_dir = tmp_path / "models" / "score"
    conf_dir = tmp_path / "models" / "conf"
    score_dir.mkdir(parents=True)
    conf_dir.mkdir(parents=True)
    cfg = tmp_path / "c.yaml"
    cfg.write_text(
        f"input:\n  structure_dir: {sdir}\n"
        f"models:\n  score_model_dir: {score_dir}\n  confidence_model_dir: {conf_dir}\n"
        f"output:\n  output_dir: {out}\n  overwrite: true\n  format: pdb\n  include_protein: false\n"
        f"prediction:\n  save_filtered: false\n")

    monkeypatch.setattr(P, "_load_models", lambda *a, **k: (object(), object()))
    monkeypatch.setattr(P, "load_esm_model", lambda *a, **k: (object(), object()))
    monkeypatch.setattr(P, "embed_complex", lambda *a, **k: None)
    monkeypatch.setattr(P, "set_seed", lambda *a, **k: None)

    import warnings

    def fake_run_inference(args, save_pos_path, device, **kw):
        print(NOISE)  # simulate noisy lower-level stdout + tqdm
        warnings.warn("DataParallel is usually much slower than DistributedDataParallel", UserWarning)
        if fail:
            raise RuntimeError("simulated failure: could not parse structure")
        os.makedirs(save_pos_path, exist_ok=True)
        nm = os.path.splitext(os.path.basename(args.split_test))[0]
        with open(os.path.join(save_pos_path, f"{nm}_centroid.txt"), "w") as f:
            f.write("1 2 3\n4 5 6\n")  # 2 waters
        open(os.path.join(save_pos_path, f"{nm}_centroid.pdb"), "w").close()
        return "graph_cache"

    monkeypatch.setattr(P, "run_inference", fake_run_inference)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda *a, **k: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda *a, **k: 6_700_000_000)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda *a, **k: None)
    monkeypatch.chdir(tmp_path)
    return str(cfg)


def _run(monkeypatch, cfg, *flags):
    code = 0
    try:
        P.main(["--config", cfg, *flags])
    except SystemExit as e:
        code = e.code or 0
    return code


NOISY_STRINGS = ["FutureWarning", "DataParallel", "common t schedule", "esm_embeddings_path",
                 "cache path is", "HAPPENING", "it/s"]


def test_default_output_is_clean(tmp_path, monkeypatch, capsys):
    cfg = _setup(monkeypatch, tmp_path, ["5SRF"])
    _run(monkeypatch, cfg)
    cap = capsys.readouterr()
    combined = cap.out + cap.err

    for noisy in NOISY_STRINGS:
        assert noisy not in combined, f"noisy string leaked: {noisy!r}"

    assert "SuperWater prediction" in cap.out
    assert "Input structures: 1" in cap.out
    assert "Output directory:" in cap.out
    assert "Output format: pdb" in cap.out
    assert "Include protein: no" in cap.out
    assert "Predicting 5SRF" in cap.out
    assert "2 waters" in cap.out
    assert "peak GPU" in cap.out
    assert "Summary:" in cap.out
    assert "succeeded: 5SRF" in cap.out


def test_batch_output_and_summary(tmp_path, monkeypatch, capsys):
    cfg = _setup(monkeypatch, tmp_path, ["A", "B", "C"])
    _run(monkeypatch, cfg)
    out = capsys.readouterr().out
    assert "Input structures: 3" in out
    assert "[1/3] A" in out and "[2/3] B" in out and "[3/3] C" in out
    assert "succeeded: A, B, C" in out
    assert "skipped: none" in out
    assert "failed: none" in out


def test_verbose_exposes_lowlevel_details(tmp_path, monkeypatch, capsys):
    cfg = _setup(monkeypatch, tmp_path, ["5SRF"])
    _run(monkeypatch, cfg, "--verbose")
    out = capsys.readouterr().out
    assert "common t schedule" in out          # lower-level stdout shown in verbose
    assert "esm_embeddings_path" in out


def test_failure_is_concise_by_default(tmp_path, monkeypatch, capsys):
    cfg = _setup(monkeypatch, tmp_path, ["5SRF"], fail=True)
    code = _run(monkeypatch, cfg)
    cap = capsys.readouterr()
    combined = cap.out + cap.err
    assert "FAILED" in combined
    assert "simulated failure" in combined
    assert "Traceback" not in combined         # no traceback in default mode
    assert code == 1
    assert "failed: 5SRF" in cap.out


def test_failure_shows_traceback_in_debug(tmp_path, monkeypatch, capsys):
    cfg = _setup(monkeypatch, tmp_path, ["5SRF"], fail=True)
    code = _run(monkeypatch, cfg, "--debug")
    cap = capsys.readouterr()
    assert "Traceback" in (cap.out + cap.err)   # full traceback in debug mode
    assert code == 1
