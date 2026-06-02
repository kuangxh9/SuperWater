"""One-command water prediction from a YAML config.

    python -m superwater.predict --config examples/configs/predict_5srf.yaml
    superwater-predict --config examples/configs/predict_5srf.yaml

``input.structure_dir`` points at a folder of structures (.pdb/.cif/.mmcif); every
supported file is predicted and its outputs are written to a per-structure subfolder
under ``output.output_dir``. For each structure the pipeline organizes the input,
generates ESM-2 embeddings in-process (no cloned ESM repo needed), samples water
positions with the score model, scores them with the confidence model, clusters, and
writes final outputs as PDB (default) or mmCIF. The score, confidence and ESM models
are loaded once and reused across the whole batch.
"""
import glob
import os
import shutil
import sys
import time
from argparse import ArgumentParser
from functools import partial

import torch
import yaml

from superwater.paths import DUMMY_WATER_DIR
from superwater.utils.utils import resolve_model_dir, get_model
from superwater.utils.parsing import parse_inference_args
from superwater.utils.diffusion_utils import t_to_sigma as t_to_sigma_compl
from superwater.esm_embeddings import embed_complex, load_esm_model
from superwater.confidence.dataset import get_args
from superwater.inference import run_inference, set_seed, load_confidence_model
from superwater.structure_io import to_input_pdb, write_protein_with_waters, SUPPORTED_STRUCTURE_EXTS

DEFAULT_SCORE_MODEL = "models/water_score_res15"
DEFAULT_CONFIDENCE_MODEL = "models/water_confidence_res15_sigmoid"


def _fail(msg):
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def load_config(path):
    if not os.path.exists(path):
        _fail(f"Config file not found: {path}")
    with open(path) as f:
        return yaml.safe_load(f) or {}


def discover_structures(structure_dir):
    """Return (structures, skipped) for ``structure_dir``.

    ``structures`` is a list of (name, path) for supported files (name = filename stem),
    sorted by name and de-duplicated by stem; ``skipped`` lists unsupported filenames.
    """
    structures, skipped, seen = [], [], set()
    for fname in sorted(os.listdir(structure_dir)):
        path = os.path.join(structure_dir, fname)
        if not os.path.isfile(path):
            continue
        name, ext = os.path.splitext(fname)
        if ext.lower() not in SUPPORTED_STRUCTURE_EXTS:
            skipped.append(fname)
        elif name in seen:
            print(f"  Skipping {fname}: duplicate structure name '{name}'")
        else:
            seen.add(name)
            structures.append((name, path))
    return structures, skipped


def build_inference_args(cfg, data_dir, emb_dir, split_path, score_dir, conf_dir, cache_path):
    """Translate the prediction YAML into the Namespace that run_inference() expects."""
    pred = cfg.get('prediction', {})
    args = parse_inference_args([])  # defaults, then override
    args.original_model_dir = score_dir
    args.confidence_dir = conf_dir
    args.data_dir = data_dir
    args.split_test = split_path
    args.esm_embeddings_path = emb_dir
    args.cache_path = cache_path
    args.all_atoms = True
    args.ckpt = 'best_model.pt'
    args.running_mode = 'test'
    args.mad_prediction = True
    args.save_pos = True
    args.water_ratio = int(pred.get('water_ratio', 10))
    args.inference_steps = int(pred.get('inference_steps', 20))
    args.cap = float(pred.get('confidence_cutoff', 0.1))
    args.batch_size = int(pred.get('batch_size', 1))
    return args


def _load_models(score_dir, conf_dir, device):
    """Load the confidence and score models once, for reuse across the whole batch."""
    confidence_model = load_confidence_model(conf_dir, device)
    score_args = get_args(score_dir)
    score_model = get_model(score_args, device, t_to_sigma=partial(t_to_sigma_compl, args=score_args),
                            no_parallel=True)
    score_model.load_state_dict(torch.load(f"{score_dir}/best_model.pt", map_location="cpu"), strict=True)
    score_model.eval()
    return score_model, confidence_model


def _cleanup_run(work_root, graph_cache_path, keep_embeddings, keep_graph_cache):
    """Remove this run's intermediate files.

    Always removes cheap per-run work (organized input, split file, score-sampling cache).
    Embeddings and the PyG graph cache are expensive and reusable, so they are kept unless
    explicitly disabled.
    """
    for sub in ("data", "cache"):
        shutil.rmtree(os.path.join(work_root, sub), ignore_errors=True)
    for txt in glob.glob(os.path.join(work_root, "*.txt")):
        try:
            os.remove(txt)
        except OSError:
            pass
    if not keep_embeddings:
        shutil.rmtree(os.path.join(work_root, "embeddings"), ignore_errors=True)
    if os.path.isdir(work_root) and not os.listdir(work_root):
        try:
            os.rmdir(work_root)
        except OSError:
            pass
    if not keep_graph_cache and graph_cache_path and os.path.isdir(graph_cache_path):
        shutil.rmtree(graph_cache_path, ignore_errors=True)


def predict_one(name, src_path, out_dir, score_dir, conf_dir, device, cfg, get_esm_model, overwrite,
                score_model=None, confidence_model=None):
    """Predict waters for a single structure, writing outputs into ``out_dir``.

    The input may be .pdb/.cif/.mmcif (CIF is converted to PDB). Returns True on success.
    """
    pred = cfg.get('prediction', {})
    out_cfg = cfg.get('output', {})
    runtime = cfg.get('runtime', {})
    output_format = str(out_cfg.get('format', 'pdb')).lower()
    structure_ext = 'cif' if output_format == 'cif' else 'pdb'
    save_structure = bool(pred.get('save_structure', pred.get('save_pdb', True)))
    save_filtered = bool(pred.get('save_filtered', True))
    include_protein = bool(out_cfg.get('include_protein', False))
    cleanup = bool(runtime.get('cleanup_intermediates', False))
    keep_embeddings = bool(runtime.get('keep_embeddings', True))
    keep_graph_cache = bool(runtime.get('keep_graph_cache', True))

    work_root = os.path.join('outputs', 'work', name)
    data_dir = os.path.join(work_root, 'data')
    emb_dir = os.path.join(work_root, 'embeddings')
    cache_path = os.path.join(work_root, 'cache')
    if overwrite and os.path.isdir(cache_path):
        shutil.rmtree(cache_path)

    complex_dir = os.path.join(data_dir, name)
    os.makedirs(complex_dir, exist_ok=True)
    protein_dst = os.path.join(complex_dir, f"{name}_protein_processed.pdb")
    try:
        to_input_pdb(src_path, protein_dst)  # copies .pdb, converts .cif/.mmcif
    except Exception as e:
        print(f"  Failed to read structure {os.path.basename(src_path)}: {e}")
        return False
    for ext in ('mol2', 'pdb'):
        shutil.copy(os.path.join(DUMMY_WATER_DIR, f"_water.{ext}"), os.path.join(complex_dir, f"{name}_water.{ext}"))
    split_path = os.path.join(work_root, f"{name}.txt")
    with open(split_path, 'w') as f:
        f.write(name + "\n")

    if overwrite or not os.path.exists(os.path.join(emb_dir, f"{name}_chain_0.pt")):
        model, alphabet = get_esm_model()
        embed_complex(name, protein_dst, emb_dir, model, alphabet, device)

    inf_args = build_inference_args(cfg, data_dir, emb_dir, split_path, score_dir, conf_dir, cache_path)
    os.makedirs(out_dir, exist_ok=True)
    graph_cache_path = run_inference(inf_args, out_dir + os.sep, device, per_complex_subdir=False,
                                     save_structure=save_structure, save_filtered=save_filtered,
                                     output_format=output_format,
                                     score_model=score_model, confidence_model=confidence_model)

    # Optionally fold the protein into the final structure file (waters are written by run_inference).
    centroid_txt = os.path.join(out_dir, f"{name}_centroid.txt")
    if include_protein and save_structure and os.path.exists(centroid_txt):
        write_protein_with_waters(protein_dst, centroid_txt,
                                  os.path.join(out_dir, f"{name}_centroid.{structure_ext}"), structure_ext)

    ok = (not save_structure) or os.path.exists(os.path.join(out_dir, f"{name}_centroid.{structure_ext}"))

    if cleanup:
        _cleanup_run(work_root, graph_cache_path, keep_embeddings, keep_graph_cache)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return ok


def _resolve_inputs(inp):
    """Return (structures, per_structure_subdir) from the config's input section.

    Supports the folder schema (input.structure_dir) and, for backward compatibility,
    the old single-file schema (input.protein_pdb).
    """
    structure_dir = inp.get('structure_dir')
    if structure_dir:
        if not os.path.isdir(structure_dir):
            _fail(f"input.structure_dir not found or not a directory: {structure_dir}")
        structures, skipped = discover_structures(structure_dir)
        if skipped:
            print(f"Ignoring {len(skipped)} unsupported file(s): {', '.join(skipped)}")
        if not structures:
            _fail(f"No supported structures ({', '.join(SUPPORTED_STRUCTURE_EXTS)}) in {structure_dir}")
        override = inp.get('name')
        if override and len(structures) == 1:
            structures = [(override, structures[0][1])]
        return structures, True  # per-structure subdirs

    protein_pdb = inp.get('protein_pdb')  # legacy single-file schema
    if protein_pdb:
        print("NOTE: 'input.protein_pdb' is deprecated; prefer 'input.structure_dir'.")
        if not os.path.exists(protein_pdb):
            _fail(f"Input protein file not found: {protein_pdb}")
        name = inp.get('name') or os.path.splitext(os.path.basename(protein_pdb))[0]
        return [(name, protein_pdb)], False  # write directly into output_dir

    _fail("config requires 'input.structure_dir' (a folder of .pdb/.cif/.mmcif files)")


def main(argv=None):
    parser = ArgumentParser(description="One-command water prediction from a YAML config.")
    parser.add_argument('--config', required=True, help='Path to a prediction YAML config.')
    cli = parser.parse_args(argv)
    cfg = load_config(cli.config)

    inp = cfg.get('input', {})
    models_cfg = cfg.get('models', {})
    out_cfg = cfg.get('output', {})
    runtime = cfg.get('runtime', {})

    structures, per_structure_subdir = _resolve_inputs(inp)

    score_dir = resolve_model_dir(models_cfg.get('score_model_dir', DEFAULT_SCORE_MODEL))
    conf_dir = resolve_model_dir(models_cfg.get('confidence_model_dir', DEFAULT_CONFIDENCE_MODEL))
    if not os.path.isdir(score_dir):
        _fail(f"Score model dir not found: {score_dir}")
    if not os.path.isdir(conf_dir):
        _fail(f"Confidence model dir not found: {conf_dir}")

    output_format = str(out_cfg.get('format', 'pdb')).lower()
    if output_format not in ('pdb', 'cif'):
        _fail(f"output.format must be 'pdb' or 'cif', got: {output_format!r}")

    if str(runtime.get('device', 'cuda')).lower() == 'cpu':
        _fail("CPU prediction is not supported; an NVIDIA GPU with CUDA is required.")
    if not torch.cuda.is_available():
        _fail("CUDA is not available. SuperWater requires an NVIDIA GPU. "
              "Run `python scripts/check_gpu.py` to diagnose.")
    device = torch.device('cuda')

    set_seed(int(runtime.get('seed', 42)))

    output_base = out_cfg.get('output_dir') or os.path.join('outputs', 'predictions')
    overwrite = bool(out_cfg.get('overwrite', False))

    # Decide what to run, skipping already-present outputs when not overwriting.
    succeeded, failed, skipped, to_process = [], [], [], []
    for name, src_path in structures:
        out_dir = os.path.join(output_base, name) if per_structure_subdir else output_base
        if os.path.isdir(out_dir) and os.listdir(out_dir) and not overwrite:
            print(f"{name}: output exists, skipping (set output.overwrite: true)")
            skipped.append(name)
        else:
            to_process.append((name, src_path, out_dir))

    # Load the score + confidence models once for the batch (only if there is work to do);
    # the ESM model is loaded lazily on first embedding need.
    score_model = confidence_model = None
    if to_process:
        print("Loading score and confidence models ...")
        score_model, confidence_model = _load_models(score_dir, conf_dir, device)

    esm = {}

    def get_esm_model():
        if not esm:
            print("Loading ESM-2 model (first run downloads ~2.5GB to ~/.cache/torch)...")
            esm['m'], esm['a'] = load_esm_model(device)
        return esm['m'], esm['a']

    print(f"Predicting waters for {len(to_process)} structure(s) "
          f"-> {output_base} (format={output_format}, include_protein={bool(out_cfg.get('include_protein', False))})")
    for idx, (name, src_path, out_dir) in enumerate(to_process, start=1):
        t0 = time.time()
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        print(f"[{idx}/{len(to_process)}] {name}: predicting ...")
        try:
            ok = predict_one(name, src_path, out_dir, score_dir, conf_dir, device, cfg, get_esm_model,
                             overwrite, score_model=score_model, confidence_model=confidence_model)
        except Exception as e:
            print(f"  {name} failed: {e}")
            ok = False
        mem = f", peak GPU {torch.cuda.max_memory_allocated() / 1e9:.1f} GB" if device.type == 'cuda' else ""
        print(f"  {name}: {'done' if ok else 'FAILED'} in {time.time() - t0:.1f}s{mem}")
        (succeeded if ok else failed).append(name)

    print("\n=== Prediction summary ===")
    print(f"  succeeded: {', '.join(succeeded) or '(none)'}")
    if skipped:
        print(f"  skipped (already present): {', '.join(skipped)}")
    if failed:
        print(f"  failed: {', '.join(failed)}")
    print(f"  outputs in: {output_base}")
    if failed:
        sys.exit(1)


if __name__ == '__main__':
    main()
