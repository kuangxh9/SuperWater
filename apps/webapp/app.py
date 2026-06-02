"""Local web UI for SuperWater batch water prediction.

Start with:
    python apps/webapp/app.py
then open http://localhost:8891/

Upload one or more .pdb/.cif/.mmcif structures, choose prediction settings, and the
app runs `superwater.predict` for the whole batch and lets you download per-structure
results (or all at once). Heavy paths (models/, outputs/, ESM) are handled internally.
"""
import os
import shutil
import subprocess

import yaml
from flask import Flask, render_template, request, redirect, url_for, send_file, session
from werkzeug.utils import secure_filename

# The app lives at apps/webapp/, so the repository root is two levels up. Prediction
# runs with cwd=REPO_ROOT and the config uses paths relative to it.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

# Absolute static/template folders so the app works regardless of the launch CWD.
app = Flask(__name__,
            static_folder=os.path.join(SCRIPT_DIR, "static"),
            template_folder=os.path.join(SCRIPT_DIR, "templates"))
app.secret_key = "SESSION_DUMMY_KEY"

SUPPORTED_EXTS = (".pdb", ".cif", ".mmcif")
UPLOAD_REL = os.path.join("data", "webapp", "uploads")
OUTPUT_REL = os.path.join("outputs", "webapp", "predictions")
CONFIG_REL = os.path.join("data", "webapp", "predict.yaml")
UPLOAD_DIR = os.path.join(REPO_ROOT, UPLOAD_REL)
OUTPUT_DIR = os.path.join(REPO_ROOT, OUTPUT_REL)


def _collect_results():
    """List per-structure prediction results currently on disk."""
    results = []
    if not os.path.isdir(OUTPUT_DIR):
        return results
    for name in sorted(os.listdir(OUTPUT_DIR)):
        d = os.path.join(OUTPUT_DIR, name)
        if not os.path.isdir(d):
            continue
        structure = next((f for ext in ("pdb", "cif")
                          for f in [f"{name}_centroid.{ext}"] if os.path.exists(os.path.join(d, f))), None)
        txt = os.path.join(d, f"{name}_centroid.txt")
        n_waters = sum(1 for _ in open(txt)) if os.path.exists(txt) else 0
        results.append({"name": name, "ok": structure is not None,
                        "n_waters": n_waters, "structure": structure})
    return results


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/download-demo/<filename>")
def download_file(filename):
    return send_file(os.path.join(app.static_folder, "files", filename), as_attachment=True)


@app.route("/inference")
def inference():
    return render_template(
        "inference.html",
        error=session.pop("error", None),
        log=session.pop("log", None),
        settings=session.get("settings", {"water_ratio": 10, "inference_steps": 20,
                                           "confidence_cutoff": 0.1, "output_format": "pdb",
                                           "overwrite": True, "include_protein": False,
                                           "cleanup_intermediates": False}),
        results=_collect_results(),
    )


@app.route("/run", methods=["POST"])
def run():
    # --- save uploaded structures ---
    files = request.files.getlist("structures")
    if os.path.isdir(UPLOAD_DIR):
        shutil.rmtree(UPLOAD_DIR)
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    saved, ignored = [], []
    for f in files:
        if not f or not f.filename:
            continue
        fname = secure_filename(os.path.basename(f.filename))
        if fname.lower().endswith(SUPPORTED_EXTS):
            f.save(os.path.join(UPLOAD_DIR, fname))
            saved.append(fname)
        else:
            ignored.append(fname)

    if not saved:
        session["error"] = "Please select at least one .pdb, .cif or .mmcif file."
        return redirect(url_for("inference"))

    # --- read settings ---
    def _num(key, default, cast):
        try:
            return cast(request.form.get(key, default))
        except (TypeError, ValueError):
            return default

    settings = {
        "water_ratio": _num("water_ratio", 10, int),
        "inference_steps": _num("inference_steps", 20, int),
        "confidence_cutoff": _num("confidence_cutoff", 0.1, float),
        "output_format": "cif" if request.form.get("output_format") == "cif" else "pdb",
        "overwrite": request.form.get("overwrite") == "on",
        "include_protein": request.form.get("include_protein") == "on",
        "cleanup_intermediates": request.form.get("cleanup_intermediates") == "on",
    }
    session["settings"] = settings

    # --- write a predict config and run the batch pipeline ---
    cfg = {
        "input": {"structure_dir": UPLOAD_REL},
        "models": {"score_model_dir": "models/water_score_res15",
                   "confidence_model_dir": "models/water_confidence_res15_sigmoid"},
        "output": {"output_dir": OUTPUT_REL, "overwrite": settings["overwrite"],
                   "format": settings["output_format"], "include_protein": settings["include_protein"]},
        "runtime": {"device": "cuda", "seed": 42,
                    "cleanup_intermediates": settings["cleanup_intermediates"],
                    "keep_embeddings": True, "keep_graph_cache": True},
        "prediction": {"water_ratio": settings["water_ratio"],
                       "inference_steps": settings["inference_steps"],
                       "confidence_cutoff": settings["confidence_cutoff"],
                       "batch_size": 1, "save_structure": True, "save_filtered": True},
    }
    os.makedirs(os.path.dirname(os.path.join(REPO_ROOT, CONFIG_REL)), exist_ok=True)
    with open(os.path.join(REPO_ROOT, CONFIG_REL), "w") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False)

    result = subprocess.run(["python", "-m", "superwater.predict", "--config", CONFIG_REL],
                            capture_output=True, text=True, check=False, cwd=REPO_ROOT)

    log = (result.stdout or "") + (("\n" + result.stderr) if result.stderr else "")
    if ignored:
        log = f"Ignored unsupported files: {', '.join(ignored)}\n" + log
    session["log"] = log
    if result.returncode != 0 and not _collect_results():
        session["error"] = "Prediction failed. See the log below."
    return redirect(url_for("inference"))


def _zip_and_send(folder, archive_basename):
    if not os.path.isdir(folder):
        session["error"] = "Result not found. Run a prediction first."
        return redirect(url_for("inference"))
    zip_path = os.path.join(REPO_ROOT, "outputs", "webapp", archive_basename)
    if os.path.exists(zip_path + ".zip"):
        os.remove(zip_path + ".zip")
    shutil.make_archive(zip_path, "zip", folder)
    return send_file(zip_path + ".zip", as_attachment=True)


@app.route("/download/<name>")
def download_one(name):
    name = secure_filename(name)
    return _zip_and_send(os.path.join(OUTPUT_DIR, name), f"{name}_results")


@app.route("/download_all")
def download_all():
    return _zip_and_send(OUTPUT_DIR, "all_results")


@app.route("/cleanup")
def cleanup():
    for path in (UPLOAD_DIR, os.path.join(REPO_ROOT, "outputs", "webapp"),
                 os.path.join(REPO_ROOT, "outputs", "work")):
        shutil.rmtree(path, ignore_errors=True)
    session.pop("log", None)
    session["error"] = None
    return redirect(url_for("inference"))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8891, debug=True)
