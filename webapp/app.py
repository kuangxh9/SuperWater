import os
import shutil
import subprocess
from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    send_file,
    session,
    Response,
)
import traceback

app = Flask(__name__, static_folder="static", template_folder="templates")
app.secret_key = "SESSION_DUMMY_KEY"

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/download-demo/<filename>")
def download_file(filename):
    return send_file(
        os.path.join(app.static_folder, "files", filename), as_attachment=True
    )

@app.route("/inference")
def inference():
    error = session.pop("error", None)
    output = session.pop("output", None)
    inference_done = session.pop("inference_done", False)
    return render_template(
        "inference.html", error=error, output=output, inference_done=inference_done
    )

@app.route("/upload_single", methods=["POST"])
def upload_single():
    file = request.files.get("single_pdb")
    if not file:
        session["error"] = "File cannot be empty."
        session["output"] = ""
        session["inference_done"] = False
        return redirect(url_for("inference"))

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_root = os.path.join(script_dir, "..", "data")  # e.g. parent dir + data
    upload_dir = os.path.join(data_root, "web_upload_data")

    if os.path.exists(upload_dir):
        for f in os.listdir(upload_dir):
            os.remove(os.path.join(upload_dir, f))
    else:
        os.makedirs(upload_dir, exist_ok=True)

    file_path = os.path.join(upload_dir, file.filename)
    file.save(file_path)

    cmd = [
        "python",
        os.path.join(script_dir, "..", "organize_pdb_dataset.py"),
        "--raw_data",
        "web_upload_data",
        "--data_root",
        "data",
        "--output_dir",
        "web_upload_data_organized",
        "--splits_path",
        "data/splits",
        "--dummy_water_dir",
        "data/dummy_water",
        "--logs_dir",
        "logs",
    ]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
        cwd=os.path.join(script_dir, ".."),
    )

    if result.returncode != 0:
        session["error"] = result.stderr
        session["output"] = result.stdout
        session["inference_done"] = False
    else:
        session["error"] = None
        session["output"] = result.stdout
        session["inference_done"] = False

    return redirect(url_for("inference"))


@app.route("/upload_folder", methods=["POST"])
def upload_folder():
    files = request.files.getlist("pdb_folder")
    if not files or len(files) == 0:
        session["error"] = "No folder or PDB files selected."
        session["output"] = ""
        session["inference_done"] = False
        return redirect(url_for("inference"))

    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_path = os.path.join(script_dir, "..", "data")
    upload_dir = os.path.join(base_path, "web_upload_data")

    if os.path.exists(upload_dir):
        for f in os.listdir(upload_dir):
            os.remove(os.path.join(upload_dir, f))
    else:
        os.makedirs(upload_dir, exist_ok=True)

    saved_any_file = False
    for file in files:
        if file and file.filename.endswith(".pdb"):
            file_path = os.path.join(upload_dir, os.path.basename(file.filename))
            file.save(file_path)
            saved_any_file = True

    if not saved_any_file:
        session["error"] = "No .pdb files found in selected folder."
        session["output"] = ""
        session["inference_done"] = False
        return redirect(url_for("inference"))

    cmd = [
        "python",
        os.path.join(script_dir, "..", "organize_pdb_dataset.py"),
        "--raw_data",
        "web_upload_data",
        "--data_root",
        "data",
        "--output_dir",
        "web_upload_data_organized",
        "--splits_path",
        "data/splits",
        "--dummy_water_dir",
        "data/dummy_water",
        "--logs_dir",
        "logs",
    ]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
        cwd=os.path.join(script_dir, ".."),
    )
    if result.returncode != 0:
        session["error"] = result.stderr
        session["output"] = result.stdout
        session["inference_done"] = False
    else:
        session["error"] = None
        session["output"] = result.stdout
        session["inference_done"] = False

    return redirect(url_for("inference"))


@app.route("/start_inference", methods=["POST"])
def start_inference():
    water_ratio = request.form.get("water_ratio", "1")
    filter_threshold = request.form.get("filter_threshold", "0.1")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.join(script_dir, "..")
    esm_dir = os.path.join(script_dir, "..", "data")

    data_dir = os.path.join(parent_dir, "data", "web_upload_data_organized")
    if not os.path.isdir(data_dir) or not os.listdir(data_dir):
        session[
            "error"
        ] = "Please upload files first! (web_upload_data_organized is empty)"
        session["output"] = ""
        session["inference_done"] = False
        return redirect(url_for("inference"))

    # 1) ESM embeddings
    cmd1 = [
        "python",
        os.path.join(
            script_dir, "..", "datasets", "esm_embedding_preparation_water.py"
        ),
        "--data_dir",
        "data/web_upload_data_organized",
        "--out_file",
        "data/prepared_for_esm_web_upload_data_organized.fasta",
    ]
    result1 = subprocess.run(
        cmd1, capture_output=True, text=True, check=False, cwd=parent_dir
    )
    if result1.returncode != 0:
        session["error"] = result1.stderr
        session["output"] = result1.stdout
        session["inference_done"] = False
        return redirect(url_for("inference"))

    # 2) ESM Feature
    cmd2 = [
        "python",
        os.path.join(script_dir, "..", "esm", "scripts", "extract.py"),
        "esm2_t33_650M_UR50D",
        "prepared_for_esm_web_upload_data_organized.fasta",
        "web_upload_data_organized_embeddings_output",
        "--repr_layers",
        "33",
        "--include",
        "per_tok",
        "--truncation_seq_length",
        "4096",
    ]
    result2 = subprocess.run(
        cmd2, capture_output=True, text=True, check=False, cwd=esm_dir
    )
    if result2.returncode != 0:
        session["error"] = result2.stderr
        session["output"] = result2.stdout
        session["inference_done"] = False
        return redirect(url_for("inference"))

    pred_dir_name = f"web_prediction_rr{water_ratio}_cap{filter_threshold}"
    cmd3 = [
        "python",
        "-m",
        "inference_water_pos",
        "--original_model_dir",
        "workdir/all_atoms_score_model_res15_17092",
        "--confidence_dir",
        "workdir/confidence_model_17092_sigmoid_rr15",
        "--data_dir",
        "data/web_upload_data_organized",
        "--ckpt",
        "best_model.pt",
        "--all_atoms",
        "--cache_path",
        "data/cache_confidence",
        "--save_pos_path",
        pred_dir_name,
        "--split_test",
        "data/splits/web_upload_data_organized.txt",
        "--inference_steps",
        "20",
        "--esm_embeddings_path",
        "data/web_upload_data_organized_embeddings_output",
        "--cap",
        filter_threshold,
        "--running_mode",
        "test",
        "--mad_prediction",
        "--save_pos",
        "--water_ratio",
        water_ratio,
    ]
    result3 = subprocess.run(
        cmd3, capture_output=True, text=True, check=False, cwd=parent_dir
    )
    if result3.returncode != 0:
        session["error"] = result3.stderr
        session["output"] = result3.stdout
        session["inference_done"] = False
        return redirect(url_for("inference"))

    session["error"] = None
    session["output"] = result3.stdout
    session["inference_done"] = True
    session["pred_dir"] = pred_dir_name

    return redirect(url_for("inference"))


@app.route("/download_prediction")
def download_prediction():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.join(script_dir, "..")

    pred_dir_name = session.get("pred_dir", None)
    if not pred_dir_name:
        session["error"] = "No inference result found. Please run inference first."
        session["output"] = ""
        return redirect(url_for("inference"))

    pred_dir = os.path.join(parent_dir, "inference_out", pred_dir_name)
    if not os.path.isdir(pred_dir):
        session[
            "error"
        ] = f"Prediction folder {pred_dir_name} not found in inference_out/"
        session["output"] = ""
        return redirect(url_for("inference"))

    zip_basename = os.path.join(parent_dir, "inference_out", pred_dir_name)
    zip_path = zip_basename + ".zip"
    if os.path.exists(zip_path):
        os.remove(zip_path)

    shutil.make_archive(base_name=zip_basename, format="zip", root_dir=pred_dir)

    session["error"] = None
    session[
        "output"
    ] = "File is generated. Please click 'Cleanup' after your download if you wish to remove files."
    session["inference_done"] = False

    return send_file(zip_path, as_attachment=True)


@app.route("/cleanup_prediction")
def cleanup_prediction():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.join(script_dir, "..")

    # 1) data/splits/web_upload_data_organized.txt
    try:
        os.remove(
            os.path.join(parent_dir, "data", "splits", "web_upload_data_organized.txt")
        )
    except:
        pass

    # 2) data/web_upload_data_organized
    try:
        shutil.rmtree(os.path.join(parent_dir, "data", "web_upload_data_organized"))
    except:
        pass

    # 3) data/web_upload_data
    try:
        shutil.rmtree(os.path.join(parent_dir, "data", "web_upload_data"))
    except:
        pass

    # 4) data/web_upload_data_organized_embeddings_output
    try:
        shutil.rmtree(
            os.path.join(
                parent_dir, "data", "web_upload_data_organized_embeddings_output"
            )
        )
    except:
        pass

    # 5) data/prepared_for_esm_web_upload_data_organized.fasta
    try:
        os.remove(
            os.path.join(
                parent_dir, "data", "prepared_for_esm_web_upload_data_organized.fasta"
            )
        )
    except:
        pass

    # 6) confidence cache
    try:
        shutil.rmtree(
            os.path.join(
                parent_dir,
                "data",
                "cache_confidence",
                "model_all_atoms_score_model_res15_17092_split_web_upload_data_organized_limit_0",
            )
        )
    except:
        pass

    # 7) all atom cache
    try:
        shutil.rmtree(
            os.path.join(
                parent_dir,
                "data",
                "cache_allatoms",
                "limit0_INDEXweb_upload_data_organized_maxLigSizeNone_H0_recRad15.0_recMax24_atomRad5_atomMax8_esmEmbeddings",
            )
        )
    except:
        pass

    # 8) remove inference_out/pred_dir if needed
    pred_dir_name = session.get("pred_dir", None)
    if pred_dir_name:
        pred_dir = os.path.join(parent_dir, "inference_out", pred_dir_name)
        try:
            shutil.rmtree(pred_dir)
        except:
            pass
        # remove the zip if it still exists
        zip_path = pred_dir + ".zip"
        try:
            os.remove(zip_path)
        except:
            pass

    session["error"] = None
    session["output"] = "Cleanup done. All files are removed!"
    return redirect(url_for("inference"))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8891, debug=True)
