"""Integration smoke test for the `superwater-organize` entry point.

Exercises the real organize_dataset.main() against a tiny synthetic dataset.
"""
import sys

from superwater.organize_dataset import main as organize_main


def test_organize_dataset_end_to_end(tmp_path, monkeypatch):
    # Lay out: <tmp>/data/raw/<id>.pdb  and  <tmp>/data/dummy/_water.{pdb,mol2}
    data = tmp_path / "data"
    raw = data / "raw"
    dummy = data / "dummy"
    raw.mkdir(parents=True)
    dummy.mkdir(parents=True)

    (raw / "1abc.pdb").write_text("ATOM      1  CA  ALA A   1       0.000   0.000   0.000\n")
    (dummy / "_water.pdb").write_text("HETATM    1  O   HOH A   1       0.000   0.000   0.000\n")
    (dummy / "_water.mol2").write_text("@<TRIPOS>MOLECULE\nwater\n")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", [
        "prog",
        "--raw_data", "raw",
        "--data_root", "data",
        "--output_dir", "org_out",
        "--splits_path", "data/splits",
        "--dummy_water_dir", "data/dummy",
        "--logs_dir", "logs",
    ])

    organize_main()

    out_folder = data / "org_out" / "1abc"
    assert (out_folder / "1abc_protein_processed.pdb").exists()
    assert (out_folder / "1abc_water.pdb").exists()
    assert (out_folder / "1abc_water.mol2").exists()

    split_file = data / "splits" / "org_out.txt"
    assert split_file.exists()
    assert "1abc" in split_file.read_text().split()
