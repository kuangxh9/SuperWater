"""Tests for structure-format helpers (water PDB/CIF writers, CIF->PDB input conversion)."""
from pathlib import Path

import pytest

from superwater import structure_io as sio

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_PDB = REPO_ROOT / "examples" / "data" / "batch_structures" / "5SRF.pdb"


def _count_atoms(path):
    from Bio.PDB import PDBParser
    from Bio.PDB.MMCIFParser import MMCIFParser
    parser = MMCIFParser(QUIET=True) if str(path).endswith(".cif") else PDBParser(QUIET=True)
    return sum(1 for _ in parser.get_structure("s", str(path)).get_atoms())


def test_supported_exts():
    assert ".pdb" in sio.SUPPORTED_STRUCTURE_EXTS
    assert ".cif" in sio.SUPPORTED_STRUCTURE_EXTS
    assert ".mmcif" in sio.SUPPORTED_STRUCTURE_EXTS


def test_convert_txt_to_pdb(tmp_path):
    txt = tmp_path / "c.txt"
    txt.write_text("1.000 2.000 3.000\n4.000 5.000 6.000\n")
    out = tmp_path / "w.pdb"
    sio.convert_txt_to_pdb(str(txt), str(out))
    lines = [ln for ln in out.read_text().splitlines() if ln.startswith("HETATM")]
    assert len(lines) == 2
    assert "HOH" in lines[0] and lines[0].rstrip().endswith("O")
    # waters must be distinct residues so a strict parser preserves the count
    assert _count_atoms(out) == 2


def test_convert_txt_to_cif_preserves_count(tmp_path):
    txt = tmp_path / "c.txt"
    txt.write_text("1.0 2.0 3.0\n4.0 5.0 6.0\n7.0 8.0 9.0\n")
    out = tmp_path / "w.cif"
    sio.convert_txt_to_cif(str(txt), str(out))
    assert _count_atoms(out) == 3


def test_write_water_structure_dispatch(tmp_path):
    txt = tmp_path / "c.txt"
    txt.write_text("1.0 2.0 3.0\n")
    sio.write_water_structure(str(txt), str(tmp_path / "a.pdb"), "pdb")
    sio.write_water_structure(str(txt), str(tmp_path / "a.cif"), "cif")
    assert (tmp_path / "a.pdb").exists() and (tmp_path / "a.cif").exists()
    with pytest.raises(ValueError):
        sio.write_water_structure(str(txt), str(tmp_path / "a.xyz"), "xyz")


def test_to_input_pdb_copies_pdb(tmp_path):
    src = tmp_path / "x.pdb"
    src.write_text("ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\n")
    dst = tmp_path / "out.pdb"
    sio.to_input_pdb(str(src), str(dst))
    assert dst.read_text() == src.read_text()


def test_to_input_pdb_rejects_unsupported(tmp_path):
    bad = tmp_path / "x.xyz"
    bad.write_text("nonsense")
    with pytest.raises(ValueError):
        sio.to_input_pdb(str(bad), str(tmp_path / "out.pdb"))


@pytest.mark.skipif(not EXAMPLE_PDB.exists(), reason="bundled example missing")
@pytest.mark.parametrize("fmt", ["pdb", "cif"])
def test_write_protein_with_waters_adds_waters(tmp_path, fmt):
    n_prot = _count_atoms(EXAMPLE_PDB)
    txt = tmp_path / "c.txt"
    txt.write_text("1.0 2.0 3.0\n4.0 5.0 6.0\n7.0 8.0 9.0\n")
    out = tmp_path / f"pw.{fmt}"
    sio.write_protein_with_waters(str(EXAMPLE_PDB), str(txt), str(out), fmt)

    from Bio.PDB import PDBParser
    from Bio.PDB.MMCIFParser import MMCIFParser
    parser = MMCIFParser(QUIET=True) if fmt == "cif" else PDBParser(QUIET=True)
    structure = parser.get_structure("s", str(out))
    n_hoh = sum(1 for r in structure.get_residues() if r.resname == "HOH")
    assert _count_atoms(out) == n_prot + 3   # protein atoms + 3 waters
    assert n_hoh == 3                         # waters present as distinct HOH residues


@pytest.mark.skipif(not EXAMPLE_PDB.exists(), reason="bundled example missing")
def test_cif_to_pdb_roundtrip_preserves_atoms(tmp_path):
    # Derive a .cif from the bundled protein, then convert it back to .pdb.
    from Bio.PDB import PDBParser
    from Bio.PDB.mmcifio import MMCIFIO
    structure = PDBParser(QUIET=True).get_structure("p", str(EXAMPLE_PDB))
    n_src = sum(1 for _ in structure.get_atoms())
    cif = tmp_path / "p.cif"
    io = MMCIFIO(); io.set_structure(structure); io.save(str(cif))
    out_pdb = tmp_path / "p.pdb"
    sio.to_input_pdb(str(cif), str(out_pdb))
    assert _count_atoms(out_pdb) == n_src
