"""I/O helper tests: coords->PDB conversion and reading real water positions."""
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_WATER_PDB = REPO_ROOT / "examples" / "data" / "5SRF" / "5SRF_water.pdb"
EXAMPLE_WATER_MOL2 = REPO_ROOT / "examples" / "data" / "5SRF" / "5SRF_water.mol2"


def test_convert_txt_to_pdb(tmp_path):
    from superwater.inference import convert_txt_to_pdb

    txt = tmp_path / "coords.txt"
    txt.write_text("1.000 2.000 3.000\n4.000 5.000 6.000\n")
    out = tmp_path / "out.pdb"
    convert_txt_to_pdb(str(txt), str(out))

    lines = [ln for ln in out.read_text().splitlines() if ln.startswith("HETATM")]
    assert len(lines) == 2
    assert "HOH" in lines[0] and lines[0].rstrip().endswith("O")
    assert "1.000" in lines[0] and "2.000" in lines[0] and "3.000" in lines[0]


@pytest.mark.skipif(not EXAMPLE_WATER_PDB.exists(), reason="bundled example data missing")
def test_find_real_water_pos_pdb():
    from superwater.utils.find_water_pos import find_real_water_pos

    pos = find_real_water_pos(str(EXAMPLE_WATER_PDB))
    assert isinstance(pos, np.ndarray)
    assert pos.ndim == 2 and pos.shape[1] == 3
    assert len(pos) > 0


@pytest.mark.skipif(not EXAMPLE_WATER_MOL2.exists(), reason="bundled example data missing")
def test_find_real_water_pos_mol2():
    from superwater.utils.find_water_pos import find_real_water_pos

    pos = find_real_water_pos(str(EXAMPLE_WATER_MOL2))
    assert pos.ndim == 2 and pos.shape[1] == 3
    assert len(pos) > 0


def test_find_real_water_pos_rejects_unknown_extension(tmp_path):
    from superwater.utils.find_water_pos import find_real_water_pos

    bad = tmp_path / "water.xyz"
    bad.write_text("nonsense")
    with pytest.raises(ValueError):
        find_real_water_pos(str(bad))
