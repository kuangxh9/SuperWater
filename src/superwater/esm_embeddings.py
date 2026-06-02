"""In-process generation of ESM-2 per-residue embeddings.

The score/confidence models consume per-residue ESM-2 (``esm2_t33_650M_UR50D``,
layer 33, 1280-dim) embeddings, one ``<name>_chain_<i>.pt`` file per protein chain,
each holding ``{'representations': {33: tensor[L, 1280]}}`` -- the same format that
Meta's ``esm/scripts/extract.py`` produces. Generating them here (via the ``fair-esm``
package) removes the need to clone that repo for a single prediction.
"""
import os

import torch
from Bio.PDB import PDBParser

MODEL_NAME = "esm2_t33_650M_UR50D"
REPR_LAYER = 33
EMBED_DIM = 1280
# Matches the README/extract.py setting used to train the shipped models.
TRUNCATION_SEQ_LENGTH = 4096

# MSE is selenomethionine: chemically almost identical to MET (S replaced by Se).
THREE_TO_ONE = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q', 'GLU': 'E',
    'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'MSE': 'M',
    'PHE': 'F', 'PRO': 'P', 'PYL': 'O', 'SER': 'S', 'SEC': 'U', 'THR': 'T', 'TRP': 'W',
    'TYR': 'Y', 'VAL': 'V', 'ASX': 'B', 'GLX': 'Z', 'XAA': 'X', 'XLE': 'J',
}


def get_chain_sequences(pdb_path):
    """Return one amino-acid sequence per chain, in PDB chain order.

    A residue is treated as an amino acid only if it has the CA/N/C backbone atoms;
    waters and other heteroatoms are skipped. Empty strings are kept for non-protein
    chains so that chain indices stay aligned with the receptor graph builder.
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_path)[0]
    sequences = []
    for chain in structure:
        seq = ''
        for residue in chain:
            if residue.get_resname() == 'HOH':
                continue
            atom_names = {atom.name for atom in residue}
            if {'CA', 'N', 'C'} <= atom_names:
                seq += THREE_TO_ONE.get(residue.get_resname(), '-')
        sequences.append(seq)
    return sequences


def load_esm_model(device):
    """Load (and cache to ~/.cache/torch) the ESM-2 650M model and its alphabet."""
    import esm
    model, alphabet = esm.pretrained.load_model_and_alphabet(MODEL_NAME)
    return model.to(device).eval(), alphabet


def _embed_sequence(seq, model, alphabet, device, truncation=TRUNCATION_SEQ_LENGTH):
    batch_converter = alphabet.get_batch_converter(truncation)
    _, _, tokens = batch_converter([("protein", seq)])
    tokens = tokens.to(device)
    with torch.inference_mode():
        out = model(tokens, repr_layers=[REPR_LAYER], return_contacts=False)
    length = min(truncation, len(seq))
    # Drop the leading BOS token; keep one representation per residue.
    return out["representations"][REPR_LAYER][0, 1:length + 1].cpu().clone()


def embed_complex(name, pdb_path, out_dir, model, alphabet, device, truncation=TRUNCATION_SEQ_LENGTH):
    """Write ``<name>_chain_<i>.pt`` embedding files for every chain of one protein."""
    os.makedirs(out_dir, exist_ok=True)
    for i, seq in enumerate(get_chain_sequences(pdb_path)):
        emb = _embed_sequence(seq, model, alphabet, device, truncation) if seq else torch.zeros(0, EMBED_DIM)
        torch.save({'representations': {REPR_LAYER: emb}}, os.path.join(out_dir, f"{name}_chain_{i}.pt"))


def embed_dataset(data_dir, out_dir, device, truncation=TRUNCATION_SEQ_LENGTH):
    """Embed every complex in an organized dataset dir (each subfolder is one complex)."""
    model, alphabet = load_esm_model(device)
    names = sorted(d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d)))
    for name in names:
        pdb_path = os.path.join(data_dir, name, f"{name}_protein_processed.pdb")
        if not os.path.exists(pdb_path):
            print(f"Skipping {name}: {pdb_path} not found")
            continue
        print(f"Embedding {name} ...")
        embed_complex(name, pdb_path, out_dir, model, alphabet, device, truncation)
