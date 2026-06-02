"""Build a FASTA file of per-chain protein sequences for an organized dataset.

Used by the batch ESM workflow: this FASTA is fed to Meta's esm/scripts/extract.py
to produce per-residue embeddings. For one-off prediction, prefer the in-process
``superwater.embed`` / ``superwater.predict`` path (no cloned ESM repo needed).
"""
import os
from argparse import ArgumentParser

from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
from tqdm import tqdm

from superwater.esm_embeddings import get_chain_sequences


def main(argv=None):
    parser = ArgumentParser()
    parser.add_argument('--out_file', type=str, default="data/prepared_for_esm_water.fasta")
    parser.add_argument('--data_dir', type=str, default='data/waterbind/', help='Organized dataset dir')
    args = parser.parse_args(argv)

    records = []
    for name in tqdm(os.listdir(args.data_dir)):
        if name == '.DS_Store':
            continue
        processed = os.path.join(args.data_dir, name, f'{name}_protein_processed.pdb')
        rec_path = processed if os.path.exists(processed) else os.path.join(args.data_dir, name, f'{name}_protein.pdb')
        for i, seq in enumerate(get_chain_sequences(rec_path)):
            # extract.py truncates IDs to 4 chars, so the FASTA label uses name[:4].
            record = SeqRecord(Seq(seq), f'{name[:4]}_chain_{i}')
            record.description = ''
            records.append(record)
    SeqIO.write(records, args.out_file, "fasta")


if __name__ == '__main__':
    main()
