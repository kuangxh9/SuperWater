import os
from argparse import FileType, ArgumentParser

import numpy as np
import pandas as pd
from Bio.PDB import PDBParser
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from tqdm import tqdm
from Bio import SeqIO


parser = ArgumentParser()
parser.add_argument('--out_file', type=str, default="data/prepared_for_esm_water.fasta")
parser.add_argument('--dataset', type=str, default="waterbind")
parser.add_argument('--data_dir', type=str, default='data/waterbind/', help='')
args = parser.parse_args()


biopython_parser = PDBParser()

three_to_one = {'ALA':	'A',
'ARG':	'R',
'ASN':	'N',
'ASP':	'D',
'CYS':	'C',
'GLN':	'Q',
'GLU':	'E',
'GLY':	'G',
'HIS':	'H',
'ILE':	'I',
'LEU':	'L',
'LYS':	'K',
'MET':	'M',
'MSE':  'M', # MSE this is almost the same AA as MET. The sulfur is just replaced by Selen
'PHE':	'F',
'PRO':	'P',
'PYL':	'O',
'SER':	'S',
'SEC':	'U',
'THR':	'T',
'TRP':	'W',
'TYR':	'Y',
'VAL':	'V',
'ASX':	'B',
'GLX':	'Z',
'XAA':	'X',
'XLE':	'J'}

def get_structure_from_file(file_path):
    structure = biopython_parser.get_structure('random_id', file_path)
    structure = structure[0]
    l = []
    for i, chain in enumerate(structure):
        seq = ''
        for res_idx, residue in enumerate(chain):
            if residue.get_resname() == 'HOH':
                continue
            residue_coords = []
            c_alpha, n, c = None, None, None
            for atom in residue:
                if atom.name == 'CA':
                    c_alpha = list(atom.get_vector())
                if atom.name == 'N':
                    n = list(atom.get_vector())
                if atom.name == 'C':
                    c = list(atom.get_vector())
            if c_alpha != None and n != None and c != None:  # only append residue if it is an amino acid
                try:
                    seq += three_to_one[residue.get_resname()]
                except Exception as e:
                    seq += '-'
                    print("encountered unknown AA: ", residue.get_resname(), ' in the complex ', file_path, '. Replacing it with a dash - .')
        l.append(seq)
    return l

data_dir = args.data_dir
names = os.listdir(data_dir)

# if args.dataset == 'waterbind':
#     sequences = []
#     ids = []

#     for name in tqdm(names):
#         # if name.endswith(".pdb"):  # check for .pdb files
#             # rec_path = os.path.join(data_dir, name)
#         if name == '.DS_Store': continue
#         if os.path.exists(os.path.join(data_dir, name, f'{name}_protein_processed.pdb')):
#             rec_path = os.path.join(data_dir, name, f'{name}_protein_processed.pdb')
#         else:
#             rec_path = os.path.join(data_dir, name, f'{name}_protein.pdb')
#         l = get_structure_from_file(rec_path)
#         for i, seq in enumerate(l):
#             sequences.append(seq)
#             ids.append(f'{name[:4]}_chain_{i}')
#     records = []
#     for (index, seq) in zip(ids, sequences):
#         record = SeqRecord(Seq(seq), str(index))
#         record.description = ''
#         records.append(record)
#     SeqIO.write(records, args.out_file, "fasta")

    
    
sequences = []
ids = []

for name in tqdm(names):
    # if name.endswith(".pdb"):  # check for .pdb files
        # rec_path = os.path.join(data_dir, name)
    if name == '.DS_Store': continue
    if os.path.exists(os.path.join(data_dir, name, f'{name}_protein_processed.pdb')):
        rec_path = os.path.join(data_dir, name, f'{name}_protein_processed.pdb')
    else:
        rec_path = os.path.join(data_dir, name, f'{name}_protein.pdb')
    l = get_structure_from_file(rec_path)
    for i, seq in enumerate(l):
        sequences.append(seq)
        ids.append(f'{name[:4]}_chain_{i}')
records = []
for (index, seq) in zip(ids, sequences):
    record = SeqRecord(Seq(seq), str(index))
    record.description = ''
    records.append(record)
SeqIO.write(records, args.out_file, "fasta")