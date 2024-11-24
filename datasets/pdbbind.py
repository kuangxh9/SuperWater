import binascii
import glob
import hashlib
import os
import pickle
from collections import defaultdict
from multiprocessing import Pool
import random
import copy
import re
import numpy as np
import torch
from rdkit.Chem import MolToSmiles, MolFromSmiles, AddHs
from torch_geometric.data import Dataset, HeteroData
from torch_geometric.loader import DataLoader, DataListLoader
from torch_geometric.transforms import BaseTransform
from tqdm import tqdm

from datasets.process_mols import read_molecule, get_rec_graph, generate_conformer, \
    get_lig_graph_with_matching, extract_receptor_structure, parse_receptor, parse_pdb_from_path
from utils.diffusion_utils import modify_conformer, set_time
from utils.utils import read_strings_from_txt
from utils import so3, torus


class NoiseTransform(BaseTransform):
    def __init__(self, t_to_sigma, all_atom):
        self.t_to_sigma = t_to_sigma
        self.all_atom = all_atom

    def __call__(self, data):
        t = np.random.uniform()
        t_tr = t
        return self.apply_noise(data, t_tr)

    def apply_noise(self, data, t_tr, tr_update = None):
        tr_sigma = self.t_to_sigma(t_tr)
        set_time(data, t_tr, 1, self.all_atom, device=None)
        tr_update = torch.normal(mean=0, std=tr_sigma, size=data['ligand'].pos.shape) if tr_update is None else tr_update
        modify_conformer(data, tr_update)
        
        data.tr_score = -tr_update / tr_sigma ** 2
        return data


class PDBBind(Dataset):
    def __init__(self, root, transform=None, cache_path='data/cache', split_path='data/', limit_complexes=0,
                 receptor_radius=30, num_workers=1, c_alpha_max_neighbors=None, popsize=15, maxiter=15,
                 matching=False, keep_original=False, max_lig_size=None, remove_hs=False, num_conformers=1, all_atoms=False,
                 atom_radius=5, atom_max_neighbors=None, esm_embeddings_path=None, require_ligand=False,
                 ligands_list=None, protein_path_list=None, ligand_descriptions=None, keep_local_structures=False):

        super(PDBBind, self).__init__(root, transform)
        self.pdbbind_dir = root
        self.max_lig_size = max_lig_size
        self.split_path = split_path
        self.limit_complexes = limit_complexes
        self.receptor_radius = receptor_radius
        self.num_workers = num_workers
        self.c_alpha_max_neighbors = c_alpha_max_neighbors
        self.remove_hs = remove_hs
        self.esm_embeddings_path = esm_embeddings_path
        self.require_ligand = require_ligand
        self.protein_path_list = protein_path_list
        self.ligand_descriptions = ligand_descriptions
        self.keep_local_structures = keep_local_structures
        if matching or protein_path_list is not None and ligand_descriptions is not None:
            cache_path += '_torsion'
        if all_atoms:
            cache_path += '_allatoms'
        self.full_cache_path = os.path.join(cache_path, f'limit{self.limit_complexes}'
                                                        f'_INDEX{os.path.splitext(os.path.basename(self.split_path))[0]}'
                                                        f'_maxLigSize{self.max_lig_size}_H{int(not self.remove_hs)}'
                                                        f'_recRad{self.receptor_radius}_recMax{self.c_alpha_max_neighbors}'
                                            + ('' if not all_atoms else f'_atomRad{atom_radius}_atomMax{atom_max_neighbors}')
                                            + ('' if not matching or num_conformers == 1 else f'_confs{num_conformers}')
                                            + ('' if self.esm_embeddings_path is None else f'_esmEmbeddings')
                                            + ('' if not keep_local_structures else f'_keptLocalStruct')
                                            + ('' if protein_path_list is None or ligand_descriptions is None else str(binascii.crc32(''.join(ligand_descriptions + protein_path_list).encode()))))
        self.popsize, self.maxiter = popsize, maxiter
        self.matching, self.keep_original = matching, keep_original
        self.num_conformers = num_conformers
        self.all_atoms = all_atoms
        self.atom_radius, self.atom_max_neighbors = atom_radius, atom_max_neighbors
        if not os.path.exists(os.path.join(self.full_cache_path, "done.txt")):
            os.makedirs(self.full_cache_path, exist_ok=True)
            self.preprocessing()
            
        self.files = [f for f in os.listdir(self.full_cache_path) if f.endswith('.pt')]

    def len(self):
        return len(self.files)

    def get(self, idx):
        file_path = os.path.join(self.full_cache_path, self.files[idx])
        data = torch.load(file_path)
        return data

    def preprocessing(self):
        print(f'Processing complexes from [{self.split_path}] and saving it to [{self.full_cache_path}]')

        complex_names_all = read_strings_from_txt(self.split_path)
        if self.limit_complexes is not None and self.limit_complexes != 0:
            complex_names_all = complex_names_all[:self.limit_complexes]
        print(f'Loading {len(complex_names_all)} complexes.')

        if self.num_workers > 1:
            complex_names = complex_names_all
            if self.num_workers > 1:
                p = Pool(self.num_workers, maxtasksperchild=1)
                p.__enter__()
            with tqdm(total=len(complex_names), desc=f'loading complexes') as pbar:
                map_fn = p.imap if self.num_workers > 1 else map
                for complex, lig in map_fn(self.get_complex, complex_names):
                    full_path = os.path.join(self.full_cache_path, f"{complex[0].name}.pt")
                    torch.save(complex[0], full_path)
                    pbar.update()
            if self.num_workers > 1: p.__exit__(None, None, None)
        else:
            complex_names = complex_names_all

            with tqdm(total=len(complex_names), desc=f'loading complexes') as pbar:
                for complex, lig in map(self.get_complex, complex_names):
                    full_path = os.path.join(self.full_cache_path, f"{complex[0].name}.pt")
                    torch.save(complex[0], full_path)
                    pbar.update()
        with open(os.path.join(self.full_cache_path, 'done.txt'), 'w') as file:
            file.write("done")
        
    def find_lm_embeddings_chains(self, base_name):
        pattern = f"{self.esm_embeddings_path}/{base_name}_chain_*.pt"
    
        file_list = glob.glob(pattern)
        file_list.sort(key=lambda x: int(re.search(r"_chain_(\d+)\.pt$", x).group(1)))
        
        lm_embeddings_chains = [torch.load(filename)['representations'][33] for filename in file_list]
        return lm_embeddings_chains
    
    def get_complex(self, name):
        lm_embedding_chains = self.find_lm_embeddings_chains(name)
        if not os.path.exists(os.path.join(self.pdbbind_dir, name)):
            print("Folder not found", name)
            return [], []
        try:
            rec_model, rec_lig_model = parse_receptor(name, self.pdbbind_dir)
        except Exception as e:
            print(f'Skipping {name} because of the error:')
            print(e)
            return [], []
        
        ligs = read_mols(self.pdbbind_dir, name, remove_hs=False)

        complex_graphs = []
        failed_indices = []
        for i, lig in enumerate(ligs):
            if self.max_lig_size is not None and lig.GetNumHeavyAtoms() > self.max_lig_size:
                print(f'Ligand with {lig.GetNumHeavyAtoms()} heavy atoms is larger than max_lig_size {self.max_lig_size}. Not including {name} in preprocessed data.')
                continue
            complex_graph = HeteroData()
            complex_graph['name'] = f"{name}"
            
            try:
                get_lig_graph_with_matching(lig, complex_graph, self.popsize, self.maxiter, self.matching, self.keep_original,
                                            self.num_conformers, remove_hs=self.remove_hs)

                rec, rec_lig, rec_coords, all_coords, c_alpha_coords, n_coords, c_coords, lm_embeddings = extract_receptor_structure(copy.deepcopy(rec_model), copy.deepcopy(rec_lig_model), lig, lm_embedding_chains=lm_embedding_chains)

                if lm_embeddings is not None and len(c_alpha_coords) != len(lm_embeddings):
                    print(f'LM embeddings for complex {name} did not have the right length for the protein. Skipping {name}.')
                    failed_indices.append(i)
                    continue

                get_rec_graph(rec, rec_lig, rec_coords, all_coords, c_alpha_coords, n_coords, c_coords, complex_graph, rec_radius=self.receptor_radius,
                              c_alpha_max_neighbors=self.c_alpha_max_neighbors, all_atoms=self.all_atoms,
                              atom_radius=self.atom_radius, atom_max_neighbors=self.atom_max_neighbors, remove_hs=self.remove_hs, lm_embeddings=lm_embeddings)

            except Exception as e:
                print(f'Skipping {name} because of the error:')
                print(e)
                failed_indices.append(i)
                continue

            protein_center = torch.mean(complex_graph['receptor'].pos, dim=0, keepdim=True)
            complex_graph['receptor'].pos -= protein_center
            if self.all_atoms:
                complex_graph['atom'].pos -= protein_center

            if (not self.matching) or self.num_conformers == 1:
                complex_graph['ligand'].pos -= protein_center
            else:
                for p in complex_graph['ligand'].pos:
                    p -= protein_center

            complex_graph.original_center = protein_center
            complex_graphs.append(complex_graph)
        for idx_to_delete in sorted(failed_indices, reverse=True):
            del ligs[idx_to_delete]
        
        return complex_graphs, ligs

def print_statistics(complex_graphs):
    statistics = ([], [], [], [])

    for complex_graph in complex_graphs:
        lig_pos = complex_graph['ligand'].pos if torch.is_tensor(complex_graph['ligand'].pos) else complex_graph['ligand'].pos[0]
        radius_protein = torch.max(torch.linalg.vector_norm(complex_graph['receptor'].pos, dim=1))
        molecule_center = torch.mean(lig_pos, dim=0)
        radius_molecule = torch.max(
            torch.linalg.vector_norm(lig_pos - molecule_center.unsqueeze(0), dim=1))
        distance_center = torch.linalg.vector_norm(molecule_center)
        statistics[0].append(radius_protein)
        statistics[1].append(radius_molecule)
        statistics[2].append(distance_center)
        if "rmsd_matching" in complex_graph:
            statistics[3].append(complex_graph.rmsd_matching)
        else:
            statistics[3].append(0)

    name = ['radius protein', 'radius molecule', 'distance protein-mol', 'rmsd matching']
    print('Number of complexes: ', len(complex_graphs))
    for i in range(4):
        array = np.asarray(statistics[i])
        print(f"{name[i]}: mean {np.mean(array)}, std {np.std(array)}, max {np.max(array)}")


def construct_loader(args, t_to_sigma):
    transform = NoiseTransform(t_to_sigma=t_to_sigma, all_atom=args.all_atoms)
    
    common_args = {'transform': transform, 'root': args.data_dir, 'limit_complexes': args.limit_complexes,
                   'receptor_radius': args.receptor_radius,
                   'c_alpha_max_neighbors': args.c_alpha_max_neighbors,
                   'remove_hs': args.remove_hs, 'max_lig_size': args.max_lig_size,
                   'popsize': args.matching_popsize, 'maxiter': args.matching_maxiter,
                   'num_workers': args.num_workers, 'all_atoms': args.all_atoms,
                   'atom_radius': args.atom_radius, 'atom_max_neighbors': args.atom_max_neighbors,
                   'esm_embeddings_path': args.esm_embeddings_path}    
    train_dataset = PDBBind(cache_path=args.cache_path, split_path=args.split_train, keep_original=True,
                            num_conformers=args.num_conformers, **common_args)
    val_dataset = PDBBind(cache_path=args.cache_path, split_path=args.split_val, keep_original=True, **common_args)

    loader_class = DataListLoader if torch.cuda.is_available() else DataLoader
    train_loader = loader_class(dataset=train_dataset, batch_size=args.batch_size, num_workers=args.num_dataloader_workers, shuffle=True, pin_memory=args.pin_memory)
    val_loader = loader_class(dataset=val_dataset, batch_size=args.batch_size, num_workers=args.num_dataloader_workers, shuffle=True, pin_memory=args.pin_memory)
    infer_loader = loader_class(dataset=val_dataset, batch_size=1, num_workers=args.num_dataloader_workers, shuffle=False, pin_memory=args.pin_memory)

    return train_loader, val_loader, infer_loader


def read_mol(pdbbind_dir, name, remove_hs=False):
    lig = read_molecule(os.path.join(pdbbind_dir, name, f'{name}_water.mol2'), remove_hs=remove_hs, sanitize=True)
    return lig


def read_mols(pdbbind_dir, name, remove_hs=False):
    ligs = []
    for file in os.listdir(os.path.join(pdbbind_dir, name)):
        if file.endswith(".mol2"):
            lig = read_molecule(os.path.join(pdbbind_dir, name, file[:-5] + ".mol2"), remove_hs=remove_hs, sanitize=True)
            if lig is not None:
                ligs.append(lig)
    return ligs