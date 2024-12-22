import itertools
import math
import os
import pickle
import random
from argparse import Namespace
from functools import partial
import copy
from scipy.spatial import cKDTree

import numpy as np
import pandas as pd
import torch
import yaml
from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from datasets.pdbbind import PDBBind
from utils.diffusion_utils import get_t_schedule
from utils.sampling import randomize_position, sampling
from utils.utils import get_model
from utils.diffusion_utils import t_to_sigma as t_to_sigma_compl
from utils.sampling import sampling, randomize_position_multiple, larger_original_graph, sampling_test1
from utils.nearest_point_dist import get_nearest_point_distances
from utils.find_water_pos import find_real_water_pos


class ListDataset(Dataset):
    def __init__(self, list):
        super().__init__()
        self.data_list = list

    def len(self) -> int:
        return len(self.data_list)

    def get(self, idx: int) -> Data:
        return self.data_list[idx]

def get_cache_path(args, split):
    cache_path = args.cache_path
    cache_path += '_torsion'
    if args.all_atoms:
        cache_path += '_allatoms'
    split_path = args.split_train if split == 'train' else args.split_val
    cache_path = os.path.join(cache_path, f'limit{args.limit_complexes}_INDEX{os.path.splitext(os.path.basename(split_path))[0]}_maxLigSize{args.max_lig_size}_H{int(not args.remove_hs)}_recRad{args.receptor_radius}_recMax{args.c_alpha_max_neighbors}'
                                       + ('' if not args.all_atoms else f'_atomRad{args.atom_radius}_atomMax{args.atom_max_neighbors}')
    
                              + ('' if args.esm_embeddings_path is None else f'_esmEmbeddings'))
    return cache_path


def get_args(original_model_dir):
    with open(f'{original_model_dir}/model_parameters.yml') as f:
        model_args = Namespace(**yaml.full_load(f))
    return model_args

def find_tp_coords(real_water_pos, predicted_water_pos, threshold=1.0):
    tree = cKDTree(real_water_pos)
    indices = tree.query_ball_point(predicted_water_pos, r=threshold)
    tp_points = predicted_water_pos[np.array([len(i) > 0 for i in indices])]
    return tp_points

class ConfidenceDataset(Dataset):
    def __init__(self, loader, cache_path, original_model_dir, split, device, limit_complexes,
                 inference_steps, samples_per_complex, all_atoms,
                 args, model_ckpt, balance=False, use_original_model_cache=True, mad_classification_cutoff=2,
                 cache_ids_to_combine=None, cache_creation_id=None, running_mode=None, water_ratio=15, resample_steps=1):

        super(ConfidenceDataset, self).__init__()
        self.loader = loader
        self.device = device
        self.inference_steps = inference_steps
        self.limit_complexes = limit_complexes
        self.all_atoms = all_atoms
        self.original_model_dir = original_model_dir
        self.balance = balance
        self.use_original_model_cache = use_original_model_cache
        self.mad_classification_cutoff = mad_classification_cutoff
        self.cache_ids_to_combine = cache_ids_to_combine
        self.cache_creation_id = cache_creation_id
        self.samples_per_complex = samples_per_complex
        self.model_ckpt = model_ckpt
        self.args = args
        
        self.running_mode = running_mode
        self.water_ratio = water_ratio
        self.resample_steps = resample_steps
        
        self.original_model_args, original_model_cache = get_args(original_model_dir), self.loader.dataset.full_cache_path
        
        # check if the docked positions have already been computed, if not run the preprocessing (docking every complex)
        self.full_cache_path = os.path.join(cache_path, f'model_{os.path.splitext(os.path.basename(original_model_dir))[0]}'
                                            f'_split_{split}_limit_{limit_complexes}')
        print("cache path is ", self.full_cache_path)
        if (not os.path.exists(os.path.join(self.full_cache_path, "water_positions.pkl")) and self.cache_creation_id is None) or \
            (not os.path.exists(os.path.join(self.full_cache_path, f"water_positions_id{self.cache_creation_id}.pkl")) and self.cache_creation_id is not None):
            os.makedirs(self.full_cache_path, exist_ok=True)
            self.preprocessing(original_model_cache)
        
        all_mads_unsorted, all_full_water_positions_unsorted, all_names_unsorted = [], [], []
        for idx, cache_id in enumerate(self.cache_ids_to_combine):
            print(f'HAPPENING | Loading positions and MADs from cache_id from the path: {os.path.join(self.full_cache_path, "water_positions_"+ str(cache_id)+ ".pkl")}')
            if not os.path.exists(os.path.join(self.full_cache_path, f"water_positions_id{cache_id}.pkl")): raise Exception(f'The generated water positions with cache_id do not exist: {cache_id}') # be careful with changing this error message since it is sometimes cought in a try catch
            with open(os.path.join(self.full_cache_path, f"water_positions_id{cache_id}.pkl"), 'rb') as f:
                full_water_positions, mads = pickle.load(f)
            with open(os.path.join(self.full_cache_path, f"complex_names_in_same_order_id{cache_id}.pkl"), 'rb') as f:
                names_unsorted = pickle.load(f)
            all_names_unsorted.append(names_unsorted)
            all_mads_unsorted.append(mads)
            all_full_water_positions_unsorted.append(full_water_positions)

        names_order = list(set(sum(all_names_unsorted, [])))
        all_mads, all_full_water_positions, all_names = [], [], []
        for idx, (mads_unsorted, full_water_positions_unsorted, names_unsorted) in enumerate(zip(all_mads_unsorted,all_full_water_positions_unsorted, all_names_unsorted)):
            name_to_pos_dict = {name: (mad, pos) for name, mad, pos in zip(names_unsorted, full_water_positions_unsorted, mads_unsorted) }
            intermediate_mads = [name_to_pos_dict[name][1] for name in names_order]
            all_mads.append((intermediate_mads))
            intermediate_pos = [name_to_pos_dict[name][0] for name in names_order]
            all_full_water_positions.append((intermediate_pos))
            
        self.full_water_positions, self.mads = [], []
        for positions_tuple in list(zip(*all_full_water_positions)):
            self.full_water_positions.append(np.concatenate(positions_tuple, axis=0))
        for positions_tuple in list(zip(*all_mads)):
            self.mads.append(np.concatenate(positions_tuple, axis=0))
        generated_mad_complex_names = names_order
        
        print('Number of complex graphs: ', len(self.loader.dataset))
            
        print('Number of MADs and positions for the complex graphs: ', len(self.full_water_positions))

        self.all_samples_per_complex = samples_per_complex * (1 if self.cache_ids_to_combine is None else len(self.cache_ids_to_combine))

        self.positions_mads_dict = {name: (pos, mad) for name, pos, mad in zip (generated_mad_complex_names, self.full_water_positions, self.mads)}
        self.dataset_names = list(self.positions_mads_dict.keys())
        if limit_complexes > 0:
            self.dataset_names = self.dataset_names[:limit_complexes]

    def len(self):
        return len(self.dataset_names)

    def get(self, idx):
        complex_name = self.dataset_names[idx]
        complex_graph = torch.load(os.path.join(self.loader.dataset.full_cache_path, f"{complex_name}.pt"))
        positions, mads = self.positions_mads_dict[self.dataset_names[idx]]
        
        assert(complex_graph.name == self.dataset_names[idx])
        complex_graph['ligand'].x =  complex_graph['ligand'].x[-1].repeat(positions.shape[-2], 1)
        if self.balance:
            if isinstance(self.mad_classification_cutoff, list): raise ValueError("a list for --mad_classification_cutoff can only be used without --balance")
            label = random.randint(0, 1)
            success = mads < self.mad_classification_cutoff
            n_success = np.count_nonzero(success)
            if label == 0 and n_success != self.all_samples_per_complex:
                # sample negative complexpr
                sample = random.randint(0, self.all_samples_per_complex - n_success - 1)
                lig_pos = positions[~success][sample]
                complex_graph['ligand'].pos = torch.from_numpy(lig_pos)
            else:
                # sample positive complex
                if n_success > 0: # if no successfull sample returns the matched complex
                    sample = random.randint(0, n_success - 1)
                    lig_pos = positions[success][sample]
                    complex_graph['ligand'].pos = torch.from_numpy(lig_pos)
            complex_graph.y = torch.tensor(label).float()
        else:
            sample = random.randint(0, self.all_samples_per_complex - 1)
            
            complex_graph['ligand'].pos = torch.from_numpy(positions[sample])
            complex_graph.y = torch.tensor(mads < self.mad_classification_cutoff).float()
                
            if isinstance(self.mad_classification_cutoff, list):
                complex_graph.y_binned = torch.tensor(np.logical_and(mads[sample] < self.mad_classification_cutoff + [math.inf],mads[sample] >= [0] + self.mad_classification_cutoff), dtype=torch.float).unsqueeze(0)
                complex_graph.y = torch.tensor(mads[sample] < self.mad_classification_cutoff[0]).unsqueeze(0).float()
            
            complex_graph.mad = torch.tensor(mads).float()

        complex_graph['ligand'].node_t = {'tr': 0 * torch.ones(complex_graph['ligand'].num_nodes)}
        complex_graph['receptor'].node_t = {'tr': 0 * torch.ones(complex_graph['receptor'].num_nodes)}
        if self.all_atoms:
            complex_graph['atom'].node_t = {'tr': 0 * torch.ones(complex_graph['atom'].num_nodes)}
        complex_graph.complex_t = {'tr': 0 * torch.ones(1)}
        return complex_graph

    def preprocessing(self, original_model_cache):
        t_to_sigma = partial(t_to_sigma_compl, args=self.original_model_args)
        
        model = get_model(self.original_model_args, self.device, t_to_sigma=t_to_sigma, no_parallel=True)
        state_dict = torch.load(f'{self.original_model_dir}/{self.model_ckpt}', map_location=torch.device('cpu'))
        model.load_state_dict(state_dict, strict=True)
        model = model.to(self.device)
        model.eval()
        
        tr_schedule = get_t_schedule(inference_steps=self.inference_steps)
        
        print('Running mode: ', self.running_mode)

        if self.running_mode == "train":
            water_ratio = self.water_ratio
            resample_steps = self.resample_steps
        elif self.running_mode == "test":
            water_ratio = self.water_ratio
            resample_steps = self.resample_steps
        else:
            raise ValueError("Invalid running mode!")
        total_resample_ratio = water_ratio * resample_steps
        print('common t schedule', tr_schedule)
        print('water_number/residue_number ratio: ', water_ratio)
        print('resampling steps: ', resample_steps)
        print('total resampling ratio: ', total_resample_ratio)
        
        mads, full_water_positions, names = [], [], []
        for idx, orig_complex_graph in tqdm(enumerate(self.loader)):
            data_list = [copy.deepcopy(orig_complex_graph) for _ in range(self.samples_per_complex)]
            res_num = int(orig_complex_graph[0]['receptor'].pos.shape[0])
            step_num_water = int(res_num * water_ratio)
            total_num_water = int(res_num * total_resample_ratio)
            total_sampled_water = 0
        
            prediction_list = []
            confidence_list = []
            
            for i in range(resample_steps):
                sample_data_list = copy.deepcopy(data_list)
                randomize_position_multiple(sample_data_list, False, self.original_model_args.tr_sigma_max, water_num=step_num_water)

                predictions, confidences = sampling_test1(data_list=sample_data_list, model=model,
                                                    inference_steps=self.inference_steps,
                                                    tr_schedule=tr_schedule,
                                                    device=self.device, t_to_sigma=t_to_sigma, model_args=self.original_model_args)
                prediction_list.append(predictions)
                confidence_list.append(confidences)
                
             
            orig_complex_graph['ligand'].orig_pos = (orig_complex_graph['ligand'].pos.cpu().numpy() + orig_complex_graph.original_center.cpu().numpy())
            orig_water_pos = np.expand_dims(orig_complex_graph['ligand'].orig_pos - orig_complex_graph.original_center.cpu().numpy(), axis=0)

            if isinstance(orig_complex_graph['ligand'].orig_pos, list):
                orig_complex_graph['ligand'].orig_pos = orig_complex_graph['ligand'].orig_pos[0]
                
            real_water_pos = find_real_water_pos(os.path.join(self.args.data_dir, f"{orig_complex_graph.name[0]}/{orig_complex_graph.name[0]}_water.pdb"))
                   
            water_pos_list = []
            for complex_graphs in prediction_list:
                for complex_graph in complex_graphs:
                    water_pos_list.append(complex_graph['ligand'].pos.cpu().numpy())
            
            all_water_pos = np.concatenate(water_pos_list, axis=0)
            water_pos = np.asarray([all_water_pos], dtype=np.float32)
            
            
            positions_new = water_pos.squeeze(0) + orig_complex_graph.original_center.cpu().numpy()
            mad, indices = get_nearest_point_distances(positions_new, real_water_pos)
            
            mads.append(mad)
            full_water_positions.append(water_pos)
    
            names.append(orig_complex_graph.name[0])
            assert(len(orig_complex_graph.name) == 1) # I just put this assert here because of the above line where I assumed that the list is always only lenght 1. Just in case it isn't maybe check what the names in there are.
        with open(os.path.join(self.full_cache_path, f"water_positions{'' if self.cache_creation_id is None else '_id' + str(self.cache_creation_id)}.pkl"), 'wb') as f:
            pickle.dump((full_water_positions, mads), f)
        with open(os.path.join(self.full_cache_path, f"complex_names_in_same_order{'' if self.cache_creation_id is None else '_id' + str(self.cache_creation_id)}.pkl"), 'wb') as f:
            pickle.dump((names), f)


