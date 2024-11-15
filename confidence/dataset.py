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
    # if not args.no_torsion:
    cache_path += '_torsion'
    if args.all_atoms:
        cache_path += '_allatoms'
    split_path = args.split_train if split == 'train' else args.split_val
    cache_path = os.path.join(cache_path, f'limit{args.limit_complexes}_INDEX{os.path.splitext(os.path.basename(split_path))[0]}_maxLigSize{args.max_lig_size}_H{int(not args.remove_hs)}_recRad{args.receptor_radius}_recMax{args.c_alpha_max_neighbors}'
                                       + ('' if not args.all_atoms else f'_atomRad{args.atom_radius}_atomMax{args.atom_max_neighbors}')
                                    #    + ('' if args.no_torsion or args.num_conformers == 1 else
                                    #        f'_confs{args.num_conformers}')
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
                 args, model_ckpt, balance=False, use_original_model_cache=True, rmsd_classification_cutoff=2,
                 cache_ids_to_combine=None, cache_creation_id=None, running_mode=None, add_perturbation=False):

        super(ConfidenceDataset, self).__init__()
        ##
        self.loader = loader
        ##
        self.device = device
        self.inference_steps = inference_steps
        self.limit_complexes = limit_complexes
        self.all_atoms = all_atoms
        self.original_model_dir = original_model_dir
        self.balance = balance
        self.use_original_model_cache = use_original_model_cache
        self.rmsd_classification_cutoff = rmsd_classification_cutoff
        self.cache_ids_to_combine = cache_ids_to_combine
        self.cache_creation_id = cache_creation_id
        self.samples_per_complex = samples_per_complex
        self.model_ckpt = model_ckpt
        self.args = args
        self.add_perturbation = add_perturbation
        
        self.running_mode = running_mode
        
        self.original_model_args, original_model_cache = get_args(original_model_dir), self.loader.dataset.full_cache_path
        
        # check if the docked positions have already been computed, if not run the preprocessing (docking every complex)
        self.full_cache_path = os.path.join(cache_path, f'model_{os.path.splitext(os.path.basename(original_model_dir))[0]}'
                                            f'_split_{split}_limit_{limit_complexes}')
        print("cache path is ", self.full_cache_path)
        if (not os.path.exists(os.path.join(self.full_cache_path, "ligand_positions.pkl")) and self.cache_creation_id is None) or \
            (not os.path.exists(os.path.join(self.full_cache_path, f"ligand_positions_id{self.cache_creation_id}.pkl")) and self.cache_creation_id is not None):
            os.makedirs(self.full_cache_path, exist_ok=True)
            self.preprocessing(original_model_cache)
        
        all_rmsds_unsorted, all_full_ligand_positions_unsorted, all_names_unsorted = [], [], []
        for idx, cache_id in enumerate(self.cache_ids_to_combine):
            print(f'HAPPENING | Loading positions and rmsds from cache_id from the path: {os.path.join(self.full_cache_path, "ligand_positions_"+ str(cache_id)+ ".pkl")}')
            if not os.path.exists(os.path.join(self.full_cache_path, f"ligand_positions_id{cache_id}.pkl")): raise Exception(f'The generated ligand positions with cache_id do not exist: {cache_id}') # be careful with changing this error message since it is sometimes cought in a try catch
            with open(os.path.join(self.full_cache_path, f"ligand_positions_id{cache_id}.pkl"), 'rb') as f:
                full_ligand_positions, rmsds = pickle.load(f)
            with open(os.path.join(self.full_cache_path, f"complex_names_in_same_order_id{cache_id}.pkl"), 'rb') as f:
                names_unsorted = pickle.load(f)
            all_names_unsorted.append(names_unsorted)
            all_rmsds_unsorted.append(rmsds)
            all_full_ligand_positions_unsorted.append(full_ligand_positions)

        names_order = list(set(sum(all_names_unsorted, [])))
        all_rmsds, all_full_ligand_positions, all_names = [], [], []
        for idx, (rmsds_unsorted, full_ligand_positions_unsorted, names_unsorted) in enumerate(zip(all_rmsds_unsorted,all_full_ligand_positions_unsorted, all_names_unsorted)):
            name_to_pos_dict = {name: (rmsd, pos) for name, rmsd, pos in zip(names_unsorted, full_ligand_positions_unsorted, rmsds_unsorted) }
            intermediate_rmsds = [name_to_pos_dict[name][1] for name in names_order]
            all_rmsds.append((intermediate_rmsds))
            intermediate_pos = [name_to_pos_dict[name][0] for name in names_order]
            all_full_ligand_positions.append((intermediate_pos))
            
        self.full_ligand_positions, self.rmsds = [], []
        for positions_tuple in list(zip(*all_full_ligand_positions)):
            self.full_ligand_positions.append(np.concatenate(positions_tuple, axis=0))
        for positions_tuple in list(zip(*all_rmsds)):
#             self.rmsds.append(np.stack(positions_tuple, axis=0))
            self.rmsds.append(np.concatenate(positions_tuple, axis=0))
        generated_rmsd_complex_names = names_order
        
        
#         print("self.rmsds: ", len(self.rmsds))
        print('Number of complex graphs: ', len(self.loader.dataset))
            
        print('Number of RMSDs and positions for the complex graphs: ', len(self.full_ligand_positions))

        self.all_samples_per_complex = samples_per_complex * (1 if self.cache_ids_to_combine is None else len(self.cache_ids_to_combine))

        self.positions_rmsds_dict = {name: (pos, rmsd) for name, pos, rmsd in zip (generated_rmsd_complex_names, self.full_ligand_positions, self.rmsds)}
        # self.dataset_names = list(set(self.positions_rmsds_dict.keys()))
        self.dataset_names = list(self.positions_rmsds_dict.keys())
        if limit_complexes > 0:
            self.dataset_names = self.dataset_names[:limit_complexes]

    def len(self):
        return len(self.dataset_names)

    def get(self, idx):
        # complex_graph = copy.deepcopy(self.complex_graph_dict[self.dataset_names[idx]])
        complex_name = self.dataset_names[idx]
        complex_graph = torch.load(os.path.join(self.loader.dataset.full_cache_path, f"{complex_name}.pt"))
        positions, rmsds = self.positions_rmsds_dict[self.dataset_names[idx]]
        
        # ## recacluateing rmsds to consider original pdb file
        # real_zinc_pos = find_real_zinc_pos(os.path.join(self.args.data_dir, f"{self.dataset_names[idx]}.pdb"))
        # positions_new = positions.squeeze(0) + complex_graph.original_center.numpy()
        # rmsds, indices = get_nearest_point_distances(positions_new, real_zinc_pos)
        # ##
        assert(complex_graph.name == self.dataset_names[idx])
        ## modify x
        complex_graph['ligand'].x =  complex_graph['ligand'].x[-1].repeat(positions.shape[-2], 1)
        ##
        if self.balance:
            if isinstance(self.rmsd_classification_cutoff, list): raise ValueError("a list for --rmsd_classification_cutoff can only be used without --balance")
            label = random.randint(0, 1)
            success = rmsds < self.rmsd_classification_cutoff
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
#             print("get: len(rmsds): ", len(rmsds))
#             print("get: len(rmsds): ", len(rmsds[sample]))
#             print("get: len(positions): ", len(positions))
#             print("get: len(positions): ", len(positions[sample]))
            
            complex_graph['ligand'].pos = torch.from_numpy(positions[sample])
            # complex_graph.y = torch.tensor(rmsds < self.rmsd_classification_cutoff).float().unsqueeze(0)
            complex_graph.y = torch.tensor(rmsds < self.rmsd_classification_cutoff).float()
            
            #testcode
#             complex_graph.y = torch.tensor(rmsds[sample] < self.rmsd_classification_cutoff).float().unsqueeze(0)
                
            if isinstance(self.rmsd_classification_cutoff, list):
                complex_graph.y_binned = torch.tensor(np.logical_and(rmsds[sample] < self.rmsd_classification_cutoff + [math.inf],rmsds[sample] >= [0] + self.rmsd_classification_cutoff), dtype=torch.float).unsqueeze(0)
                complex_graph.y = torch.tensor(rmsds[sample] < self.rmsd_classification_cutoff[0]).unsqueeze(0).float()
            
#             print('rmsds', torch.tensor(rmsds).float().shape)
#             print('complex_graph.y', complex_graph.y.shape)
#             sss
            complex_graph.rmsd = torch.tensor(rmsds).float()

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
        print("Add perturbation: ", self.add_perturbation)
#         print(running_modes)

        if self.running_mode == "train":
            water_ratio = 15
            resample_steps = 1
        elif self.running_mode == "test":
            water_ratio = 15
            resample_steps = 1
        else:
            raise ValueError("Invalid running mode!")
        total_resample_ratio = water_ratio * resample_steps
        print('common t schedule', tr_schedule)
        print('water_number/residue_number ratio: ', water_ratio)
        print('resampling steps: ', resample_steps)
        print('total resampling ratio: ', total_resample_ratio)

        # print('HAPPENING | loading cached complexes of the original model to create the confidence dataset RMSDs and predicted positions. Doing that from: ', os.path.join(self.complex_graphs_cache, "heterographs.pkl"))
        # with open(os.path.join(original_model_cache, "heterographs.pkl"), 'rb') as f:
        #     complex_graphs = pickle.load(f)
        # dataset = ListDataset(complex_graphs)
        # loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)
        
        rmsds, full_ligand_positions, names = [], [], []
        for idx, orig_complex_graph in tqdm(enumerate(self.loader)):
            data_list = [copy.deepcopy(orig_complex_graph) for _ in range(self.samples_per_complex)]
            # ## set num_metal to be similar to residue number
            # res_num = int(orig_complex_graph[0]['receptor'].pos.shape[0])
            # num_metal = res_num if res_num < 50 else 50
            # ##
            res_num = int(orig_complex_graph[0]['receptor'].pos.shape[0])
            step_num_water = int(res_num * water_ratio)
            total_num_water = int(res_num * total_resample_ratio)
            total_sampled_water = 0

            
#             original code
# 
#             randomize_position_multiple(data_list, False, self.original_model_args.tr_sigma_max, water_num=num_water)
#             predictions_list, confidences = sampling_test1(data_list=data_list, model=model,
#                                                 inference_steps=self.inference_steps,
#                                                 tr_schedule=tr_schedule,
#                                                 device=self.device, t_to_sigma=t_to_sigma, model_args=self.original_model_args)
#             original code

#             orig_complex_graph['ligand'].orig_pos = (orig_complex_graph['ligand'].pos.cpu().numpy() + orig_complex_graph.original_center.cpu().numpy())
#             orig_ligand_pos = np.expand_dims(orig_complex_graph['ligand'].orig_pos - orig_complex_graph.original_center.cpu().numpy(), axis=0)


#             if isinstance(orig_complex_graph['ligand'].orig_pos, list):
#                 orig_complex_graph['ligand'].orig_pos = orig_complex_graph['ligand'].orig_pos[0]  
#             real_water_pos = find_real_water_pos(os.path.join(self.args.data_dir, f"{orig_complex_graph.name[0]}/{orig_complex_graph.name[0]}_water.pdb"))
#             real_water_pos_centered = real_water_pos - orig_complex_graph.original_center.cpu().numpy()
            
            
            prediction_list = []
            confidence_list = []
            
#             print('total_num_water: ', total_num_water)
            
#             while total_sampled_water < total_num_water:
#                 sample_data_list = copy.deepcopy(data_list)
#                 randomize_position_multiple(sample_data_list, False, self.original_model_args.tr_sigma_max, water_num=step_num_water)
#                 predictions, confidences = sampling_test1(data_list=sample_data_list, model=model,
#                                                     inference_steps=self.inference_steps,
#                                                     tr_schedule=tr_schedule,
#                                                     device=self.device, t_to_sigma=t_to_sigma, model_args=self.original_model_args) 
#                 if total_sampled_water < 0.4 * total_num_water:
#                     predicted_water_pos = predictions[0]['ligand'].pos.cpu().numpy()
#                     tp_coords = torch.tensor(find_tp_coords(real_water_pos_centered, predicted_water_pos), dtype=torch.float32)
#                     predictions[0]['ligand'].pos = tp_coords
#                     total_sampled_water += tp_coords.shape[0]
# #                     print("predictions[0]['ligand'].pos: ", predictions[0]['ligand'].pos.shape)
# #                     print("predictions[0]['ligand'].pos: ", predictions[0]['ligand'].pos[0])
# #                     print('total_sampled_water: ', total_sampled_water)
#                 else:
#                     total_sampled_water += predictions[0]['ligand'].pos.shape[0]
#                 prediction_list.append(predictions)
#                 confidence_list.append(confidences)
            

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
            orig_ligand_pos = np.expand_dims(orig_complex_graph['ligand'].orig_pos - orig_complex_graph.original_center.cpu().numpy(), axis=0)

            if isinstance(orig_complex_graph['ligand'].orig_pos, list):
                orig_complex_graph['ligand'].orig_pos = orig_complex_graph['ligand'].orig_pos[0]
                
            real_water_pos = find_real_water_pos(os.path.join(self.args.data_dir, f"{orig_complex_graph.name[0]}/{orig_complex_graph.name[0]}_water.pdb"))
                   
            ligand_pos_list = []
            for complex_graphs in prediction_list:
                for complex_graph in complex_graphs:
                    ligand_pos_list.append(complex_graph['ligand'].pos.cpu().numpy())
            
            all_ligand_pos = np.concatenate(ligand_pos_list, axis=0)
            ligand_pos = np.asarray([all_ligand_pos], dtype=np.float32)
            
            ## recacluateing rmsds to consider original pdb file
            
            positions_new = ligand_pos.squeeze(0) + orig_complex_graph.original_center.cpu().numpy()
            rmsd, indices = get_nearest_point_distances(positions_new, real_water_pos)
            ##
            # rmsd, min_indices = get_nearest_point_distances(ligand_pos, orig_ligand_pos)
            
            
#             print()
            
#             print('real_water_pos: ', real_water_pos.shape)
#             print('rmsd.shape: ', rmsd.shape)
#             print("sum(rmsd<1): ",sum(rmsd < 1))
#             print(rmssd)
            
            rmsds.append(rmsd)
#             full_ligand_positions.append(np.asarray([complex_graph['ligand'].pos.cpu().numpy() for complex_graph in predictions_list]))
            full_ligand_positions.append(ligand_pos)
    
            
#             full_ligand_positions_origin = []
#             full_ligand_positions_origin.append(np.asarray([complex_graph['ligand'].pos.cpu().numpy() for complex_graph in predictions_list]))
#             print('full_ligand_positions_origin: ', full_ligand_positions_origin)
#             print('full_ligand_positions: ', full_ligand_positions)
            
#             print(ligand_poss)
            names.append(orig_complex_graph.name[0])
            assert(len(orig_complex_graph.name) == 1) # I just put this assert here because of the above line where I assumed that the list is always only lenght 1. Just in case it isn't maybe check what the names in there are.
        with open(os.path.join(self.full_cache_path, f"ligand_positions{'' if self.cache_creation_id is None else '_id' + str(self.cache_creation_id)}.pkl"), 'wb') as f:
            pickle.dump((full_ligand_positions, rmsds), f)
        with open(os.path.join(self.full_cache_path, f"complex_names_in_same_order{'' if self.cache_creation_id is None else '_id' + str(self.cache_creation_id)}.pkl"), 'wb') as f:
            pickle.dump((names), f)


