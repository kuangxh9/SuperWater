import gc
import math
import os
import numpy as np
import random
import shutil
import torch.nn as nn
from argparse import Namespace, ArgumentParser, FileType
import torch.nn.functional as F
from functools import partial
import wandb
import torch
import time
from sklearn.metrics import roc_auc_score
from torch_geometric.loader import DataListLoader, DataLoader
from tqdm import tqdm
from datasets.pdbbind import PDBBind, NoiseTransform
from confidence.dataset import ConfidenceDataset
from utils.training import AverageMeter
from scipy.spatial.distance import cdist

torch.multiprocessing.set_sharing_strategy('file_system')

import yaml
from utils.utils import save_yaml_file, get_optimizer_and_scheduler, get_model
from utils.diffusion_utils import t_to_sigma as t_to_sigma_compl
from confidence.dataset import get_args
from sklearn.cluster import DBSCAN
from utils.cluster_centroid import find_centroids
from utils.find_water_pos import find_real_water_pos
from utils.nearest_point_dist import get_nearest_point_distances

from utils.parsing import parse_inference_args

args = parse_inference_args()

total_sampling_ratio = args.water_ratio * args.resample_steps

if args.config:
    config_dict = yaml.load(args.config, Loader=yaml.FullLoader)
    arg_dict = args.__dict__
    for key, value in config_dict.items():
        if isinstance(value, list):
            for v in value:
                arg_dict[key].append(v)
        else:
            arg_dict[key] = value
    args.config = args.config.name
assert (args.main_metric_goal == 'max' or args.main_metric_goal == 'min')

save_pos_path = "inference_out/" + f"inferenced_pos_rr{total_sampling_ratio}_cap{args.cap}" + "/"
os.makedirs(save_pos_path, exist_ok=True)

torch.manual_seed(42)

def convert_txt_to_pdb(txt_file_path: str, output_pdb_path: str):
    with open(txt_file_path, 'r') as file:
        lines = file.readlines()

    pdb_lines = []
    for i, line in enumerate(lines, start=1):
        coords = list(map(float, line.strip().split()))
        pdb_line = f'HETATM{i:>5}  O   HOH A{1:>4}    {coords[0]:8.3f}{coords[1]:8.3f}{coords[2]:8.3f}  1.00  0.00           O\n'
        pdb_lines.append(pdb_line)

    with open(output_pdb_path, 'w') as pdb_file:
        pdb_file.writelines(pdb_lines)

    print(f"Successfully saved PDB file to: {output_pdb_path}")

def test_epoch(model, loader, mad_prediction, filter=True, use_sigmoid=args.use_sigmoid, quiet=False):
    model.eval()
    log_data = []
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    total_ratio = args.water_ratio * args.resample_steps
    for data in tqdm(loader, total=len(loader)):

        start_time = time.time()
        pdb_name = data[0].name
        try:
            with torch.no_grad():
                pred = model(data)
            labels = torch.cat([graph.y for graph in data]).to(device)
            if use_sigmoid:
                probabilities = torch.sigmoid(pred)
                pred_labels = (probabilities > args.cap).float()
            else:
                probabilities = pred
                probabilities[probabilities > 1] = 1

                probabilities = 1 - probabilities
                pred_labels = (probabilities > args.cap).float()
            
            positions = torch.cat([graph['ligand'].pos for graph in data]).to(device)
            positions_adjusted = positions.cpu().numpy() + data[0].original_center.numpy()
            num_sampled_positions = len(positions_adjusted)
                
            try:                
                centroids =  find_centroids(positions_adjusted, 
                                            probabilities.cpu().numpy(), 
                                            threshold=args.cap, 
                                            cluster_distance=1.52, 
                                            use_weighted_avg=True,
                                            clash_distance=2.2)  
                print('centroids: ', len(centroids))
                
                if centroids is None:
                    raise Exception(f"Centroid is None. Cannot process PDB {data[0].name}")
                
                # save file
                try:
                    if args.save_pos:        
                        pdb_name = data[0].name

                        pdb_folder = os.path.join(save_pos_path, pdb_name)
                        os.makedirs(pdb_folder, exist_ok=True)

                        filtered_file_path = os.path.join(pdb_folder, f"{pdb_name}_filtered.txt")
                        filtered_probabilities_reshaped = probabilities.reshape(-1, 1).cpu().numpy()
                        combined_pos_prob = np.hstack((positions_adjusted, filtered_probabilities_reshaped))
                        np.savetxt(filtered_file_path, combined_pos_prob, fmt='%.3f')

                        save_txt_path = os.path.join(pdb_folder, f'{pdb_name}_centroid.txt')
                        np.savetxt(save_txt_path, centroids, fmt='%.8f')
                        print(f"Saved centroids for {pdb_name} to {save_txt_path}")
                        
                        save_pdb_path = os.path.join(pdb_folder, f'{pdb_name}_centroid.pdb')
                        convert_txt_to_pdb(save_txt_path, save_pdb_path)
                except Exception as e:
                    print('Cannot save pdb: ', data[0].name, e)
                
            except Exception as e:
                print(f"An error occurred on {data[0].name}", e)
                continue

        except RuntimeError as e:
            if 'out of memory' in str(e):
                print('| WARNING: ran out of memory, skipping batch')
                for p in model.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                torch.cuda.empty_cache()
                continue
            else:
                raise e

        end_time = time.time()
        processing_time = end_time - start_time
        log_data.append((pdb_name, f"{processing_time:.2f}"))
    with open(f"logs/inference_log_rr{total_ratio}.txt", "w") as log_file:
        for record in log_data:
            log_file.write(f"{record[0]} {record[1]}\n")

def evalulation(args, model, val_loader, run_dir):
    print("Starting testing...")
    test_epoch(model, val_loader, args.mad_prediction)

def construct_loader_origin(args_confidence, args, t_to_sigma):    
    confi_common_args = {'transform': None, 'root': args_confidence.data_dir, 'limit_complexes': args.limit_complexes,
                   'receptor_radius': args.receptor_radius,
                   'c_alpha_max_neighbors': args.c_alpha_max_neighbors,
                   'remove_hs': args.remove_hs, 'max_lig_size': args.max_lig_size,
                   'popsize': args.matching_popsize, 'maxiter': args.matching_maxiter,
                   'num_workers': args.num_workers, 'all_atoms': args.all_atoms,
                   'atom_radius': args.atom_radius, 'atom_max_neighbors': args.atom_max_neighbors,
                   'esm_embeddings_path': args_confidence.esm_embeddings_path}
    
    print('esm_embeddings_path:', args_confidence.esm_embeddings_path)
    
    test_dataset = PDBBind(cache_path=args.cache_path, split_path=args_confidence.split_test, keep_original=True,
                           **confi_common_args)
    
    loader_class = DataLoader
    
    test_loader = loader_class(dataset=test_dataset, batch_size=args_confidence.batch_size_preprocessing,
                               num_workers=args_confidence.num_workers, shuffle=False, pin_memory=args.pin_memory)
    
    return test_loader


def construct_loader_confidence(args, device):
    common_args = {'cache_path': args.cache_path, 'original_model_dir': args.original_model_dir, 'device': device,
                   'inference_steps': args.inference_steps, 'samples_per_complex': args.samples_per_complex,
                   'limit_complexes': args.limit_complexes, 'all_atoms': args.all_atoms, 'balance': args.balance,
                   'mad_classification_cutoff': args.mad_classification_cutoff,
                   'use_original_model_cache': args.use_original_model_cache,
                   'cache_creation_id': args.cache_creation_id, "cache_ids_to_combine": args.cache_ids_to_combine,
                   "model_ckpt": args.ckpt,
                   "running_mode": args.running_mode,
                   "water_ratio": args.water_ratio,
                   "resample_steps": args.resample_steps,
                   "save_visualization": args.save_visualization}
    loader_class = DataListLoader if torch.cuda.is_available() else DataLoader
    exception_flag = False

    # construct original loader
    original_model_args = get_args(args.original_model_dir)
    t_to_sigma = partial(t_to_sigma_compl, args=original_model_args)
    
    test_loader = construct_loader_origin(args, original_model_args, t_to_sigma)

    test_dataset = ConfidenceDataset(loader=test_loader, split=os.path.splitext(os.path.basename(args.split_test))[0],
                                     args=args, **common_args)
    test_loader = loader_class(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)  ## TODO True

    if exception_flag: raise Exception('We encountered the exception during train dataset loading: ', e)
        
    return test_loader


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


if __name__ == '__main__':
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with open(f'{args.original_model_dir}/model_parameters.yml') as f:
        score_model_args = Namespace(**yaml.full_load(f))
    with open(f'{args.confidence_dir}/model_parameters.yml') as f:
        confidence_args = Namespace(**yaml.full_load(f))

    
    test_loader = construct_loader_confidence(args, device)

    t_to_sigma = partial(t_to_sigma_compl, args=score_model_args)
    model = get_model(confidence_args, device, t_to_sigma=None, confidence_mode=True)

    # Load state_dict
    state_dict = torch.load(f'{args.confidence_dir}/best_model.pt', map_location='cpu')
    # Adjust for DataParallel wrapping
    new_state_dict = {'module.' + k: v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict, strict=True)

    numel = sum([p.numel() for p in model.parameters()])
    print('Loading trained confidence model with', numel, 'parameters')

    if args.wandb:
        wandb.init(
            entity='Xiaohan Kuang',
            settings=wandb.Settings(start_method="fork"),
            project=args.project,
            name=args.run_name,
            config=args
        )
        wandb.log({'numel': numel})

    # record parameters
    run_dir = os.path.join(args.log_dir, args.run_name)
    yaml_file_name = os.path.join(run_dir, 'model_parameters.yml')
    save_yaml_file(yaml_file_name, args.__dict__)
    args.device = device

    evalulation(args, model, test_loader, run_dir)
