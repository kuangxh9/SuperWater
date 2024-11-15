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
from utils.cluster_centroid import find_centroid
from utils.find_water_pos import find_real_water_pos
from utils.nearest_point_dist import get_nearest_point_distances

parser = ArgumentParser()
parser.add_argument('--config', type=FileType(mode='r'), default=None)
parser.add_argument('--original_model_dir', type=str, default='workdir',
                    help='Path to folder with trained model and hyperparameters')
parser.add_argument('--confidence_dir', type=str, default='workdir',
                    help='Path to folder with trained confidence model and hyperparameters')
parser.add_argument('--restart_dir', type=str, default=None, help='')
parser.add_argument('--use_original_model_cache', action='store_true', default=False,
                    help='If this is true, the same dataset as in the original model will be used. Otherwise, the dataset parameters are used.')
parser.add_argument('--data_dir', type=str, default='data/cleaned_metal_protein/',
                    help='Folder containing original structures')
parser.add_argument('--ckpt', type=str, default='best_model.pt', help='Checkpoint to use inside the folder')
parser.add_argument('--model_save_frequency', type=int, default=0,
                    help='Frequency with which to save the last model. If 0, then only the early stopping criterion best model is saved and overwritten.')
parser.add_argument('--best_model_save_frequency', type=int, default=0,
                    help='Frequency with which to save the best model. If 0, then only the early stopping criterion best model is saved and overwritten.')
parser.add_argument('--run_name', type=str, default='evaluation', help='')
parser.add_argument('--project', type=str, default='diffwater_evalution', help='')
parser.add_argument('--split_train', type=str, default='data/splits/timesplit_no_lig_overlap_train',
                    help='Path of file defining the split')
parser.add_argument('--split_val', type=str, default='data/splits/timesplit_no_lig_overlap_val',
                    help='Path of file defining the split')
parser.add_argument('--split_test', type=str, default='data/splits/timesplit_test',
                    help='Path of file defining the split')

# Inference parameters for creating the positions and rmsds that the confidence predictor will be trained on.
parser.add_argument('--cache_path', type=str, default='data/cacheNew',
                    help='Folder from where to load/restore cached dataset')
parser.add_argument('--cache_ids_to_combine', nargs='+', type=str, default=None,
                    help='RMSD value below which a prediction is considered a postitive. This can also be multiple cutoffs.')
parser.add_argument('--cache_creation_id', type=int, default=None,
                    help='number of times that inference is run on the full dataset before concatenating it and coming up with the full confidence dataset')
parser.add_argument('--wandb', action='store_true', default=False, help='')
parser.add_argument('--inference_steps', type=int, default=20, help='Number of denoising steps')
parser.add_argument('--samples_per_complex', type=int, default=3, help='')
parser.add_argument('--balance', action='store_true', default=False,
                    help='If this is true than we do not force the samples seen during training to be the same amount of negatives as positives')
parser.add_argument('--rmsd_prediction', action='store_true', default=False, help='')
parser.add_argument('--rmsd_classification_cutoff', type=float, default=2,
                    help='RMSD value below which a prediction is considered a postitive. This can also be multiple cutoffs.')

parser.add_argument('--log_dir', type=str, default='workdir', help='')
parser.add_argument('--main_metric', type=str, default='accuracy',
                    help='Metric to track for early stopping. Mostly [loss, accuracy, ROC AUC]')
parser.add_argument('--main_metric_goal', type=str, default='max', help='Can be [min, max]')
parser.add_argument('--transfer_weights', action='store_true', default=False, help='')
parser.add_argument('--batch_size', type=int, default=5, help='')
parser.add_argument('--batch_size_preprocessing', type=int, default=4, help='Number of workers')
parser.add_argument('--lr', type=float, default=1e-3, help='')
parser.add_argument('--w_decay', type=float, default=0.0, help='')
parser.add_argument('--scheduler', type=str, default='plateau', help='')
parser.add_argument('--scheduler_patience', type=int, default=20, help='')
parser.add_argument('--n_epochs', type=int, default=1, help='')

# Dataset
parser.add_argument('--limit_complexes', type=int, default=0, help='')
parser.add_argument('--all_atoms', action='store_true', default=False, help='')
parser.add_argument('--multiplicity', type=int, default=1, help='')
parser.add_argument('--chain_cutoff', type=float, default=10, help='')
parser.add_argument('--receptor_radius', type=float, default=30, help='')
parser.add_argument('--c_alpha_max_neighbors', type=int, default=10, help='')
parser.add_argument('--atom_radius', type=float, default=5, help='')
parser.add_argument('--atom_max_neighbors', type=int, default=8, help='')
parser.add_argument('--matching_popsize', type=int, default=20, help='')
parser.add_argument('--matching_maxiter', type=int, default=20, help='')
parser.add_argument('--max_lig_size', type=int, default=None, help='Maximum number of heavy atoms')
parser.add_argument('--remove_hs', action='store_true', default=False, help='remove Hs')
parser.add_argument('--num_conformers', type=int, default=1, help='')
parser.add_argument('--esm_embeddings_path', type=str, default=None,
                    help='If this is set then the LM embeddings at that path will be used for the receptor features')
parser.add_argument('--no_torsion', action='store_true', default=False, help='')

# Model
parser.add_argument('--num_conv_layers', type=int, default=2, help='Number of interaction layers')
parser.add_argument('--max_radius', type=float, default=5.0, help='Radius cutoff for geometric graph')
parser.add_argument('--scale_by_sigma', action='store_true', default=True, help='Whether to normalise the score')
parser.add_argument('--ns', type=int, default=16, help='Number of hidden features per node of order 0')
parser.add_argument('--nv', type=int, default=4, help='Number of hidden features per node of order >0')
parser.add_argument('--distance_embed_dim', type=int, default=32, help='')
parser.add_argument('--cross_distance_embed_dim', type=int, default=32, help='')
parser.add_argument('--no_batch_norm', action='store_true', default=False, help='If set, it removes the batch norm')
parser.add_argument('--use_second_order_repr', action='store_true', default=False,
                    help='Whether to use only up to first order representations or also second')
parser.add_argument('--cross_max_distance', type=float, default=80, help='')
parser.add_argument('--dynamic_max_cross', action='store_true', default=False, help='')
parser.add_argument('--dropout', type=float, default=0.0, help='MLP dropout')
parser.add_argument('--embedding_type', type=str, default="sinusoidal", help='')
parser.add_argument('--sigma_embed_dim', type=int, default=32, help='')
parser.add_argument('--embedding_scale', type=int, default=10000, help='')
parser.add_argument('--confidence_no_batchnorm', action='store_true', default=False, help='')
parser.add_argument('--confidence_dropout', type=float, default=0.0, help='MLP dropout in confidence readout')

parser.add_argument('--num_workers', type=int, default=1, help='Number of workers')
parser.add_argument('--prob_thresh', type=float, default=0.5, help='confidence model prob threshold')
parser.add_argument('--save_pos', action='store_true', default=False, help='')
parser.add_argument('--cluster_eps', type=float, default=1, help='')
parser.add_argument('--cluster_min_samples', type=int, default=1, help='')
parser.add_argument('--running_mode', type=str, default="test")
parser.add_argument('--use_sigmoid', action='store_true', default=False, help='')

args = parser.parse_args()
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

# print('---------------', args.data_dir)

save_pos_path = "inference_out/" + f"inferenced_pos_prob{args.prob_thresh}_scale4" + "/"    ### TODO: to be changed
os.makedirs(save_pos_path, exist_ok=True)

### testing
torch.manual_seed(42)

def test_epoch(model, loader, rmsd_prediction, filter=True, use_sigmoid=args.use_sigmoid, quiet=False):
    model.eval()
    meter = AverageMeter(['confidence_loss', 'accuracy', 'ROC AUC'], unpooled_metrics=True)
    all_labels = []
    #     n_filtered_positions, n_real_water_pos, n_qualified_real_idx = 0, 0, 0

    all_centroid_coverage_rate = []
    all_centroid_accuracy = []
    all_centroid_true_ratio = []
    
    all_filtered_coverage_rate = []
    all_filtered_accuracy = []
    all_filtered_true_ratio = []

    for data in tqdm(loader, total=len(loader)):
        try:
            with torch.no_grad():
                pred = model(data)
            labels = torch.cat([graph.y for graph in data]).to(device)
            # labels = labels.flatten()
            ### note: change pred > 0 to pred > -5 for much better recall
            if use_sigmoid:
                probabilities = torch.sigmoid(pred)
                pred_labels = (probabilities > args.prob_thresh).float()
            else:
#                 pred_labels = (pred > 0).float()
                probabilities = pred
                probabilities[probabilities>1] = 1
        
#                 print(probabilities[probabilities>1])
#                 sss
                probabilities = 1-probabilities
                pred_labels = (probabilities>args.prob_thresh).float()
            
            positions = torch.cat([graph['ligand'].pos for graph in data]).to(device)
            num_sampled_positions = len(positions)
            if filter:
                filtered_positions = positions[pred_labels == 1]
                filtered_positions = filtered_positions.cpu().numpy()
                filtered_probabilities = probabilities[pred_labels == 1].cpu().numpy()
            else:
                filtered_positions = positions.cpu().numpy()
            try:                
                num_filtered_positions = len(filtered_positions)

#                 dbscan = DBSCAN(eps=args.cluster_eps, min_samples=args.cluster_min_samples)
#                 clusters = dbscan.fit_predict(filtered_positions)
#                 centroids = find_centroid(filtered_positions, clusters) + data[0].original_center.numpy()
                
                # testing
                filtered_positions_adjusted = filtered_positions + data[0].original_center.numpy()
        
                real_water_pos = find_real_water_pos(
                    os.path.join(args.data_dir, f"{data[0].name}/{data[0].name}_water.pdb"))
#                 nearest_dist_positions, indices = get_nearest_point_distances(centroids, real_water_pos)
                
                # testing
                nearest_dist_positions_filtered, indices_filtered = get_nearest_point_distances(filtered_positions_adjusted, real_water_pos)
                
#                 qualified_centroids, qualified_centroid_real_idx = [], []
#                 for i, (centroid_dist, idx) in enumerate(zip(nearest_dist_positions, indices)):
#                     if centroid_dist <= args.rmsd_classification_cutoff:
#                         qualified_centroids.append(centroids[i])
#                         qualified_centroid_real_idx.append(float(idx))
                
                # testing
                qualified_filtered_pos, qualified_filtered_real_idx = [], []
                for i, (filtered_pos_dist, idx) in enumerate(zip(nearest_dist_positions_filtered, indices_filtered)):
                    if filtered_pos_dist <= args.rmsd_classification_cutoff:
                        qualified_filtered_pos.append(filtered_positions_adjusted[i])
                        qualified_filtered_real_idx.append(float(idx))
                
#                 print('filtered_positions_adjusted: ', filtered_positions_adjusted.shape)
#                 print('filtered_probabilities: ', filtered_probabilities.shape)
#                 print('filtered_probabilities: ', filtered_probabilities[:5])
                
                # save file
                try:
                    if args.save_pos:        
                        pdb_name = data[0].name

                        pdb_folder = os.path.join(save_pos_path, pdb_name)
                        os.makedirs(pdb_folder, exist_ok=True)

                        sampled_file_path = os.path.join(pdb_folder, f"{pdb_name}_sampled.txt")
#                         centroid_file_path = os.path.join(pdb_folder, f"{pdb_name}_centroid.txt")
                        filtered_file_path = os.path.join(pdb_folder, f"{pdb_name}_filtered.txt")

                        filtered_probabilities_reshaped = filtered_probabilities.reshape(-1, 1)
                        combined_pos_prob = np.hstack((filtered_positions_adjusted, filtered_probabilities_reshaped))

                        np.savetxt(sampled_file_path, positions.cpu().numpy() + data[0].original_center.numpy(), fmt='%.3f')
#                         np.savetxt(centroid_file_path, centroids, fmt='%.3f')
                        np.savetxt(filtered_file_path, combined_pos_prob, fmt='%.3f')
                        print(f"Successfully saved pdb {pdb_name} position.")
                except:
                    print('Cannot save pdb: ', data[0].name)
                    ss
                
                
                filter_ratio = num_filtered_positions/num_sampled_positions * 1
                
#                 centroid_accuracy = len(qualified_centroid_real_idx) / len(centroids) * 100
#                 centroid_coverage = len(set(qualified_centroid_real_idx)) / len(real_water_pos) * 100
#                 centroid_true_ratio = len(centroids) / len(real_water_pos) * 1
                
#                 testing
#                 print('qualified_filtered_real_idx', len(qualified_filtered_real_idx))
    
                filtered_pos_accuracy = len(qualified_filtered_real_idx) / len(filtered_positions_adjusted) * 100
                filtered_pos_coverage = len(set(qualified_filtered_real_idx)) / len(real_water_pos) * 100
                filtered_pos_true_ratio = len(filtered_positions_adjusted) / len(real_water_pos) * 1
                
                #                 print('----------------------------- test -----------------------------')
                #                 print('qualified_real_idx:', len(set(qualified_real_idx)))
                #                 print('filtered_positions: ', len(filtered_positions))
                #                 print('real_water_pos: ', len(real_water_pos))
                #                 print('centroids: ', len(centroids))
                #                 print('qualified_centroids: ', len(qualified_centroids))
                #                 print('----------------------------- test -----------------------------')
                
#                 all_centroid_accuracy.append(centroid_accuracy)
#                 all_centroid_coverage_rate.append(centroid_coverage)
#                 all_centroid_true_ratio.append(centroid_true_ratio)
                
                all_filtered_coverage_rate.append(filtered_pos_coverage)
                all_filtered_accuracy.append(filtered_pos_accuracy)
                all_filtered_true_ratio.append(filtered_pos_true_ratio)

                if not quiet:
                    print("name: ", data[0].name, "original#: ", num_sampled_positions, ", filtered#: ", num_filtered_positions)
#                     print("centroid accuracy: ", round(centroid_accuracy, 2), ", centroid coverage: ", round(centroid_coverage, 2),  ", centroid/true: ", centroid_true_ratio)
                    print("filtered_accuracy: ", round(filtered_pos_accuracy, 2), ", filtered_coverage_rate: ", round(filtered_pos_coverage, 2), ", filtered/true: ", filtered_pos_true_ratio)
                    print()
#                     print("original#: ", num_sampled_positions, ", filtered#: ", num_filtered_positions, ", centroid#: ", len(centroids), ", pred/true: ", round(pred_true_ratio, 2))

            #                 return
#                 sss
            except Exception as e:
                filtered_positions = []
                qualified_real_idx = []
                real_water_pos = find_real_water_pos(
                    os.path.join(args.data_dir, f"{data[0].name}/{data[0].name}_water.mol2"))
                print(f"An error occurred on {data[0].name}", e)
                continue
            #             n_filtered_positions += len(filtered_positions)
            #             n_real_water_pos += len(real_water_pos)
            #             n_qualified_real_idx += len(set(qualified_real_idx))

            confidence_loss = F.binary_cross_entropy_with_logits(pred, labels)
            predicted_labels = (pred > 0).float()

            confidence_classification_accuracy = torch.mean((labels == (pred > 0).float()).float())
            # roc_auc = roc_auc_score(labels.detach().cpu().numpy(), pred.detach().cpu().numpy())
            try:
                roc_auc = roc_auc_score(labels.detach().cpu().numpy(), pred.detach().cpu().numpy())
            except ValueError as e:
                if 'Only one class present in y_true. ROC AUC score is not defined in that case.' in str(e):
                    roc_auc = 0
                else:
                    raise e
            meter.add([confidence_loss.cpu().detach(), confidence_classification_accuracy.cpu().detach(),
                       torch.tensor(roc_auc)])
            all_labels.append(labels)

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
    
#     total_centroid_accuracy = np.mean(all_centroid_accuracy)
#     total_centroid_coverage = np.mean(all_centroid_coverage_rate)
#     total_centroid_ratio = np.mean(all_centroid_true_ratio)
    
    total_filtered_accuracy = np.mean(all_filtered_accuracy)
    total_filtered_coverage = np.mean(all_filtered_coverage_rate)
    total_filtered_ratio = np.mean(all_filtered_true_ratio)
    
    print("Probability Threshold: ", args.prob_thresh, ", Cluster eps: ", args.cluster_eps, ", Cluster MinPoints: ", args.cluster_min_samples)
#     print("Average centroid accuracy: ", round(total_centroid_accuracy, 2), ", Average centroid coverage: ", round(total_centroid_coverage,2), ", Average centroid ratio: ", round(total_centroid_ratio,2))
    print("Average filtered accuracy: ", round(total_filtered_accuracy,2), ", Average filtered coverage: ", round(total_filtered_coverage,2), ", Average filtered ratio: ", round(total_filtered_ratio,2))
    all_labels = torch.cat(all_labels)

    if rmsd_prediction:
        baseline_metric = ((all_labels - all_labels.mean()).abs()).mean()
    else:
        baseline_metric = all_labels.sum() / len(all_labels)
    results = meter.summary()
    results.update({'baseline_metric': baseline_metric})
    return meter.summary(), baseline_metric


def evalulation(args, model, val_loader, run_dir):
    best_val_metric = math.inf if args.main_metric_goal == 'min' else 0
    best_epoch = 0

    print("Starting training...")
    for epoch in range(args.n_epochs):
        logs = {}

        val_metrics, baseline_metric = test_epoch(model, val_loader, args.rmsd_prediction)
        if args.rmsd_prediction:
            print("Epoch {}: Validation loss {:.4f}".format(epoch, val_metrics['confidence_loss']))
        else:
            print("Epoch {}: Validation loss {:.4f}  accuracy {:.4f}".format(epoch, val_metrics['confidence_loss'],
                                                                             val_metrics['accuracy']))

        if args.wandb:
            logs.update({'valinf_' + k: v for k, v in val_metrics.items()}, step=epoch + 1)
            # logs.update({'train_' + k: v for k, v in train_metrics.items()}, step=epoch + 1)
            logs.update({'mean_rmsd' if args.rmsd_prediction else 'fraction_positives': baseline_metric})
            wandb.log(logs, step=epoch + 1)

    print("Best Validation accuracy {} on Epoch {}".format(best_val_metric, best_epoch))


def construct_loader_origin(args_confidence, args, t_to_sigma):
    ## the only difference compared to construct_loader is that we set batch_size = 1
    ## and we used DataLoader not DataLoaderList
#     print('enter construct_loader_origin')
    
#     print('---------------', args_confidence.data_dir)
    
    common_args = {'transform': None, 'root': args.data_dir, 'limit_complexes': args.limit_complexes,
                   'receptor_radius': args.receptor_radius,
                   'c_alpha_max_neighbors': args.c_alpha_max_neighbors,
                   'remove_hs': args.remove_hs, 'max_lig_size': args.max_lig_size,
                   'popsize': args.matching_popsize, 'maxiter': args.matching_maxiter,
                   'num_workers': args.num_workers, 'all_atoms': args.all_atoms,
                   'atom_radius': args.atom_radius, 'atom_max_neighbors': args.atom_max_neighbors,
                   'esm_embeddings_path': args.esm_embeddings_path}
    train_dataset = PDBBind(cache_path=args.cache_path, split_path=args_confidence.split_train, keep_original=True,
                            num_conformers=args.num_conformers, **common_args)
    # val_dataset = PDBBind(cache_path=args.cache_path, split_path=args_confidence.split_val, keep_original=True, **common_args)
#     print()
#     print(args)
#     print()
#     print(args_confidence)
#     print()
    
    confi_common_args = {'transform': None, 'root': args_confidence.data_dir, 'limit_complexes': args.limit_complexes,
                   'receptor_radius': args.receptor_radius,
                   'c_alpha_max_neighbors': args.c_alpha_max_neighbors,
                   'remove_hs': args.remove_hs, 'max_lig_size': args.max_lig_size,
                   'popsize': args.matching_popsize, 'maxiter': args.matching_maxiter,
                   'num_workers': args.num_workers, 'all_atoms': args.all_atoms,
                   'atom_radius': args.atom_radius, 'atom_max_neighbors': args.atom_max_neighbors,
                   'esm_embeddings_path': args_confidence.esm_embeddings_path}
    
    test_dataset = PDBBind(cache_path=args.cache_path, split_path=args_confidence.split_test, keep_original=True,
                           **confi_common_args)
    
#     test_dataset = PDBBind(cache_path=args.cache_path, split_path=args_confidence.split_test, keep_original=True,
#                            **common_args)
    
#     print('-------- after test ------')

    loader_class = DataLoader
    train_loader = loader_class(dataset=train_dataset, batch_size=args_confidence.batch_size_preprocessing,
                                num_workers=args_confidence.num_workers, shuffle=False, pin_memory=args.pin_memory)
    test_loader = loader_class(dataset=test_dataset, batch_size=args_confidence.batch_size_preprocessing,
                               num_workers=args_confidence.num_workers, shuffle=False, pin_memory=args.pin_memory)
    # infer_loader = loader_class(dataset=test_dataset, batch_size=args_confidence.batch_size_preprocessing, num_workers=args_confidence.num_workers, shuffle=False, pin_memory=args.pin_memory)

    return train_loader, test_loader


def construct_loader_confidence(args, device):

    common_args = {'cache_path': args.cache_path, 'original_model_dir': args.original_model_dir, 'device': device,
                   'inference_steps': args.inference_steps, 'samples_per_complex': args.samples_per_complex,
                   'limit_complexes': args.limit_complexes, 'all_atoms': args.all_atoms, 'balance': args.balance,
                   'rmsd_classification_cutoff': args.rmsd_classification_cutoff,
                   'use_original_model_cache': args.use_original_model_cache,
                   'cache_creation_id': args.cache_creation_id, "cache_ids_to_combine": args.cache_ids_to_combine,
                   "model_ckpt": args.ckpt,
                   "running_mode": args.running_mode}
    loader_class = DataListLoader if torch.cuda.is_available() else DataLoader
    exception_flag = False
    # construct original loader
    original_model_args = get_args(args.original_model_dir)
    t_to_sigma = partial(t_to_sigma_compl, args=original_model_args)
    train_loader, test_loader = construct_loader_origin(args, original_model_args, t_to_sigma)

#     try:
#         train_dataset = ConfidenceDataset(loader=train_loader,
#                                           split=os.path.splitext(os.path.basename(args.split_train))[0], args=args,
#                                           **common_args)
#         train_loader = loader_class(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
#     except Exception as e:
#         if 'The generated ligand positions with cache_id do not exist:' in str(e):
#             print("HAPPENING | Encountered the following exception when loading the confidence train dataset:")
#             print(str(e))
#             print(
#                 "HAPPENING | We are still continuing because we want to try to generate the validation dataset if it has not been created yet:")
#             exception_flag = True
#         else:
#             raise e
#     print('------ Here ------')
    test_dataset = ConfidenceDataset(loader=test_loader, split=os.path.splitext(os.path.basename(args.split_test))[0],
                                     args=args, **common_args)
    test_loader = loader_class(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)  ## TODO True

    if exception_flag: raise Exception('We encountered the exception during train dataset loading: ', e)
    return train_loader, test_loader


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

    # construct loader
    train_loader, test_loader = construct_loader_confidence(args, device)
    t_to_sigma = partial(t_to_sigma_compl, args=score_model_args)
    # model = get_model(score_model_args if args.transfer_weights else args, device, t_to_sigma=None, confidence_mode=True)
    model = get_model(confidence_args, device, t_to_sigma=None, confidence_mode=True)

    # Load state_dict
    state_dict = torch.load(f'{args.confidence_dir}/best_model.pt', map_location='cpu')
    # Adjust for DataParallel wrapping
    new_state_dict = {'module.' + k: v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict, strict=True)
    # model.load_state_dict(state_dict)

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