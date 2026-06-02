import os
import random
import time
from argparse import Namespace
from functools import partial

import numpy as np
import torch
import wandb
import yaml
from torch_geometric.loader import DataListLoader, DataLoader
from tqdm import tqdm

from superwater.datasets.pdbbind import PDBBind
from superwater.confidence.dataset import ConfidenceDataset, get_args
from superwater.utils.utils import save_yaml_file, get_model, resolve_model_dir
from superwater.utils.diffusion_utils import t_to_sigma as t_to_sigma_compl
from superwater.utils.cluster_centroid import find_centroids
from superwater.utils.parsing import parse_inference_args
# convert_txt_to_pdb is re-exported here for backward compatibility.
from superwater.structure_io import convert_txt_to_pdb, write_water_structure  # noqa: F401

torch.multiprocessing.set_sharing_strategy('file_system')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _save_prediction_outputs(out_dir, pdb_name, positions_adjusted, probabilities, centroids,
                             save_structure=True, save_filtered=True, output_format="pdb"):
    os.makedirs(out_dir, exist_ok=True)
    if save_filtered:
        probs_col = probabilities.reshape(-1, 1).cpu().numpy()
        np.savetxt(os.path.join(out_dir, f"{pdb_name}_filtered.txt"),
                   np.hstack((positions_adjusted, probs_col)), fmt='%.3f')
    save_txt_path = os.path.join(out_dir, f'{pdb_name}_centroid.txt')
    np.savetxt(save_txt_path, centroids, fmt='%.8f')
    print(f"Saved {len(centroids)} water centroids for {pdb_name} to {save_txt_path}")
    if save_structure:
        ext = 'cif' if str(output_format).lower() == 'cif' else 'pdb'
        write_water_structure(save_txt_path, os.path.join(out_dir, f'{pdb_name}_centroid.{ext}'), ext)


def test_epoch(model, loader, args, save_pos_path, per_complex_subdir=True,
               save_structure=True, save_filtered=True, output_format="pdb"):
    model.eval()
    log_data = []
    log_dir = "outputs/logs"
    os.makedirs(log_dir, exist_ok=True)
    total_ratio = args.water_ratio * args.resample_steps
    for data in tqdm(loader, total=len(loader)):
        start_time = time.time()
        pdb_name = data[0].name
        try:
            with torch.no_grad():
                pred = model(data)
            if args.use_sigmoid:
                probabilities = torch.sigmoid(pred)
            else:
                # The model outputs a normalized mean-absolute-deviation; convert to a
                # "keep" probability where higher = more confident (1 - clamped MAD).
                probabilities = pred
                probabilities[probabilities > 1] = 1
                probabilities = 1 - probabilities

            positions = torch.cat([graph['ligand'].pos for graph in data]).to(device)
            # Sampled positions are centered on the protein; shift back to the original frame.
            positions_adjusted = positions.cpu().numpy() + data[0].original_center.numpy()

            try:
                centroids = find_centroids(positions_adjusted, probabilities.cpu().numpy(),
                                           threshold=args.cap, cluster_distance=1.52,
                                           use_weighted_avg=True, clash_distance=2.2)
                if centroids is None:
                    raise Exception(f"Centroid is None. Cannot process PDB {pdb_name}")
                print('centroids: ', len(centroids))

                if args.save_pos:
                    out_dir = os.path.join(save_pos_path, pdb_name) if per_complex_subdir else save_pos_path
                    try:
                        _save_prediction_outputs(out_dir, pdb_name, positions_adjusted, probabilities,
                                                 centroids, save_structure=save_structure,
                                                 save_filtered=save_filtered, output_format=output_format)
                    except Exception as e:
                        print('Cannot save outputs for', pdb_name, e)
            except Exception as e:
                print(f"An error occurred on {pdb_name}", e)
                continue

        except RuntimeError as e:
            if 'out of memory' in str(e):
                print('| WARNING: ran out of memory, skipping batch')
                for p in model.parameters():
                    if p.grad is not None:
                        del p.grad
                torch.cuda.empty_cache()
                continue
            else:
                raise e

        log_data.append((pdb_name, f"{time.time() - start_time:.2f}"))
    with open(f"{log_dir}/inference_log_rr{total_ratio}.txt", "w") as log_file:
        for record in log_data:
            log_file.write(f"{record[0]} {record[1]}\n")


def construct_loader_origin(args_inference, args_score, t_to_sigma):
    # Dataset preprocessing parameters (receptor radius, neighbours, cache, ...) come from the
    # *score model's* saved args so the graphs match what the model was trained on; the data to
    # run on (data_dir, split, ESM embeddings) comes from the inference args.
    common_args = {'transform': None, 'root': args_inference.data_dir, 'limit_complexes': args_score.limit_complexes,
                   'receptor_radius': args_score.receptor_radius,
                   'c_alpha_max_neighbors': args_score.c_alpha_max_neighbors,
                   'remove_hs': args_score.remove_hs, 'max_lig_size': args_score.max_lig_size,
                   'popsize': args_score.matching_popsize, 'maxiter': args_score.matching_maxiter,
                   'num_workers': args_score.num_workers, 'all_atoms': args_score.all_atoms,
                   'atom_radius': args_score.atom_radius, 'atom_max_neighbors': args_score.atom_max_neighbors,
                   'esm_embeddings_path': args_inference.esm_embeddings_path}

    print('esm_embeddings_path:', args_inference.esm_embeddings_path)

    test_dataset = PDBBind(cache_path=args_score.cache_path, split_path=args_inference.split_test, keep_original=True,
                           **common_args)

    return DataLoader(dataset=test_dataset, batch_size=args_inference.batch_size_preprocessing,
                      num_workers=args_inference.num_workers, shuffle=False, pin_memory=args_score.pin_memory)


def construct_loader_confidence(args, device, score_model=None):
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
                   "save_visualization": args.save_visualization,
                   "score_model": score_model}
    loader_class = DataListLoader if torch.cuda.is_available() else DataLoader

    original_model_args = get_args(args.original_model_dir)
    t_to_sigma = partial(t_to_sigma_compl, args=original_model_args)
    test_loader = construct_loader_origin(args, original_model_args, t_to_sigma)

    test_dataset = ConfidenceDataset(loader=test_loader, split=os.path.splitext(os.path.basename(args.split_test))[0],
                                     args=args, **common_args)
    return loader_class(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)


def load_confidence_model(confidence_dir, device):
    with open(f'{confidence_dir}/model_parameters.yml') as f:
        confidence_args = Namespace(**yaml.full_load(f))
    model = get_model(confidence_args, device, t_to_sigma=None, confidence_mode=True)
    # The checkpoint was saved from a DataParallel-wrapped module, so its keys lack the
    # 'module.' prefix that get_model() (which re-wraps on CUDA) expects.
    # weights_only=True: these checkpoints are plain tensor state dicts.
    state_dict = torch.load(f'{confidence_dir}/best_model.pt', map_location='cpu', weights_only=True)
    model.load_state_dict({'module.' + k: v for k, v in state_dict.items()}, strict=True)
    print('Loaded confidence model with', sum(p.numel() for p in model.parameters()), 'parameters')
    return model


def run_inference(args, save_pos_path, device, per_complex_subdir=True,
                  save_structure=True, save_filtered=True, output_format="pdb",
                  score_model=None, confidence_model=None):
    """Run the full pipeline (score sampling -> confidence scoring -> clustering -> save)
    for every complex listed in ``args.split_test``. Reused by both the CLI and predict.

    ``score_model``/``confidence_model`` may be preloaded and reused across a batch.
    Returns the PyG graph-cache directory used (for optional cleanup)."""
    os.makedirs(save_pos_path, exist_ok=True)
    test_loader = construct_loader_confidence(args, device, score_model=score_model)
    model = confidence_model if confidence_model is not None else load_confidence_model(args.confidence_dir, device)
    print("Starting prediction...")
    test_epoch(model, test_loader, args, save_pos_path, per_complex_subdir=per_complex_subdir,
               save_structure=save_structure, save_filtered=save_filtered, output_format=output_format)
    return test_loader.dataset.loader.dataset.full_cache_path  # PyG graph cache dir


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


def main():
    args = parse_inference_args()
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

    args.original_model_dir = resolve_model_dir(args.original_model_dir)
    args.confidence_dir = resolve_model_dir(args.confidence_dir)

    total_sampling_ratio = args.water_ratio * args.resample_steps
    sub_dir = args.save_pos_path or f"inferenced_pos_rr{total_sampling_ratio}_cap{args.cap}"
    save_pos_path = os.path.join("outputs", sub_dir) + os.sep

    set_seed(42)

    if args.wandb:
        wandb.init(entity=args.wandb_entity, settings=wandb.Settings(start_method="fork"),
                   project=args.project, name=args.run_name, config=args)

    run_dir = os.path.join(args.log_dir, args.run_name)
    save_yaml_file(os.path.join(run_dir, 'model_parameters.yml'), args.__dict__)
    args.device = device

    run_inference(args, save_pos_path, device, output_format=getattr(args, 'output_format', 'pdb'))


if __name__ == '__main__':
    main()
