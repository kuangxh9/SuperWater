import gc
import math
import os

import shutil

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

torch.multiprocessing.set_sharing_strategy('file_system')

import yaml
from utils.utils import save_yaml_file, get_optimizer_and_scheduler, get_model
from utils.diffusion_utils import t_to_sigma as t_to_sigma_compl
from confidence.dataset import get_args
from utils.parsing import parse_confidence_args

args = parse_confidence_args()

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
assert(args.main_metric_goal == 'max' or args.main_metric_goal == 'min')


def sigmoid_function(target, scale=4): 
    '''
    Sigmoid function that normalizes input distance.

    Parameters
    ----------
    target : torch.Tensor
        torch.Tensor with your targets, of length (N).

    scale : float
        Number of scaling for the sigmoid function, default 2.

    Returns
    -------
    torch.Tensor of shape (N) set to [0,1] interval by the sigmoid function
    '''
    return (2/(1+torch.exp(-(scale*target)/torch.log(torch.tensor(2))))-1)**2


def train_epoch(model, loader, optimizer, mad_prediction):
    model.train()
    meter = AverageMeter(['confidence_loss'])

    for data in tqdm(loader, total=len(loader)):        
        if device.type == 'cuda' and len(data) % torch.cuda.device_count() == 1 or device.type == 'cpu' and data.num_graphs == 1:
            print("Skipping batch of size 1 since otherwise batchnorm would not work.")
        optimizer.zero_grad()
        try:
            pred = model(data)
            if mad_prediction:
                labels = torch.cat([graph.mad for graph in data]).to(device) if isinstance(data, list) else data.mad
                norm_labels = sigmoid_function(labels)
                confidence_loss = F.mse_loss(pred, norm_labels)
            else:
                if isinstance(args.mad_classification_cutoff, list):
                    labels = torch.cat([graph.y_binned for graph in data]).to(device) if isinstance(data, list) else data.y_binned
                    confidence_loss = F.cross_entropy(pred, labels)
                else:    
                    labels = torch.cat([graph.y for graph in data]).to(device) if isinstance(data, list) else data.y
                    confidence_loss = F.binary_cross_entropy_with_logits(pred, labels)
            confidence_loss.backward()
            optimizer.step()
            meter.add([confidence_loss.cpu().detach()])
        except RuntimeError as e:
            if 'out of memory' in str(e):
                print('| WARNING: ran out of memory, skipping batch')
                for p in model.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                torch.cuda.empty_cache()
                gc.collect()
                continue
            else:
                raise e

    return meter.summary()

def test_epoch(model, loader, mad_prediction):
    model.eval()
    meter = AverageMeter(['confidence_loss'], unpooled_metrics=True) if mad_prediction else AverageMeter(['confidence_loss', 'accuracy', 'ROC AUC'], unpooled_metrics=True)
    all_labels = []
    for data in tqdm(loader, total=len(loader)):
        try:
            with torch.no_grad():
                pred = model(data)
            affinity_loss = torch.tensor(0.0, dtype=torch.float, device=pred[0].device)
            accuracy = torch.tensor(0.0, dtype=torch.float, device=pred[0].device)
            if mad_prediction:
                labels = torch.cat([graph.mad for graph in data]).to(device) if isinstance(data, list) else data.mad
                norm_labels = sigmoid_function(labels)
                confidence_loss = F.mse_loss(pred, norm_labels)
                meter.add([confidence_loss.cpu().detach()])
            else:
                if isinstance(args.mad_classification_cutoff, list):
                    labels = torch.cat([graph.y_binned for graph in data]).to(device) if isinstance(data,list) else data.y_binned
                    confidence_loss = F.cross_entropy(pred, labels)
                else:
                    labels = torch.cat([graph.y for graph in data]).to(device) if isinstance(data, list) else data.y
#                     labels = labels.flatten()
                    confidence_loss = F.binary_cross_entropy_with_logits(pred, labels)
                    accuracy = torch.mean((labels == (pred > 0).float()).float())
                try:
                    roc_auc = roc_auc_score(labels.detach().cpu().numpy(), pred.detach().cpu().numpy())
                except ValueError as e:
                    if 'Only one class present in y_true. ROC AUC score is not defined in that case.' in str(e):
                        roc_auc = 0
                    else:
                        raise e
                meter.add([confidence_loss.cpu().detach(), accuracy.cpu().detach(), torch.tensor(roc_auc)])
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

    all_labels = torch.cat(all_labels)

    if mad_prediction:
        baseline_metric = ((all_labels - all_labels.mean()).abs()).mean()
    else:
        baseline_metric = all_labels.sum() / len(all_labels)
    results = meter.summary()
    results.update({'baseline_metric': baseline_metric})
    return meter.summary(), baseline_metric


def train(args, model, optimizer, scheduler, train_loader, val_loader, run_dir):
    best_val_metric = math.inf if args.main_metric_goal == 'min' else 0
    best_epoch = 0

    print("Starting training...")
    for epoch in range(args.n_epochs):
        logs = {}
        train_metrics = train_epoch(model, train_loader, optimizer, args.mad_prediction)
        print("Epoch {}: Training loss {:.4f}".format(epoch, train_metrics['confidence_loss']))
        val_metrics, baseline_metric = test_epoch(model, val_loader, args.mad_prediction)
        if args.mad_prediction:
            print("Epoch {}: Validation loss {:.4f}".format(epoch, val_metrics['confidence_loss']))
        else:
            print("Epoch {}: Validation loss {:.4f}  accuracy {:.4f}".format(epoch, val_metrics['confidence_loss'], val_metrics['accuracy']))

        if args.wandb:
            logs.update({'valinf_' + k: v for k, v in val_metrics.items()}, step=epoch + 1)
            logs.update({'train_' + k: v for k, v in train_metrics.items()}, step=epoch + 1)
            logs.update({'mean_mad' if args.mad_prediction else 'fraction_positives': baseline_metric,
                         'current_lr': optimizer.param_groups[0]['lr']})
            wandb.log(logs, step=epoch + 1)

        if scheduler:
            scheduler.step(val_metrics[args.main_metric])

        state_dict = model.module.state_dict() if device.type == 'cuda' else model.state_dict()

        if args.main_metric_goal == 'min' and val_metrics[args.main_metric] < best_val_metric or \
                args.main_metric_goal == 'max' and val_metrics[args.main_metric] > best_val_metric:
            best_val_metric = val_metrics[args.main_metric]
            best_epoch = epoch
            torch.save(state_dict, os.path.join(run_dir, 'best_model.pt'))
        if args.model_save_frequency > 0 and (epoch + 1) % args.model_save_frequency == 0:
            torch.save(state_dict, os.path.join(run_dir, f'model_epoch{epoch+1}.pt'))
        if args.best_model_save_frequency > 0 and (epoch + 1) % args.best_model_save_frequency == 0:
            shutil.copyfile(os.path.join(run_dir, 'best_model.pt'), os.path.join(run_dir, f'best_model_epoch{epoch+1}.pt'))

        torch.save({
            'epoch': epoch,
            'model': state_dict,
            'optimizer': optimizer.state_dict(),
        }, os.path.join(run_dir, 'last_model.pt'))

    print("Best Validation accuracy {} on Epoch {}".format(best_val_metric, best_epoch))

def construct_loader_origin(args_confidence, args, t_to_sigma):
    ## the only difference compared to construct_loader is that we set batch_size = 1
    ## and we used DataLoader not DataLoaderList
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
    val_dataset = PDBBind(cache_path=args.cache_path, split_path=args_confidence.split_val, keep_original=True, **common_args)

    loader_class = DataLoader
    train_loader = loader_class(dataset=train_dataset, batch_size=args_confidence.batch_size_preprocessing, num_workers=args_confidence.num_workers, shuffle=False, pin_memory=args.pin_memory)
    val_loader = loader_class(dataset=val_dataset, batch_size=args_confidence.batch_size_preprocessing, num_workers=args_confidence.num_workers, shuffle=False, pin_memory=args.pin_memory)
    infer_loader = loader_class(dataset=val_dataset, batch_size=args_confidence.batch_size_preprocessing, num_workers=args_confidence.num_workers, shuffle=False, pin_memory=args.pin_memory)

    return train_loader, val_loader, infer_loader

def construct_loader_confidence(args, device):
    common_args = {'cache_path': args.cache_path, 'original_model_dir': args.original_model_dir, 'device': device,
                   'inference_steps': args.inference_steps, 'samples_per_complex': args.samples_per_complex,
                   'limit_complexes': args.limit_complexes, 'all_atoms': args.all_atoms, 'balance': args.balance,
                   'mad_classification_cutoff': args.mad_classification_cutoff, 'use_original_model_cache': args.use_original_model_cache,
                   'cache_creation_id': args.cache_creation_id, "cache_ids_to_combine": args.cache_ids_to_combine,
                   "model_ckpt": args.ckpt,
                   "running_mode": args.running_mode,
                   "water_ratio": args.water_ratio,
                   "resample_steps": args.resample_steps}
    
    loader_class = DataListLoader if torch.cuda.is_available() else DataLoader
    exception_flag = False
    # construct original loader
    original_model_args = get_args(args.original_model_dir)
    t_to_sigma = partial(t_to_sigma_compl, args=original_model_args)
    train_loader, val_loader, infer_loader = construct_loader_origin(args, original_model_args, t_to_sigma)

    try:
        train_dataset = ConfidenceDataset(loader=train_loader, split=os.path.splitext(os.path.basename(args.split_train))[0], args=args, **common_args)
        train_loader = loader_class(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    except Exception as e:
        if 'The generated ligand positions with cache_id do not exist:' in str(e):
            print("HAPPENING | Encountered the following exception when loading the confidence train dataset:")
            print(str(e))
            print("HAPPENING | We are still continuing because we want to try to generate the validation dataset if it has not been created yet:")
            exception_flag = True
        else: raise e

    val_dataset = ConfidenceDataset(loader=val_loader, split=os.path.splitext(os.path.basename(args.split_val))[0], args=args, **common_args)
    val_loader = loader_class(dataset=val_dataset, batch_size=args.batch_size, shuffle=True)

    if exception_flag: raise Exception('We encountered the exception during train dataset loading: ', e)
    return train_loader, val_loader


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with open(f'{args.original_model_dir}/model_parameters.yml') as f:
        score_model_args = Namespace(**yaml.full_load(f))

    # construct loader
    train_loader, val_loader = construct_loader_confidence(args, device)
    model = get_model(score_model_args if args.transfer_weights else args, device, t_to_sigma=None, confidence_mode=True)
    optimizer, scheduler = get_optimizer_and_scheduler(args, model, scheduler_mode=args.main_metric_goal)

    if args.transfer_weights:
        print("HAPPENING | Transferring weights from original_model_dir to the new model after using original_model_dir's arguments to construct the new model.")
        checkpoint = torch.load(os.path.join(args.original_model_dir,args.ckpt), map_location=device)
        model_state_dict = model.state_dict()
        transfer_weights_dict = {k: v for k, v in checkpoint.items() if k in list(model_state_dict.keys())}
        model_state_dict.update(transfer_weights_dict)  # update the layers with the pretrained weights
        model.load_state_dict(model_state_dict)

    elif args.restart_dir:
        dict = torch.load(f'{args.restart_dir}/last_model.pt', map_location=torch.device('cpu'))
        model.module.load_state_dict(dict['model'], strict=True)
        optimizer.load_state_dict(dict['optimizer'])
        print("Restarting from epoch", dict['epoch'])

    numel = sum([p.numel() for p in model.parameters()])
    print('Model with', numel, 'parameters')

    if args.wandb:
        wandb.init(
            entity='xiaohan-kuang',
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

    train(args, model, optimizer, scheduler, train_loader, val_loader, run_dir)