import copy

import numpy as np
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from confidence.dataset import ListDataset
from utils import so3, torus
from utils.sampling import sampling, randomize_position_multiple
import torch
from utils.diffusion_utils import get_t_schedule
from utils.min_dist import match_points_and_get_distances

def loss_function(tr_pred, expand_tr_sigma, data, t_to_sigma, device, tr_weight=1, apply_mean=True):
    mean_dims = (0, 1) if apply_mean else 1

    # translation component
    tr_score = torch.cat([d.tr_score for d in data], dim=0) if device.type == 'cuda' else data.tr_score
    expand_tr_sigma = expand_tr_sigma.unsqueeze(-1).cpu()
    
    
    tr_loss = ((tr_pred.cpu() - tr_score) ** 2 * (expand_tr_sigma ** 2 + 1e-6)).mean(dim=mean_dims)
    
    if torch.isnan(tr_loss).any():
        print("NaN found in loss")
        tr_loss = torch.nan_to_num(tr_loss, nan=0.0)

    tr_base_loss = (tr_score ** 2 *  expand_tr_sigma ** 2).mean(dim=mean_dims).detach()

    loss = tr_loss * tr_weight
    return loss, tr_loss.detach(), tr_base_loss


class AverageMeter():
    def __init__(self, types, unpooled_metrics=False, intervals=1):
        self.types = types
        self.intervals = intervals
        self.count = 0 if intervals == 1 else torch.zeros(len(types), intervals)
        self.acc = {t: torch.zeros(intervals) for t in types}
        self.unpooled_metrics = unpooled_metrics

    def add(self, vals, interval_idx=None):
        if self.intervals == 1:
            self.count += 1 if vals[0].dim() == 0 else len(vals[0])
            for type_idx, v in enumerate(vals):
                self.acc[self.types[type_idx]] += v.sum() if self.unpooled_metrics else v
        else:
            for type_idx, v in enumerate(vals):
                self.count[type_idx].index_add_(0, interval_idx[type_idx], torch.ones(len(v)))
                if not torch.allclose(v, torch.tensor(0.0)):
                    self.acc[self.types[type_idx]].index_add_(0, interval_idx[type_idx], v)

    def summary(self):
        if self.intervals == 1:
            out = {k: v.item() / self.count for k, v in self.acc.items()}
            return out
        else:
            out = {}
            for i in range(self.intervals):
                for type_idx, k in enumerate(self.types):
                    out['int' + str(i) + '_' + k] = (
                            list(self.acc.values())[type_idx][i] / self.count[type_idx][i]).item()
            return out


def train_epoch(model, loader, optimizer, device, t_to_sigma, loss_fn, ema_weigths):
    model.train()
    meter = AverageMeter(['loss', 'tr_loss', 'tr_base_loss'])

    for data in tqdm(loader, total=len(loader)):
        if device.type == 'cuda' and len(data) == 1 or device.type == 'cpu' and data.num_graphs == 1:
            print("Skipping batch of size 1 since otherwise batchnorm would not work.")
        optimizer.zero_grad()
        try:
            tr_pred, expand_tr_sigma, expand_batch_idx = model(data)
     
            loss, tr_loss, tr_base_loss = \
                loss_fn(tr_pred, expand_tr_sigma, data=data, t_to_sigma=t_to_sigma, device=device)
                
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            ema_weigths.update(model.parameters())
            meter.add([loss.cpu().detach(), tr_loss, tr_base_loss])
            
        except RuntimeError as e:
            if 'out of memory' in str(e):
                print('| WARNING: ran out of memory, skipping batch')
                for p in model.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                torch.cuda.empty_cache()
                continue
            elif 'Input mismatch' in str(e):
                print('| WARNING: weird torch_cluster error, skipping batch')
                for p in model.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                torch.cuda.empty_cache()
                continue
            else:
                raise e

    return meter.summary()


def test_epoch(model, loader, device, t_to_sigma, loss_fn, test_sigma_intervals=False):
    model.eval()
    meter = AverageMeter(['loss', 'tr_loss', 'tr_base_loss'],
                         unpooled_metrics=True)

    if test_sigma_intervals:
        meter_all = AverageMeter(
            ['loss', 'tr_loss', 'tr_base_loss'],
            unpooled_metrics=True, intervals=10)

    for data in tqdm(loader, total=len(loader)):
        try:
            with torch.no_grad():
                tr_pred, expand_tr_sigma, expand_batch_idx = model(data)

            loss, tr_loss, tr_base_loss = \
                loss_fn(tr_pred, expand_tr_sigma, data=data, t_to_sigma=t_to_sigma, apply_mean=False, device=device)
            meter.add([loss.cpu().detach(), tr_loss, tr_base_loss])

            if test_sigma_intervals > 0:
                complex_t_tr  = torch.cat([d.complex_t['tr'] for d in data]) 
                sigma_index_tr = torch.round(complex_t_tr.cpu() * (10 - 1)).long()
                expand_sigma_index_tr = torch.index_select(sigma_index_tr, dim=0, index=expand_batch_idx.cpu())
                meter_all.add(
                    [loss.cpu().detach(), tr_loss, tr_base_loss],
                    [expand_sigma_index_tr, expand_sigma_index_tr, expand_sigma_index_tr, expand_sigma_index_tr])

        except RuntimeError as e:
            if 'out of memory' in str(e):
                print('| WARNING: ran out of memory, skipping batch')
                for p in model.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                torch.cuda.empty_cache()
                continue
            elif 'Input mismatch' in str(e):
                print('| WARNING: weird torch_cluster error, skipping batch')
                for p in model.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                torch.cuda.empty_cache()
                continue
            else:
                raise e

    out = meter.summary()
    if test_sigma_intervals > 0: out.update(meter_all.summary())
    return out