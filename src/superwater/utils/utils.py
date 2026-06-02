import os
import warnings

import torch
import yaml
from torch_geometric.nn.data_parallel import DataParallel

from superwater.models.all_atom_score_model import TensorProductScoreModel as AAScoreModel
from superwater.models.score_model import TensorProductScoreModel as CGScoreModel
from superwater.utils.diffusion_utils import get_timestep_embedding

# Backward-compatible aliases for the pre-1.0 checkpoint folder names that lived
# under workdir/. Maps old basename -> new basename under models/.
_MODEL_DIR_ALIASES = {
    "all_atoms_score_model_res15_17092": "water_score_res15",
    "confidence_model_17092_sigmoid_rr15": "water_confidence_res15_sigmoid",
}


def resolve_model_dir(path):
    """Return a usable model-checkpoint directory, falling back for renamed paths.

    Checkpoint folders moved from workdir/<cryptic_name> to models/<readable_name>.
    If ``path`` does not exist, try the known rename and a workdir->models swap,
    warn, and use it; otherwise return ``path`` unchanged so the caller raises an
    actionable error.
    """
    if os.path.isdir(path):
        return path
    candidates = []
    base = os.path.basename(os.path.normpath(path))
    if base in _MODEL_DIR_ALIASES:
        candidates.append(os.path.join("models", _MODEL_DIR_ALIASES[base]))
    if "workdir" in path.split(os.sep):
        candidates.append(path.replace("workdir", "models", 1))
    for candidate in candidates:
        if os.path.isdir(candidate):
            warnings.warn(
                f"Model directory '{path}' not found; using '{candidate}' instead. "
                f"Please update your paths to the models/ layout.", stacklevel=2)
            return candidate
    return path


def read_strings_from_txt(path):
    with open(path) as file:
        lines = file.readlines()
        return [line.rstrip() for line in lines]


def save_yaml_file(path, content):
    assert isinstance(path, str), f'path must be a string, got {path} which is a {type(path)}'
    content = yaml.dump(data=content)
    if '/' in path and os.path.dirname(path) and not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    with open(path, 'w') as f:
        f.write(content)


def get_optimizer_and_scheduler(args, model, scheduler_mode='min'):
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.w_decay)

    if args.scheduler == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=scheduler_mode, factor=0.7,
                                                               patience=args.scheduler_patience, min_lr=args.lr / 100)
    else:
        print('No scheduler')
        scheduler = None

    return optimizer, scheduler


def get_model(args, device, t_to_sigma, no_parallel=False, confidence_mode=False):
    if 'all_atoms' in args and args.all_atoms:
        model_class = AAScoreModel
    else:
        model_class = CGScoreModel

    timestep_emb_func = get_timestep_embedding(
        embedding_type=args.embedding_type,
        embedding_dim=args.sigma_embed_dim,
        embedding_scale=args.embedding_scale)

    lm_embedding_type = None
    if args.esm_embeddings_path is not None: lm_embedding_type = 'esm'

    model = model_class(t_to_sigma=t_to_sigma,
                        device=device,
                        timestep_emb_func=timestep_emb_func,
                        num_conv_layers=args.num_conv_layers,
                        lig_max_radius=args.max_radius,
                        scale_by_sigma=args.scale_by_sigma,
                        sigma_embed_dim=args.sigma_embed_dim,
                        ns=args.ns, nv=args.nv,
                        distance_embed_dim=args.distance_embed_dim,
                        cross_distance_embed_dim=args.cross_distance_embed_dim,
                        batch_norm=not args.no_batch_norm,
                        dropout=args.dropout,
                        use_second_order_repr=args.use_second_order_repr,
                        cross_max_distance=args.cross_max_distance,
                        dynamic_max_cross=args.dynamic_max_cross,
                        lm_embedding_type=lm_embedding_type,
                        confidence_mode=confidence_mode,
                        num_confidence_outputs=len(
                            args.rmsd_classification_cutoff) + 1 if 'rmsd_classification_cutoff' in args and isinstance(
                            args.rmsd_classification_cutoff, list) else 1)

    if device.type == 'cuda' and not no_parallel:
        model = DataParallel(model)
    model.to(device)
    return model


class ExponentialMovingAverage:
    """ from https://github.com/yang-song/score_sde_pytorch/blob/main/models/ema.py
    Maintains (exponential) moving average of a set of parameters. """

    def __init__(self, parameters, decay, use_num_updates=True):
        """
        Args:
          parameters: Iterable of `torch.nn.Parameter`; usually the result of
            `model.parameters()`.
          decay: The exponential decay.
          use_num_updates: Whether to use number of updates when computing
            averages.
        """
        if decay < 0.0 or decay > 1.0:
            raise ValueError('Decay must be between 0 and 1')
        self.decay = decay
        self.num_updates = 0 if use_num_updates else None
        self.shadow_params = [p.clone().detach()
                              for p in parameters if p.requires_grad]
        self.collected_params = []

    def update(self, parameters):
        """
        Update currently maintained parameters.
        Call this every time the parameters are updated, such as the result of
        the `optimizer.step()` call.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; usually the same set of
            parameters used to initialize this object.
        """
        decay = self.decay
        if self.num_updates is not None:
            self.num_updates += 1
            decay = min(decay, (1 + self.num_updates) / (10 + self.num_updates))
        one_minus_decay = 1.0 - decay
        with torch.no_grad():
            parameters = [p for p in parameters if p.requires_grad]
            for s_param, param in zip(self.shadow_params, parameters):
                s_param.sub_(one_minus_decay * (s_param - param))

    def copy_to(self, parameters):
        """
        Copy current parameters into given collection of parameters.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            updated with the stored moving averages.
        """
        parameters = [p for p in parameters if p.requires_grad]
        for s_param, param in zip(self.shadow_params, parameters):
            if param.requires_grad:
                param.data.copy_(s_param.data)

    def store(self, parameters):
        """
        Save the current parameters for restoring later.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            temporarily stored.
        """
        self.collected_params = [param.clone() for param in parameters]

    def restore(self, parameters):
        """
        Restore the parameters stored with the `store` method.
        Useful to validate the model with EMA parameters without affecting the
        original optimization process. Store the parameters before the
        `copy_to` method. After validation (or model saving), use this to
        restore the former parameters.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            updated with the stored parameters.
        """
        for c_param, param in zip(self.collected_params, parameters):
            param.data.copy_(c_param.data)

    def state_dict(self):
        return dict(decay=self.decay, num_updates=self.num_updates,
                    shadow_params=self.shadow_params)

    def load_state_dict(self, state_dict, device):
        self.decay = state_dict['decay']
        self.num_updates = state_dict['num_updates']
        self.shadow_params = [tensor.to(device) for tensor in state_dict['shadow_params']]
