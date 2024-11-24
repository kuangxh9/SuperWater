from e3nn import o3
import torch
from torch import nn
from torch.nn import functional as F
from torch_cluster import radius, radius_graph
from torch_scatter import scatter_mean
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter('ignore')
from models.score_model import AtomEncoder, TensorProductConvLayer, GaussianSmearing
from utils import so3, torus
from datasets.process_mols import lig_feature_dims, rec_residue_feature_dims, rec_atom_feature_dims
from torch_scatter import scatter_mean

class CrossAttentionLayer(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(CrossAttentionLayer, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)

    def forward(self, query_features, key_value_features):
        query = query_features.unsqueeze(1)  # (num_lig_nodes, 1, embed_dim)
        key = value = key_value_features.unsqueeze(1)  # (num_rec_nodes or num_atom_nodes, 1, embed_dim)
        attn_output, _ = self.attention(query, key, value)
        return attn_output.squeeze(1)  # (num_lig_nodes, embed_dim)

class SelfAttentionLayer(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(SelfAttentionLayer, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)

    def forward(self, features):
        query = key = value = features.unsqueeze(1)  # (num_nodes, 1, embed_dim)
        attn_output, _ = self.attention(query, key, value)
        return attn_output.squeeze(1) 


class TensorProductScoreModel(torch.nn.Module):
    def __init__(self, t_to_sigma, device, timestep_emb_func, in_lig_edge_features=4, sigma_embed_dim=32, sh_lmax=2,
                 ns=16, nv=4, num_conv_layers=2, lig_max_radius=5, rec_max_radius=30, cross_max_distance=250,
                 center_max_distance=30, distance_embed_dim=32, cross_distance_embed_dim=32, no_torsion=True,
                 scale_by_sigma=True, use_second_order_repr=False, batch_norm=True,
                 dynamic_max_cross=False, dropout=0.0, lm_embedding_type=False, confidence_mode=False,
                 confidence_dropout=0, confidence_no_batchnorm=False, num_confidence_outputs=1, recycle_output_size=0, mean_pool=True):
        super(TensorProductScoreModel, self).__init__()
        self.t_to_sigma = t_to_sigma
        self.in_lig_edge_features = in_lig_edge_features
        self.sigma_embed_dim = sigma_embed_dim
        self.lig_max_radius = lig_max_radius
        self.rec_max_radius = rec_max_radius
        self.cross_max_distance = cross_max_distance
        self.dynamic_max_cross = dynamic_max_cross
        self.center_max_distance = center_max_distance
        self.distance_embed_dim = distance_embed_dim
        self.cross_distance_embed_dim = cross_distance_embed_dim
        self.sh_irreps = o3.Irreps.spherical_harmonics(lmax=sh_lmax)
        self.ns, self.nv = ns, nv
        self.scale_by_sigma = scale_by_sigma
        self.device = device
        self.no_torsion = no_torsion
        self.num_conv_layers = num_conv_layers
        self.timestep_emb_func = timestep_emb_func
        self.confidence_mode = confidence_mode
        self.lig_rec_cross_attention = CrossAttentionLayer(embed_dim=ns, num_heads=4)        
        
        self.lig_node_embedding = AtomEncoder(emb_dim=ns, feature_dims=lig_feature_dims, sigma_embed_dim=sigma_embed_dim, additional_dim=recycle_output_size)
        self.lig_edge_embedding = nn.Sequential(nn.Linear(in_lig_edge_features + sigma_embed_dim + distance_embed_dim, ns),nn.ReLU(),nn.Dropout(dropout),nn.Linear(ns, ns))

        self.rec_node_embedding = AtomEncoder(emb_dim=ns, feature_dims=rec_residue_feature_dims, sigma_embed_dim=sigma_embed_dim, lm_embedding_type=lm_embedding_type)
        self.rec_edge_embedding = nn.Sequential(nn.Linear(sigma_embed_dim + distance_embed_dim, ns), nn.ReLU(), nn.Dropout(dropout),nn.Linear(ns, ns))

        self.atom_node_embedding = AtomEncoder(emb_dim=ns, feature_dims=rec_atom_feature_dims, sigma_embed_dim=sigma_embed_dim)
        self.atom_edge_embedding = nn.Sequential(nn.Linear(sigma_embed_dim + distance_embed_dim, ns), nn.ReLU(), nn.Dropout(dropout),nn.Linear(ns, ns))

        self.lr_edge_embedding = nn.Sequential(nn.Linear(sigma_embed_dim + cross_distance_embed_dim, ns), nn.ReLU(), nn.Dropout(dropout),nn.Linear(ns, ns))
        self.ar_edge_embedding = nn.Sequential(nn.Linear(sigma_embed_dim + distance_embed_dim, ns), nn.ReLU(), nn.Dropout(dropout),nn.Linear(ns, ns))
        self.la_edge_embedding = nn.Sequential(nn.Linear(sigma_embed_dim + cross_distance_embed_dim, ns), nn.ReLU(), nn.Dropout(dropout),nn.Linear(ns, ns))

        self.lig_distance_expansion = GaussianSmearing(0.0, lig_max_radius, distance_embed_dim)
        self.rec_distance_expansion = GaussianSmearing(0.0, rec_max_radius, distance_embed_dim)
        self.cross_distance_expansion = GaussianSmearing(0.0, cross_max_distance, cross_distance_embed_dim)
        
        if use_second_order_repr:
            irrep_seq = [
                f'{ns}x0e',
                f'{ns}x0e + {nv}x1o + {nv}x2e',
                f'{ns}x0e + {nv}x1o + {nv}x2e + {nv}x1e + {nv}x2o',
                f'{ns}x0e + {nv}x1o + {nv}x2e + {nv}x1e + {nv}x2o + {ns}x0o'
            ]
        else:
            irrep_seq = [
                f'{ns}x0e',
                f'{ns}x0e + {nv}x1o',
                f'{ns}x0e + {nv}x1o + {nv}x1e',
                f'{ns}x0e + {nv}x1o + {nv}x1e + {ns}x0o'
            ]

        # convolutional layers
        conv_layers = []
        for i in range(num_conv_layers):
            in_irreps = irrep_seq[min(i, len(irrep_seq) - 1)]
            out_irreps = irrep_seq[min(i + 1, len(irrep_seq) - 1)]
            parameters = {
                'in_irreps': in_irreps,
                'sh_irreps': self.sh_irreps,
                'out_irreps': out_irreps,
                'n_edge_features': 3 * ns,
                'residual': False,
                'batch_norm': batch_norm,
                'dropout': dropout
            }

            for _ in range(9): # 3 intra & 6 inter per each layer
                conv_layers.append(TensorProductConvLayer(**parameters))

        self.conv_layers = nn.ModuleList(conv_layers)

        # confidence and affinity prediction layers
        if self.confidence_mode:
            output_confidence_dim = num_confidence_outputs

            self.confidence_predictor = nn.Sequential(
                nn.Linear(2 * self.ns if num_conv_layers >= 3 else self.ns, ns),
                nn.BatchNorm1d(ns) if not confidence_no_batchnorm else nn.Identity(),
                nn.ReLU(),
                nn.Dropout(confidence_dropout),
                nn.Linear(ns, ns),
                nn.BatchNorm1d(ns) if not confidence_no_batchnorm else nn.Identity(),
                nn.ReLU(),
                nn.Dropout(confidence_dropout),
                nn.Linear(ns, output_confidence_dim)
            )

        else:
            # convolution for translational and rotational scores
            self.center_distance_expansion = GaussianSmearing(0.0, center_max_distance, distance_embed_dim)
            self.center_edge_embedding = nn.Sequential(
                nn.Linear(distance_embed_dim + sigma_embed_dim, ns),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(ns, ns)
            )

            self.final_conv = TensorProductConvLayer(
                in_irreps=self.conv_layers[-1].out_irreps,
                sh_irreps=self.sh_irreps,
                out_irreps=f'1x1o + 1x1e',
                n_edge_features=2 * ns,
                residual=False,
                dropout=dropout,
                batch_norm=batch_norm
            )

            self.tr_final_layer = nn.Sequential(nn.Linear(1 + sigma_embed_dim, ns), nn.Dropout(dropout), nn.ReLU(), nn.Linear(ns, 1))

    def forward(self, data):
        if not self.confidence_mode:
            tr_sigma = self.t_to_sigma(*[data.complex_t[noise_type] for noise_type in ['tr']])
        else:
            tr_sigma = data.complex_t['tr']
            
        ll_distance_cutoff = (tr_sigma * 3 + 20).unsqueeze(1)
        
        lig_node_attr = self.build_lig_node_attr(data)
        lig_node_attr = self.lig_node_embedding(lig_node_attr)

        # build receptor graph
        rec_node_attr, rec_edge_index, rec_edge_attr, rec_edge_sh = self.build_rec_conv_graph(data)
        rec_node_attr = self.rec_node_embedding(rec_node_attr)
        rec_edge_attr = self.rec_edge_embedding(rec_edge_attr)

        # build atom graph
        atom_node_attr, atom_edge_index, atom_edge_attr, atom_edge_sh = self.build_atom_conv_graph(data)
        atom_node_attr = self.atom_node_embedding(atom_node_attr)
        atom_edge_attr = self.atom_edge_embedding(atom_edge_attr)
        
        # build cross graph
        lr_cross_distance_cutoff = (tr_sigma * 3 + 20).unsqueeze(1) if self.dynamic_max_cross else self.cross_max_distance
        
        lr_edge_index, lr_edge_attr, lr_edge_sh, la_edge_index, la_edge_attr, \
            la_edge_sh, ar_edge_index, ar_edge_attr, ar_edge_sh = self.build_cross_conv_graph(data, lr_cross_distance_cutoff)
        
        lr_edge_attr= self.lr_edge_embedding(lr_edge_attr)
        la_edge_attr = self.la_edge_embedding(la_edge_attr)
        ar_edge_attr = self.ar_edge_embedding(ar_edge_attr)

        for l in range(self.num_conv_layers):
            lr_edge_attr_ = torch.cat([lr_edge_attr, lig_node_attr[lr_edge_index[0], :self.ns], rec_node_attr[lr_edge_index[1], :self.ns]], -1)
            lr_update = self.conv_layers[9*l+1](rec_node_attr, lr_edge_index, lr_edge_attr_, lr_edge_sh,
                                                out_nodes=lig_node_attr.shape[0])

            la_edge_attr_ = torch.cat([la_edge_attr, lig_node_attr[la_edge_index[0], :self.ns], atom_node_attr[la_edge_index[1], :self.ns]], -1)
            la_update = self.conv_layers[9*l+2](atom_node_attr, la_edge_index, la_edge_attr_, la_edge_sh,
                                                out_nodes=lig_node_attr.shape[0])

            if l != self.num_conv_layers-1:  # last layer optimisation
                # ATOM UPDATES
                atom_edge_attr_ = torch.cat([atom_edge_attr, atom_node_attr[atom_edge_index[0], :self.ns], atom_node_attr[atom_edge_index[1], :self.ns]], -1)
                atom_update = self.conv_layers[9*l+3](atom_node_attr, atom_edge_index, atom_edge_attr_, atom_edge_sh)

                al_edge_attr_ = torch.cat([la_edge_attr, atom_node_attr[la_edge_index[1], :self.ns], lig_node_attr[la_edge_index[0], :self.ns]], -1)
                al_update = self.conv_layers[9*l+4](lig_node_attr, torch.flip(la_edge_index, dims=[0]), al_edge_attr_,
                                                    la_edge_sh, out_nodes=atom_node_attr.shape[0])

                ar_edge_attr_ = torch.cat([ar_edge_attr, atom_node_attr[ar_edge_index[0], :self.ns], rec_node_attr[ar_edge_index[1], :self.ns]],-1)
                ar_update = self.conv_layers[9*l+5](rec_node_attr, ar_edge_index, ar_edge_attr_, ar_edge_sh, out_nodes=atom_node_attr.shape[0])

                # RECEPTOR updates
                rec_edge_attr_ = torch.cat([rec_edge_attr, rec_node_attr[rec_edge_index[0], :self.ns], rec_node_attr[rec_edge_index[1], :self.ns]], -1)
                rec_update = self.conv_layers[9*l+6](rec_node_attr, rec_edge_index, rec_edge_attr_, rec_edge_sh)

                rl_edge_attr_ = torch.cat([lr_edge_attr, rec_node_attr[lr_edge_index[1], :self.ns], lig_node_attr[lr_edge_index[0], :self.ns]], -1)
                rl_update = self.conv_layers[9*l+7](lig_node_attr, torch.flip(lr_edge_index, dims=[0]), rl_edge_attr_,
                                                    lr_edge_sh, out_nodes=rec_node_attr.shape[0])

                ra_edge_attr_ = torch.cat([ar_edge_attr, rec_node_attr[ar_edge_index[1], :self.ns], atom_node_attr[ar_edge_index[0], :self.ns]], -1)
                ra_update = self.conv_layers[9*l+8](atom_node_attr, torch.flip(ar_edge_index, dims=[0]), ra_edge_attr_,
                                                    ar_edge_sh, out_nodes=rec_node_attr.shape[0])

            # padding original features and update features with residual updates
            lig_node_attr = F.pad(lig_node_attr, (0, la_update.shape[-1] - lig_node_attr.shape[-1]))
            lig_node_attr = lig_node_attr + la_update + lr_update

            if l != self.num_conv_layers - 1:  # last layer optimisation
                atom_node_attr = F.pad(atom_node_attr, (0, atom_update.shape[-1] - rec_node_attr.shape[-1]))
                atom_node_attr = atom_node_attr + atom_update + al_update + ar_update
                rec_node_attr = F.pad(rec_node_attr, (0, rec_update.shape[-1] - rec_node_attr.shape[-1]))
                rec_node_attr = rec_node_attr + rec_update + ra_update + rl_update

        # confidence and affinity prediction
        if self.confidence_mode:
            scalar_lig_attr = torch.cat([lig_node_attr[:,:self.ns],lig_node_attr[:,-self.ns:]], dim=1) if self.num_conv_layers >= 3 else lig_node_attr[:,:self.ns]
            confidence = self.confidence_predictor(scalar_lig_attr).squeeze(dim=-1)
            return confidence

        center_edge_index, center_edge_attr, center_edge_sh = self.build_center_conv_graph(data)
        center_edge_attr = self.center_edge_embedding(center_edge_attr)
        center_edge_attr = torch.cat([center_edge_attr, lig_node_attr[center_edge_index[1], :self.ns]], -1)
        
        global_pred = self.final_conv(lig_node_attr, center_edge_index, center_edge_attr, center_edge_sh, out_nodes=lig_node_attr.shape[0], scatter_arange = True)

        if torch.isnan(global_pred).any():
            print("NaN found in global_pred")
            global_pred = torch.nan_to_num(global_pred, nan=0.0)
            
        tr_pred = global_pred[:, :3] + global_pred[:, 3:6]
        data.graph_sigma_emb = self.timestep_emb_func(data.complex_t['tr'])

        tr_norm = torch.linalg.vector_norm(tr_pred, dim=1).unsqueeze(1)

        expand_sigma_emb = torch.index_select(data.graph_sigma_emb, dim=0, index=data['ligand'].batch)

        tr_pred = tr_pred / tr_norm * self.tr_final_layer(torch.cat([tr_norm, expand_sigma_emb], dim=1))
        
        expand_tr_sigma = torch.index_select(tr_sigma, dim=0, index=data['ligand'].batch)
        if self.scale_by_sigma:
            tr_pred = tr_pred / expand_tr_sigma.unsqueeze(1)

        return tr_pred, expand_tr_sigma, data['ligand'].batch

    def build_lig_node_attr(self, data):        
        data['ligand'].node_sigma_emb = self.timestep_emb_func(data['ligand'].node_t['tr'])
        node_attr = torch.cat([data['ligand'].x, data['ligand'].node_sigma_emb], 1)
        return node_attr
    
    def build_rec_conv_graph(self, data):
        data['receptor'].node_sigma_emb = self.timestep_emb_func(data['receptor'].node_t['tr'])
        node_attr = torch.cat([data['receptor'].x, data['receptor'].node_sigma_emb], 1)

        edge_index = data['receptor', 'receptor'].edge_index
        src, dst = edge_index
        edge_vec = data['receptor'].pos[dst.long()] - data['receptor'].pos[src.long()]

        edge_length_emb = self.rec_distance_expansion(edge_vec.norm(dim=-1))
        edge_sigma_emb = data['receptor'].node_sigma_emb[edge_index[0].long()]
        edge_attr = torch.cat([edge_sigma_emb, edge_length_emb], 1)
        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization='component')

        return node_attr, edge_index, edge_attr, edge_sh

    def build_atom_conv_graph(self, data):
        # build the graph between receptor atoms
        data['atom'].node_sigma_emb = self.timestep_emb_func(data['atom'].node_t['tr'])
        node_attr = torch.cat([data['atom'].x, data['atom'].node_sigma_emb], 1)

        edge_index = data['atom', 'atom'].edge_index
        src, dst = edge_index
        edge_vec = data['atom'].pos[dst.long()] - data['atom'].pos[src.long()]

        edge_length_emb = self.lig_distance_expansion(edge_vec.norm(dim=-1))
        edge_sigma_emb = data['atom'].node_sigma_emb[edge_index[0].long()]
        edge_attr = torch.cat([edge_sigma_emb, edge_length_emb], 1)
        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization='component')

        return node_attr, edge_index, edge_attr, edge_sh

    def build_cross_conv_graph(self, data, lr_cross_distance_cutoff):
        data['ligand'].pos = data['ligand'].pos.float()
        
        if torch.is_tensor(lr_cross_distance_cutoff):
            # different cutoff for every graph
            lr_edge_index = radius(data['receptor'].pos / lr_cross_distance_cutoff[data['receptor'].batch],
                                data['ligand'].pos / lr_cross_distance_cutoff[data['ligand'].batch], 1,
                                data['receptor'].batch, data['ligand'].batch, max_num_neighbors=10000)
        else:
            lr_edge_index = radius(data['receptor'].pos, data['ligand'].pos, lr_cross_distance_cutoff,
                            data['receptor'].batch, data['ligand'].batch, max_num_neighbors=10000)
                    
        lr_edge_vec = data['receptor'].pos[lr_edge_index[1].long()] - data['ligand'].pos[lr_edge_index[0].long()]
        lr_edge_length_emb = self.cross_distance_expansion(lr_edge_vec.norm(dim=-1))
        lr_edge_sigma_emb = data['ligand'].node_sigma_emb[lr_edge_index[0].long()]
        lr_edge_attr = torch.cat([lr_edge_sigma_emb, lr_edge_length_emb], 1)
        lr_edge_sh = o3.spherical_harmonics(self.sh_irreps, lr_edge_vec, normalize=True, normalization='component')

        cutoff_d = lr_cross_distance_cutoff[data['ligand'].batch[lr_edge_index[0]]].squeeze() \
            if torch.is_tensor(lr_cross_distance_cutoff) else lr_cross_distance_cutoff

        la_edge_index = radius(data['atom'].pos, data['ligand'].pos, self.lig_max_radius,
                               data['atom'].batch, data['ligand'].batch, max_num_neighbors=10000)

        la_edge_vec = data['atom'].pos[la_edge_index[1].long()] - data['ligand'].pos[la_edge_index[0].long()]
        la_edge_length_emb = self.cross_distance_expansion(la_edge_vec.norm(dim=-1))
        la_edge_sigma_emb = data['ligand'].node_sigma_emb[la_edge_index[0].long()]
        la_edge_attr = torch.cat([la_edge_sigma_emb, la_edge_length_emb], 1)
        la_edge_sh = o3.spherical_harmonics(self.sh_irreps, la_edge_vec, normalize=True, normalization='component')

        ar_edge_index = data['atom', 'receptor'].edge_index

        valid_indices = (ar_edge_index[1] < data['receptor'].pos.size(0)) & (ar_edge_index[0] < data['atom'].pos.size(0))
        ar_edge_index = ar_edge_index[:, valid_indices]

        ar_edge_vec = data['receptor'].pos[ar_edge_index[1].long()] - data['atom'].pos[ar_edge_index[0].long()]
        ar_edge_length_emb = self.rec_distance_expansion(ar_edge_vec.norm(dim=-1))
        ar_edge_sigma_emb = data['atom'].node_sigma_emb[ar_edge_index[0].long()]
        ar_edge_attr = torch.cat([ar_edge_sigma_emb, ar_edge_length_emb], 1)
        ar_edge_sh = o3.spherical_harmonics(self.sh_irreps, ar_edge_vec, normalize=True, normalization='component')
        
        return lr_edge_index, lr_edge_attr, lr_edge_sh, la_edge_index, la_edge_attr, \
               la_edge_sh, ar_edge_index, ar_edge_attr, ar_edge_sh
    
    def build_center_conv_graph(self, data):
        edge_index = torch.cat([data['ligand'].batch.unsqueeze(0), torch.arange(len(data['ligand'].batch)).to(data['ligand'].x.device).unsqueeze(0)], dim=0)

        center_pos, count = torch.zeros((data.num_graphs, 3)).to(data['ligand'].x.device), torch.zeros((data.num_graphs, 3)).to(data['ligand'].x.device)
        center_pos.index_add_(0, index=data['ligand'].batch, source=data['ligand'].pos)
        center_pos = center_pos / torch.bincount(data['ligand'].batch).unsqueeze(1)

        edge_vec = data['ligand'].pos[edge_index[1]] - center_pos[edge_index[0]]
        edge_attr = self.center_distance_expansion(edge_vec.norm(dim=-1))
        edge_sigma_emb = data['ligand'].node_sigma_emb[edge_index[1].long()]
        edge_attr = torch.cat([edge_attr, edge_sigma_emb], 1)
        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization='component')
        return edge_index, edge_attr, edge_sh

    def build_bond_conv_graph(self, data):
        bonds = data['ligand', 'ligand'].edge_index[:, data['ligand'].edge_mask].long()
        bond_pos = (data['ligand'].pos[bonds[0]] + data['ligand'].pos[bonds[1]]) / 2
        bond_batch = data['ligand'].batch[bonds[0]]
        edge_index = radius(data['ligand'].pos, bond_pos, self.lig_max_radius, batch_x=data['ligand'].batch, batch_y=bond_batch)

        edge_vec = data['ligand'].pos[edge_index[1]] - bond_pos[edge_index[0]]
        edge_attr = self.lig_distance_expansion(edge_vec.norm(dim=-1))

        edge_attr = self.final_edge_embedding(edge_attr)
        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization='component')

        return bonds, edge_index, edge_attr, edge_sh

    
class RecycleNet(torch.nn.Module):
    def __init__(self, t_to_sigma, device, timestep_emb_func, in_lig_edge_features=4, sigma_embed_dim=32, sh_lmax=2,
                 ns=16, nv=4, num_conv_layers=2, lig_max_radius=5, rec_max_radius=30, cross_max_distance=250,
                 center_max_distance=30, distance_embed_dim=32, cross_distance_embed_dim=32, no_torsion=True,
                 scale_by_sigma=True, use_second_order_repr=False, batch_norm=True,
                 dynamic_max_cross=False, dropout=0.0, lm_embedding_type=False, confidence_mode=False,
                 confidence_dropout=0, confidence_no_batchnorm=False, num_confidence_outputs=1, recycle_output_size=3):
        super(RecycleNet, self).__init__()

        self.recycle_output_size = recycle_output_size
        self.N_cycle = 3
        self.score_model = TensorProductScoreModel(t_to_sigma, device, timestep_emb_func, in_lig_edge_features, sigma_embed_dim, sh_lmax,
                                                   ns, nv, num_conv_layers, lig_max_radius, rec_max_radius, cross_max_distance,
                                                   center_max_distance, distance_embed_dim, cross_distance_embed_dim, no_torsion,
                                                   scale_by_sigma, use_second_order_repr, batch_norm,
                                                   dynamic_max_cross, dropout, lm_embedding_type, confidence_mode,
                                                   confidence_dropout, confidence_no_batchnorm, num_confidence_outputs, recycle_output_size)

    def forward(self, data):
        training_mode = data[0]['training_mode']

        device = data['ligand'].x.device
        recycle_output = torch.zeros(data['ligand'].x.shape[0], self.recycle_output_size).to(device)
        data['ligand'].x = torch.cat((data['ligand'].x, recycle_output), dim=1)
        
        for recyc in range(self.N_cycle):
            if training_mode:
                recycle_output = recycle_output.detach()
            recycle_output, expand_tr_sigma, lig_batch  = self.score_model(data)
            data['ligand'].x[:, -self.recycle_output_size:] = recycle_output

        return recycle_output, expand_tr_sigma, lig_batch    