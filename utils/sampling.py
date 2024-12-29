import numpy as np
import torch
from torch_geometric.loader import DataLoader

from utils.diffusion_utils import modify_conformer, set_time
from utils.torsion import modify_conformer_torsion_angles
from scipy.spatial.transform import Rotation as R
# from utils.visualise import save_water_to_pdb_file
import os

def randomize_position_multiple(data_list, no_random, tr_sigma_max, water_num = 100):
    for complex_graph in data_list:
        if not no_random: 
            complex_graph['ligand'].pos = torch.normal(mean=0, std=tr_sigma_max, size=(water_num, 3))
            complex_graph['ligand'].x = complex_graph['ligand'].x[0, :].repeat(water_num, 1)
            complex_graph['ligand'].batch = torch.zeros(water_num, dtype=torch.int)

def sampling(data_list, model, inference_steps, tr_schedule, device, t_to_sigma, model_args,
             no_random=False, ode=False, save_visualization=False, visualisation_dir='inference_out/diff_process', confidence_model=None, confidence_data_list=None,
             confidence_model_args=None, batch_size=32, no_final_step_noise=False):

    N = len(data_list)

    for t_idx in range(inference_steps):
        t_tr = tr_schedule[t_idx]
        dt_tr = tr_schedule[t_idx] - tr_schedule[t_idx + 1] if t_idx < inference_steps - 1 else tr_schedule[t_idx]
       

        loader = DataLoader(data_list, batch_size=batch_size)
        new_data_list = []

        for complex_graph_batch in loader:
            b = complex_graph_batch.num_graphs
            complex_graph_batch = complex_graph_batch.to(device)

            tr_sigma = t_to_sigma(t_tr)
            set_time(complex_graph_batch, t_tr, b, model_args.all_atoms, device)
            
            with torch.no_grad():
                tr_score, _, _ = model(complex_graph_batch)
            
            if torch.isnan(tr_score).any():
                print("stop")
            tr_g = tr_sigma * torch.sqrt(torch.tensor(2 * np.log(model_args.tr_sigma_max / model_args.tr_sigma_min)))

            if ode:
                tr_perturb = (0.5 * tr_g ** 2 * dt_tr * tr_score.cpu()).cpu()
            else:
                tr_z = torch.zeros((len(tr_score), 3)) if no_random or (no_final_step_noise and t_idx == inference_steps - 1) \
                    else torch.normal(mean=0, std=1, size=(len(tr_score), 3))
                tr_perturb = (tr_g ** 2 * dt_tr * tr_score.cpu() + tr_g * np.sqrt(dt_tr) * tr_z).cpu()
            
            for i, complex_graph in enumerate(complex_graph_batch.to('cpu').to_data_list()):
                num_water = complex_graph['ligand'].pos.shape[0]
                new_data_list.extend([modify_conformer(complex_graph, tr_perturb[i*num_water:i*num_water + num_water])])
        data_list = new_data_list

        
        if save_visualization:
            os.makedirs(visualisation_dir, exist_ok=True)
            for complex_graph in data_list:
                file_path = f"{visualisation_dir}/water_{complex_graph['name'][0]}_L{inference_steps}.pdb"
                save_water_to_pdb_file(complex_graph, file_path, t_idx)

    with torch.no_grad():
        if confidence_model is not None:
            loader = DataLoader(data_list, batch_size=batch_size)
            confidence_loader = iter(DataLoader(confidence_data_list, batch_size=batch_size))
            confidence = []
            for complex_graph_batch in loader:
                complex_graph_batch = complex_graph_batch.to(device)
                if confidence_data_list is not None:
                    confidence_complex_graph_batch = next(confidence_loader).to(device)
                    confidence_complex_graph_batch['ligand'].pos = complex_graph_batch['ligand'].pos
                    set_time(confidence_complex_graph_batch, 0, N, confidence_model_args.all_atoms, device)
                    confidence.append(confidence_model(confidence_complex_graph_batch))
                else:
                    confidence.append(confidence_model(complex_graph_batch))
            confidence = torch.cat(confidence, dim=0)
        else:
            confidence = None
    
    return data_list, confidence

def save_water_to_pdb_file(complex_graph, file_path, model_idx):
    with open(file_path, "a") as pdb_file:
        pdb_file.write(f"MODEL\n")
        water_coords = complex_graph['ligand'].pos + complex_graph.original_center
        for i, coords in enumerate(water_coords, start=1):
            line = f'HETATM{i:>5}  O   HOH A{1:>4}    {coords[0]:8.3f}{coords[1]:8.3f}{coords[2]:8.3f}  1.00  0.00           O\n'
            pdb_file.write(line)
        pdb_file.write("ENDMDL\n")