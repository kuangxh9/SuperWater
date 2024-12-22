
from argparse import ArgumentParser,FileType

def parse_train_args():
    # General arguments
    parser = ArgumentParser()
    parser.add_argument('--config', type=FileType(mode='r'), default=None)
    parser.add_argument('--log_dir', type=str, default='workdir', help='Folder in which to save model and logs')
    parser.add_argument('--restart_dir', type=str, help='Folder of previous training model from which to restart')
    parser.add_argument('--cache_path', type=str, default='data/cache', help='Folder from where to load/restore cached dataset')
    parser.add_argument('--data_dir', type=str, default='data/waterbind/', help='Folder containing original structures')
    parser.add_argument('--split_train', type=str, default='data/splits/train_res15.txt', help='Path of file defining the split')
    parser.add_argument('--split_val', type=str, default='data/splits/val_res15.txt', help='Path of file defining the split')
    parser.add_argument('--split_test', type=str, default='data/splits/test_res15', help='Path of file defining the split')
    parser.add_argument('--test_sigma_intervals', action='store_true', default=False, help='Whether to log loss per noise interval')
    parser.add_argument('--val_inference_freq', type=int, default=None, help='Frequency of epochs for which to run expensive inference on val data')
    parser.add_argument('--train_inference_freq', type=int, default=None, help='Frequency of epochs for which to run expensive inference on train data')
    parser.add_argument('--inference_steps', type=int, default=20, help='Number of denoising steps for inference on val')
    parser.add_argument('--inference_earlystop_goal', type=str, default='max', help='Whether to maximize or minimize metric')
    parser.add_argument('--wandb', action='store_true', default=False, help='')
    parser.add_argument('--project', type=str, default='superwater_train', help='')
    parser.add_argument('--run_name', type=str, default='', help='')
    parser.add_argument('--cudnn_benchmark', action='store_true', default=False, help='CUDA optimization parameter for faster training')
    parser.add_argument('--num_dataloader_workers', type=int, default=0, help='Number of workers for dataloader')
    parser.add_argument('--pin_memory', action='store_true', default=False, help='pin_memory arg of dataloader')

    # Training arguments
    parser.add_argument('--n_epochs', type=int, default=400, help='Number of epochs for training')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--scheduler', type=str, default=None, help='LR scheduler')
    parser.add_argument('--scheduler_patience', type=int, default=20, help='Patience of the LR scheduler')
    parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate')
    parser.add_argument('--restart_lr', type=float, default=None, help='If this is not none, the lr of the optimizer will be overwritten with this value when restarting from a checkpoint.')
    parser.add_argument('--w_decay', type=float, default=0.0, help='Weight decay added to loss')
    parser.add_argument('--num_workers', type=int, default=1, help='Number of workers for preprocessing')
    parser.add_argument('--use_ema', action='store_true', default=False, help='Whether or not to use ema for the model weights')
    parser.add_argument('--ema_rate', type=float, default=0.999, help='decay rate for the exponential moving average model parameters ')

    # Dataset
    parser.add_argument('--limit_complexes', type=int, default=0, help='If positive, the number of training and validation complexes is capped')
    parser.add_argument('--all_atoms', action='store_true', default=False, help='Whether to use the all atoms model')
    parser.add_argument('--receptor_radius', type=float, default=30, help='Cutoff on distances for receptor edges')
    parser.add_argument('--c_alpha_max_neighbors', type=int, default=10, help='Maximum number of neighbors for each residue')
    parser.add_argument('--atom_radius', type=float, default=5, help='Cutoff on distances for atom connections')
    parser.add_argument('--atom_max_neighbors', type=int, default=8, help='Maximum number of atom neighbours for receptor')
    parser.add_argument('--matching_popsize', type=int, default=20, help='Differential evolution popsize parameter in matching')
    parser.add_argument('--matching_maxiter', type=int, default=20, help='Differential evolution maxiter parameter in matching')
    parser.add_argument('--max_lig_size', type=int, default=None, help='Maximum number of heavy atoms in ligand')
    parser.add_argument('--remove_hs', action='store_true', default=False, help='remove Hs')
    parser.add_argument('--num_conformers', type=int, default=1, help='Number of conformers to match to each ligand')
    parser.add_argument('--esm_embeddings_path', type=str, default=None, help='If this is set then the LM embeddings at that path will be used for the receptor features')

    # Diffusion
    parser.add_argument('--tr_weight', type=float, default=1, help='Weight of translation loss')
    parser.add_argument('--tr_sigma_min', type=float, default=0.1, help='Minimum sigma for translational component')
    parser.add_argument('--tr_sigma_max', type=float, default=30, help='Maximum sigma for translational component')

    # Score Model
    parser.add_argument('--num_conv_layers', type=int, default=2, help='Number of interaction layers')
    parser.add_argument('--max_radius', type=float, default=5.0, help='Radius cutoff for geometric graph')
    parser.add_argument('--scale_by_sigma', action='store_true', default=True, help='Whether to normalise the score')
    parser.add_argument('--ns', type=int, default=16, help='Number of hidden features per node of order 0')
    parser.add_argument('--nv', type=int, default=4, help='Number of hidden features per node of order >0')
    parser.add_argument('--distance_embed_dim', type=int, default=32, help='Embedding size for the distance')
    parser.add_argument('--cross_distance_embed_dim', type=int, default=32, help='Embeddings size for the cross distance')
    parser.add_argument('--no_batch_norm', action='store_true', default=False, help='If set, it removes the batch norm')
    parser.add_argument('--use_second_order_repr', action='store_true', default=False, help='Whether to use only up to first order representations or also second')
    parser.add_argument('--cross_max_distance', type=float, default=80, help='Maximum cross distance in case not dynamic')
    parser.add_argument('--dynamic_max_cross', action='store_true', default=False, help='Whether to use the dynamic distance cutoff')
    parser.add_argument('--dropout', type=float, default=0.0, help='MLP dropout')
    parser.add_argument('--embedding_type', type=str, default="sinusoidal", help='Type of diffusion time embedding')
    parser.add_argument('--sigma_embed_dim', type=int, default=32, help='Size of the embedding of the diffusion time')
    parser.add_argument('--embedding_scale', type=int, default=1000, help='Parameter of the diffusion time embedding')
    
    args = parser.parse_args()
    return args

def parse_confidence_args():
    # General arguments
    parser = ArgumentParser()
    parser.add_argument('--config', type=FileType(mode='r'), default=None)
    parser.add_argument('--original_model_dir', type=str, default='workdir', help='Path to folder with trained model and hyperparameters')
    parser.add_argument('--original_pdb_dir', type=str, default='data/waterbind', help='Path to folder with original PDB file downloaded from PDB website')
    parser.add_argument('--restart_dir', type=str, default=None, help='')
    parser.add_argument('--use_original_model_cache', action='store_true', default=False, help='If this is true, the same dataset as in the original model will be used. Otherwise, the dataset parameters are used.')
    parser.add_argument('--data_dir', type=str, default='data/waterbind/', help='Folder containing original structures')
    parser.add_argument('--ckpt', type=str, default='best_model.pt', help='Checkpoint to use inside the folder')
    parser.add_argument('--model_save_frequency', type=int, default=0, help='Frequency with which to save the last model. If 0, then only the early stopping criterion best model is saved and overwritten.')
    parser.add_argument('--best_model_save_frequency', type=int, default=0, help='Frequency with which to save the best model. If 0, then only the early stopping criterion best model is saved and overwritten.')
    parser.add_argument('--run_name', type=str, default='test_confidence', help='')
    parser.add_argument('--project', type=str, default='diffwater_confidence', help='')
    parser.add_argument('--split_train', type=str, default='data/splits/train_res15.txt', help='Path of file defining the split')
    parser.add_argument('--split_val', type=str, default='data/splits/val_res15.txt', help='Path of file defining the split')
    parser.add_argument('--split_test', type=str, default='data/splits/test_res15', help='Path of file defining the split')
    
    # Inference parameters for creating the positions and mads that the confidence predictor will be trained on.
    parser.add_argument('--cache_path', type=str, default='data/cache_confidence', help='Folder from where to load/restore cached dataset')
    parser.add_argument('--cache_ids_to_combine', nargs='+', type=str, default=None, help='')
    parser.add_argument('--cache_creation_id', type=int, default=None, help='number of times that inference is run on the full dataset before concatenating it and coming up with the full confidence dataset')
    parser.add_argument('--wandb', action='store_true', default=False, help='')
    parser.add_argument('--inference_steps', type=int, default=2, help='Number of denoising steps')
    parser.add_argument('--samples_per_complex', type=int, default=1, help='')
    parser.add_argument('--balance', action='store_true', default=False, help='If this is true than we do not force the samples seen during training to be the same amount of negatives as positives')
    parser.add_argument('--mad_prediction', action='store_true', default=False, help='')
    parser.add_argument('--mad_classification_cutoff', type=float, default=1, help='MAD value below which a prediction is considered a postitive. This can also be multiple cutoffs.')
    
    parser.add_argument('--log_dir', type=str, default='workdir', help='')
    parser.add_argument('--main_metric', type=str, default='confidence_loss', help='Metric to track for early stopping. Mostly [loss, accuracy, ROC AUC]')
    parser.add_argument('--main_metric_goal', type=str, default='min', help='Can be [min, max]')
    parser.add_argument('--transfer_weights', action='store_true', default=False, help='')
    parser.add_argument('--batch_size', type=int, default=5, help='')
    parser.add_argument('--batch_size_preprocessing', type=int, default=1, help='Number of workers')
    parser.add_argument('--lr', type=float, default=1e-3, help='')
    parser.add_argument('--w_decay', type=float, default=0.0, help='')
    parser.add_argument('--scheduler', type=str, default='plateau', help='')
    parser.add_argument('--scheduler_patience', type=int, default=50, help='')
    parser.add_argument('--n_epochs', type=int, default=5, help='')
    
    # Dataset
    parser.add_argument('--limit_complexes', type=int, default=0, help='')
    parser.add_argument('--all_atoms', action='store_true', default=False, help='')
    parser.add_argument('--multiplicity', type=int, default=1, help='')
    parser.add_argument('--chain_cutoff', type=float, default=10, help='')
    parser.add_argument('--receptor_radius', type=float, default=15, help='')
    parser.add_argument('--c_alpha_max_neighbors', type=int, default=24, help='')
    parser.add_argument('--atom_radius', type=float, default=5, help='')
    parser.add_argument('--atom_max_neighbors', type=int, default=8, help='')
    parser.add_argument('--matching_popsize', type=int, default=20, help='')
    parser.add_argument('--matching_maxiter', type=int, default=20, help='')
    parser.add_argument('--max_lig_size', type=int, default=None, help='Maximum number of heavy atoms')
    parser.add_argument('--remove_hs', action='store_true', default=False, help='remove Hs')
    parser.add_argument('--num_conformers', type=int, default=1, help='')
    parser.add_argument('--esm_embeddings_path', type=str, default=None,help='If this is set then the LM embeddings at that path will be used for the receptor features')
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
    parser.add_argument('--use_second_order_repr', action='store_true', default=False, help='Whether to use only up to first order representations or also second')
    parser.add_argument('--cross_max_distance', type=float, default=80, help='')
    parser.add_argument('--dynamic_max_cross', action='store_true', default=False, help='')
    parser.add_argument('--dropout', type=float, default=0.1, help='MLP dropout')
    parser.add_argument('--embedding_type', type=str, default="sinusoidal", help='')
    parser.add_argument('--sigma_embed_dim', type=int, default=32, help='')
    parser.add_argument('--embedding_scale', type=int, default=10000, help='')
    parser.add_argument('--confidence_no_batchnorm', action='store_true', default=False, help='')
    parser.add_argument('--confidence_dropout', type=float, default=0.0, help='MLP dropout in confidence readout')
    
    parser.add_argument('--num_workers', type=int, default=1, help='Number of workers')
    parser.add_argument('--running_mode', type=str, default='train', help='')
    parser.add_argument('--water_ratio', type=int, default=15, help='')
    parser.add_argument('--resample_steps', type=int, default=1, help='')
    
    args = parser.parse_args()
    return args

def parse_inference_args():
    # General arguments
    parser = ArgumentParser()
    parser.add_argument('--config', type=FileType(mode='r'), default=None)
    parser.add_argument('--original_model_dir', type=str, default='workdir',
                        help='Path to folder with trained model and hyperparameters')
    parser.add_argument('--confidence_dir', type=str, default='workdir',
                        help='Path to folder with trained confidence model and hyperparameters')
    parser.add_argument('--restart_dir', type=str, default=None, help='')
    parser.add_argument('--use_original_model_cache', action='store_true', default=False,
                        help='If this is true, the same dataset as in the original model will be used. Otherwise, the dataset parameters are used.')
    parser.add_argument('--data_dir', type=str, default='data/waterbind/',
                        help='Folder containing original structures')
    parser.add_argument('--ckpt', type=str, default='best_model.pt', help='Checkpoint to use inside the folder')
    parser.add_argument('--model_save_frequency', type=int, default=0,
                        help='Frequency with which to save the last model. If 0, then only the early stopping criterion best model is saved and overwritten.')
    parser.add_argument('--best_model_save_frequency', type=int, default=0,
                        help='Frequency with which to save the best model. If 0, then only the early stopping criterion best model is saved and overwritten.')
    parser.add_argument('--run_name', type=str, default='inference', help='')
    parser.add_argument('--project', type=str, default='superwater_evalution', help='')
    parser.add_argument('--split_train', type=str, default='data/splits/train_res15.txt', help='Path of file defining the split')
    parser.add_argument('--split_val', type=str, default='data/splits/val_res15.txt', help='Path of file defining the split')
    parser.add_argument('--split_test', type=str, default='data/splits/test_res15', help='Path of file defining the split')
    
    # Inference parameters for creating the positions and mads that the confidence predictor will be trained on.
    parser.add_argument('--cache_path', type=str, default='data/cacheNew',
                        help='Folder from where to load/restore cached dataset')
    parser.add_argument('--cache_ids_to_combine', nargs='+', type=str, default='1',
                        help='MAD value below which a prediction is considered a postitive. This can also be multiple cutoffs.')
    parser.add_argument('--cache_creation_id', type=int, default=1,
                        help='number of times that inference is run on the full dataset before concatenating it and coming up with the full confidence dataset')
    parser.add_argument('--wandb', action='store_true', default=False, help='')
    parser.add_argument('--inference_steps', type=int, default=20, help='Number of denoising steps')
    parser.add_argument('--samples_per_complex', type=int, default=1, help='')
    parser.add_argument('--balance', action='store_true', default=False,
                        help='If this is true than we do not force the samples seen during training to be the same amount of negatives as positives')
    parser.add_argument('--mad_prediction', action='store_true', default=False, help='')
    parser.add_argument('--mad_classification_cutoff', type=float, default=2,
                        help='MAD value below which a prediction is considered a postitive. This can also be multiple cutoffs.')
    
    parser.add_argument('--log_dir', type=str, default='workdir', help='')
    parser.add_argument('--main_metric', type=str, default='accuracy',
                        help='Metric to track for early stopping. Mostly [loss, accuracy, ROC AUC]')
    parser.add_argument('--main_metric_goal', type=str, default='max', help='Can be [min, max]')
    parser.add_argument('--transfer_weights', action='store_true', default=False, help='')
    parser.add_argument('--batch_size', type=int, default=1, help='')
    parser.add_argument('--batch_size_preprocessing', type=int, default=1, help='Number of workers')
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

    # Inference
    parser.add_argument('--cap', type=float, default=0.1, help='confidence model prob threshold')
    parser.add_argument('--save_pos', action='store_true', default=False, help='')
    parser.add_argument('--cluster_eps', type=float, default=1, help='')
    parser.add_argument('--cluster_min_samples', type=int, default=1, help='')
    parser.add_argument('--running_mode', type=str, default="test")
    parser.add_argument('--water_ratio', type=int, default=15, help='')
    parser.add_argument('--resample_steps', type=int, default=1, help='')
    parser.add_argument('--use_sigmoid', action='store_true', default=False, help='')
        
    args = parser.parse_args()
    return args