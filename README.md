# SuperWater

SuperWater is a generative model designed to predict water molecule distributions on protein surfaces using score-based diffusion models and equivariant neural networks. The model and methodology are described in our [preprint on bioRxiv](https://www.biorxiv.org/content/10.1101/2024.11.18.624208v1).

For any questions, feel free to open an issue or contact us at: xiaohan.kuang@vanderbilt.edu, zhaoqian.su@vanderbilt.edu

## Overview

<!-- ### Diffusion Process
<img src="./images/model_arch/diffusion_process.png" height="300"/> -->

### Model Architecture
<img src="./images/model_arch/superwater_model_arch.png" height="300"/>

## Data Availability
The dataset used in this project can be found at [Zenodo](https://doi.org/10.5281/zenodo.14166655).  
Download the `waterbind.zip` file, which contains 17,092 protein PDB IDs and their corresponding water molecule files.

## Environment Setup
1. Create the Conda environment by running:
    ```bash
    conda env create -f environment.yml
    ```

    Activate the environment:
    ```bash
    conda activate superwater
    ```

2. Set up ESM:
    
   Clone the [ESM GitHub repository](https://github.com/facebookresearch/esm) and save it under `esm/` in the project directory.

## Dataset Preparation
1. Create your test dataset structure:
    ```
    data/
    └── test_dataset/
        └── 5SRF/                               # Create folder for each PDB ID
            ├── 5SRF_protein_processed.pdb      # Naming pattern: <PDB_ID>_protein_processed.pdb
            ├── 5SRF_water.mol2                 # Dummy file with random water coordinates
            └── 5SRF_water.pdb                  # Dummy file with random water coordinates
    ```

    **Note:** (For Inference Only) Dummy water molecule files are placeholders required for structure preloading. Improvements for handling these files are planned for future updates.

    `<pdb_id>_water.pdb`
    ```
    HETATM    1  O   HOH A   1      0.000   0.000   0.000  1.00 0.00           O  
    TER       2      HOH A   1                                                    
    END  
    ```

    `<pdb_id>_water.mol2`
    ```
    @<TRIPOS>MOLECULE  
    water  
    1 0 0 0 0  
    SMALL  
    GASTEIGER  

    @<TRIPOS>ATOM  
        1  O         0.0000    0.0000    0.0000  O.3   1    HOH1       0.0000  

    @<TRIPOS>BOND  
    ```

2. Create a test split file:
    ```
    data/splits/test.txt             # List of PDB IDs, one per line
    ```
    Refer to `test_res15.txt` for an example.

## Retraining Process
<details>
<summary><strong>Click to expand retraining details</strong></summary>

### Step 1: Generate ESM Embeddings

1. **Prepare FASTA files**:
    ```bash
    python datasets/esm_embedding_preparation_water.py \
    --data_dir data/waterbind \
    --out_file data/prepared_for_esm_dataset_waterbind.fasta
    ```
2. **Generate embeddings**:
    ```bash
    cd data

    python ../esm/scripts/extract.py esm2_t33_650M_UR50D prepared_for_esm_dataset_waterbind.fasta \
    dataset_waterbind_embeddings_output --repr_layers 33 --include per_tok --truncation_seq_length 4096
    
    cd ..
    ```

### Step 2: Train the Score Model
```bash
python -m train \
--run_name all_atoms_score_model_res15_17092_retrain \
--test_sigma_intervals \
--esm_embeddings_path data/dataset_waterbind_embeddings_output \
--data_dir data/waterbind \
--split_train data/splits/train_res15.txt \
--split_val data/splits/val_res15.txt \
--split_test data/splits/test_res15.txt \
--log_dir workdir \
--lr 1e-3 --tr_sigma_min 0.1 --tr_sigma_max 30 \
--batch_size 8 \
--ns 24 --nv 6 \
--num_conv_layers 3 \
--dynamic_max_cross \
--scheduler plateau --scale_by_sigma \
--dropout 0.1 --all_atoms \
--c_alpha_max_neighbors 24 --remove_hs \
--receptor_radius 15 \
--num_dataloader_workers 10 \
--num_workers 10 \
--wandb \
--cudnn_benchmark \
--use_ema --distance_embed_dim 64 \
--cross_distance_embed_dim 64 \
--sigma_embed_dim 64 \
--scheduler_patience 30 \
--n_epochs 300
```

### Step 3: Train the Confidence Model
```bash
python -m confidence.confidence_train \
--original_model_dir workdir/all_atoms_score_model_res15_17092_retrain \
--data_dir data/waterbind \
--all_atoms \
--run_name confidence_model_retrain \
--split_train data/splits/train_res15.txt \
--split_val data/splits/val_res15.txt \
--split_test data/splits/test_res15.txt \
--inference_steps 20 \
--batch_size 8 \
--n_epochs 50 \
--wandb \
--lr 1e-3 \
--ns 24 \
--nv 6 \
--num_conv_layers 3 \
--dynamic_max_cross \
--scale_by_sigma \
--dropout 0.1 \
--remove_hs \
--esm_embeddings_path data/dataset_waterbind_embeddings_output \
--cache_creation_id 1 \
--cache_ids_to_combine 1 \
--running_mode train \
--mad_prediction
```
**Note**: If GPU memory is limited, consider adjusting:
```
--water_ratio 10
```
</details>

## Running Inference

### Step 1: Generate ESM Embeddings

1. **Prepare FASTA files**:
    ```bash
    python datasets/esm_embedding_preparation_water.py \
    --data_dir data/test_dataset \
    --out_file data/prepared_for_esm_test_dataset.fasta
    ```

2. **Generate embeddings**:
    ```bash
    cd data

    python ../esm/scripts/extract.py esm2_t33_650M_UR50D prepared_for_esm_test_dataset.fasta \
    test_dataset_embeddings_output --repr_layers 33 --include per_tok --truncation_seq_length 4096

    cd ..
    ```

### Step 2: Run Model Inference

Run the following command to perform inference:

```
python -m inference_water_pos \
--original_model_dir workdir/all_atoms_score_model_res15_17092 \
--confidence_dir workdir/confidence_model_17092_sigmoid_rr15 \
--data_dir data/test_dataset \
--ckpt best_model.pt \
--all_atoms \
--cache_path data/cache_confidence \
--split_test data/splits/test.txt \
--inference_steps 20 \
--esm_embeddings_path data/test_dataset_embeddings_output \
--cap 0.1 \
--running_mode test \
--mad_prediction \
--save_pos
```

**Key Parameters**:
- `--data_dir`: Path to your test dataset folder (e.g., `data/test_dataset`)
- `--split_test`: Path to the test PDB IDs file (e.g., `data/splits/test.txt`)
- `--cap`: Probability cutoff for water molecule sampling 
    - Higher values increase precision but reduce coverage 
    - Acceptable range: [0.02, 0.5]
- `--save_pos`: Saves sampled water molecule positions as `.pdb` files

**Save Diffusion Process Animation**

To save intermediate steps of the reverse diffusion process as .pdb files, add the following flag:

```
--save_visualization
``` 

- Output Directory: `inference_out/diff_process`
- Visualization: Load the `.pdb` files in PyMOL or similar tools to animate and visualize the diffusion process frame by frame.


**Note**:
- Adjust the `--water_ratio` to 10 or lower when running inference to reduce memory usage.
- When changing the dataset or adjusting parameters for resampling, ensure to either
    - Change the cache path using `--cache_path`
    - Delete the existing cache to avoid conflicts

### Output:

Predicted water molecule positions will be saved as `.pdb` files in:
```
inference_out/inferenced_pos_cap<#>/
```

## Inference Animation

The animation below illustrates how randomly distributed water molecules in 3D space align to their predicted positions on the protein surface during the reverse diffusion process.

![Inference Animation](./images/inference_out/4YL4.gif)

