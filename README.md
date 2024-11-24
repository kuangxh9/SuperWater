# SuperWater

SuperWater is a generative model designed to predict water molecule distributions on protein surfaces using score-based diffusion models and equivariant neural networks. The model and methodology are described in our [preprint on bioRxiv](https://www.biorxiv.org/content/10.1101/2024.11.18.624208v1).

For any questions, feel free to open an issue or contact us at: xiaohan.kuang@vanderbilt.edu, zhaoqian.su@vanderbilt.edu

## Overview

### Diffusion Process
<img src="./images/model_arch/diffusion_process.png" height="300"/>

### Model Architecture
<img src="./images/model_arch/superwater_model_arch.png" height="300"/>

## System Requirements

- **GPU**: Minimum 40GB memory (e.g., NVIDIA A100)
  - Required for memory-intensive reverse diffusion steps, particularly for large proteins (>600 residues)
  - Memory optimization for large proteins will be addressed in future updates
- **Training Hardware**: 4x NVIDIA A100 80GB GPUs (if retraining)

## Installation

1. Install required Python packages:
    ```bash
    pip install -r requirements.txt
    pip install "fair-esm[esmfold]"
    ```

2. Set up ESM:
    
   Clone the [ESM GitHub repository](https://github.com/facebookresearch/esm) and save it under `esm/` in your project directory.

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

    **Note:** Dummy water molecule files are placeholders required for structure preloading. Improvements for handling these files are planned for future updates.

2. Create a test split file:
    ```
    data/splits/test.txt             # List of PDB IDs, one per line
    ```
    Refer to `test_res15.txt` for an example.


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

```bash
python -m validation_recall_precision \
--original_model_dir workdir/all_atoms_score_model_res15_17092 \
--confidence_dir workdir/confidence_model_17092_sigmoid_rr15 \
--data_dir data/test_dataset \
--ckpt best_model.pt \
--all_atoms \
--run_name evaluation_all_atoms \
--cache_path data/cache_confidence \
--split_test data/splits/test.txt \
--inference_steps 20 \
--samples_per_complex 1 \
--batch_size 1 \
--batch_size_preprocessing 1 \
--esm_embeddings_path data/test_dataset_embeddings_output \
--cache_creation_id 1 \
--cache_ids_to_combine 1 \
--prob_thresh 0.05 \
--running_mode test \
--rmsd_prediction \
--save_pos
```

**Key Parameters**:
- `--data_dir`: Path to your test dataset folder (e.g., `data/test_dataset`)
- `--split_test`: Path to the test PDB IDs file (e.g., `data/splits/test.txt`)
- `--prob_thresh`: Probability cutoff for water molecule sampling (higher values increase precision but reduce coverage)
- `--save_pos`: Saves sampled water molecule positions as `.pdb` files

If you encounter out-of-memory issues, consider modifying `confidence/dataset.py` by reducing the value of `water_ratio = 15` at line 203 to a smaller value. Note that this may reduce prediction accuracy.

**Output**:
Predicted water molecule positions will be saved as `.pdb` files in:
```
inference_out/inferenced_pos_cap<prob_thresh>/
```

## Inference Animation

The animation below illustrates how randomly distributed water molecules in 3D space align to their predicted positions on the protein surface during the reverse diffusion process.

![Inference Animation](./images/inference_out/4YL4.gif)

## Coming Soon (After my final exams)
- Detailed retraining instructions
- Memory optimization for large proteins
- Improved handling of dummy water files during preprocessing
- Script for evaluation metrics and visualization