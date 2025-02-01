import os
import shutil
from collections import defaultdict
from tqdm import tqdm
from argparse import ArgumentParser

parser = ArgumentParser(description="Process PDB files and organize dataset.")
parser.add_argument("--raw_data", type=str, required=True, help="Name of the dataset folder containing PDB files.")
parser.add_argument("--data_root", type=str, default="data", help="Root directory where dataset folder is located.")
parser.add_argument("--output_dir", type=str, default="test_data", help="Directory to store processed test data.")
parser.add_argument("--splits_path", type=str, default="data/splits", help="Directory to save splits.")
parser.add_argument("--dummy_water_dir", type=str, default="data/dummy_water", help="Directory containing dummy water files.")
parser.add_argument("--logs_dir", type=str, default="logs", help="Directory to save logs.")
args = parser.parse_args()

data_path = os.path.join(args.data_root, args.raw_data)
test_data_path = os.path.join('data', args.output_dir)
dummy_water_path = args.dummy_water_dir
logs_path = args.logs_dir
splits_path = args.splits_path

os.makedirs(test_data_path, exist_ok=True)
os.makedirs(logs_path, exist_ok=True)
os.makedirs(splits_path, exist_ok=True)

pdb_files = [f for f in os.listdir(data_path) if f.endswith(".pdb")]

pdb_id_dict = {}
duplicate_truncate_pdb_id = []
truncate_map = defaultdict(list)
successful_pdb_ids = []

for pdb_file in tqdm(pdb_files, desc="Processing PDB files"):
    original_pdb_id = pdb_file.replace(".pdb", "")
    truncated_pdb_id = original_pdb_id[:4]
    
    if truncated_pdb_id in truncate_map:
        duplicate_truncate_pdb_id.append(original_pdb_id)
    else:
        pdb_id_dict[original_pdb_id] = truncated_pdb_id
        truncate_map[truncated_pdb_id].append(original_pdb_id)

unique_pdb_id_dict = {v[0]: k for k, v in truncate_map.items() if len(v) == 1}

for original_pdb_id, truncated_pdb_id in tqdm(unique_pdb_id_dict.items(), desc="Copying and renaming PDB files"):
    dest_folder = os.path.join(test_data_path, truncated_pdb_id)
    os.makedirs(dest_folder, exist_ok=True)
    successful_pdb_ids.append(truncated_pdb_id)

    src_file = os.path.join(data_path, f"{original_pdb_id}.pdb")
    dest_file = os.path.join(dest_folder, f"{truncated_pdb_id}_protein_processed.pdb")
    shutil.copy(src_file, dest_file)
    
    for water_ext in ["mol2", "pdb"]:
        water_src = os.path.join(dummy_water_path, f"_water.{water_ext}")
        water_dest = os.path.join(dest_folder, f"{truncated_pdb_id}_water.{water_ext}")
        shutil.copy(water_src, water_dest)

log_file_path = os.path.join(logs_path, "duplicate_truncate_pdb_id.txt")
with open(log_file_path, "w") as log_file:
    for dup_id in duplicate_truncate_pdb_id:
        log_file.write(f"{dup_id}\n")

split_file_path = os.path.join(splits_path, f"{args.output_dir}.txt")
with open(split_file_path, "w") as split_file:
    for pdb_id in successful_pdb_ids:
        split_file.write(f"{pdb_id}\n")

print("Processing completed.")
print(f"Successful saved test PDB IDs to {split_file_path}.")
print(f"Duplicate truncated IDs saved to {log_file_path}.")
