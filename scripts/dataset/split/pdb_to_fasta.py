import os
import subprocess
from tqdm import tqdm  # Import tqdm for the progress bar functionality
from pathlib import Path
DATA_DIR = Path('/home/qkrgangeun/LigMet/data/biolip_backup/merged')
# Load the list of PDB files
pdb_list = [f'{pdbid.strip()}.pdb' for pdbid in open('/home/qkrgangeun/LigMet/data/biolip_backup/pdb/test_pdb_noerror.txt')]
# pdb_list = [file for file in os.listdir(DATA_DIR)if file.endswith('.pdb')]
# pdb_list = ['8umc.pdb']
print(pdb_list)
# Define directories
fasta_dir = Path('/home/qkrgangeun/LigMet/data/biolip_backup/fasta/testset')
pdb_dir = Path('/home/qkrgangeun/LigMet/data/biolip_backup/merged')

# Ensure the output directory exists
os.makedirs(fasta_dir, exist_ok=True)

# Process each PDB file with a progress bar
for pdb in tqdm(pdb_list, desc="Converting PDB to FASTA"):
    prefix = os.path.splitext(pdb)[0]  # Correctly extract the file prefix
    input_pdb_path = os.path.join(pdb_dir, f'{pdb}')
    output_fasta_path = os.path.join(fasta_dir, f'{prefix}.fasta')  # Set the correct file extension

    # Convert PDB to FASTA using subprocess for better handling
    try:
        # It's generally better to separate commands and arguments in subprocess
        cmd = f"pdb_seq.py {input_pdb_path} > {output_fasta_path}"
        subprocess.run(cmd, shell=True, check=True, text=True)
        print(f'Success: {input_pdb_path} is converted to {output_fasta_path}')
    except subprocess.CalledProcessError:
        print(f'Error: Failed to convert {input_pdb_path}')
