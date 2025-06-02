import numpy as np
import pandas as pd
import argparse
from ligmet.utils.rf.label import label_grids  # Ensure label_grids is imported properly
from ligmet.featurizer import Features
from pathlib import Path

def update_labels(feature_path, metal_path, file_path, output_file):
    # Load the existing CSV
    df = pd.read_csv(file_path, compression='gzip')
    
    # Extract structure data and calculate the new labels
    structure_dict = np.load(feature_path, allow_pickle=True)
    # structure = Features(**structure_dict)
    metal_dict = np.load(metal_path, allow_pickle=True)
    # Recalculate the 'label_2.0' based on the structure
    labels = label_grids(metal_dict["metal_positions"], structure_dict["grid_positions"], 2.0)
    
    # Update the 'label_2.0' column
    df['label_2.0'] = labels.astype('bool')
    
    # Save the updated dataframe
    df.to_csv(output_file, index=False, compression="gzip")
    print(output_file, 'is compressed')
    
def main():
    parser = argparse.ArgumentParser(description='Process an npz file and extract features.')
    parser.add_argument('feature_path', type=str, help='Path to the npz file, StructureWithGrid')
    parser.add_argument('--output_dir', type=str, help='Path to the output directory')
    args = parser.parse_args()
    input_path = Path(args.feature_path)
    file_dir, file_name = input_path.parent, input_path.name
    pdb_id = file_name.split('.npz')[0]
    
    # Define output directory
    output_dir = Path(args.output_dir) if args.output_dir else file_dir.parent / 'rf_features'
    output_path = output_dir / f'{pdb_id}.csv.gz'
    output_dir.mkdir(parents=True, exist_ok=True)
    metal_path = f'/home/qkrgangeun/LigMet/data/biolip/metal_label/{pdb_id}.npz'

    print(f"Updating labels in {output_path}")
    update_labels(input_path, metal_path, output_path, output_path)


if __name__ == "__main__":
    main()
