#!/usr/bin/env python3
"""
Script to add metal labels, positions, and types to existing npz feature files.
Enhanced version with better error handling and directory checking.
"""

import os
import pandas as pd
import numpy as np
import ast

def parse_metal_position(pos_str):
    """Parse the metal position string to get coordinates."""
    try:
        return ast.literal_eval(pos_str)
    except Exception:
        return None

def parse_binding_chains(chains_str):
    """Parse the binding chains string."""
    try:
        return ast.literal_eval(chains_str)
    except Exception:
        return []

def main():
    # Paths
    csv_path = "/home/qkrgangeun/LigMet/code/text/biolip/all_metal_binding_sites_NOSSE_3.0.csv"
    pdb_dir = "/home/qkrgangeun/LigMet/benchmark/mymodel/af3_testset/chimera_aligned/pdb"
    features_dir = "/home/qkrgangeun/LigMet/benchmark/mymodel/af3_testset/chimera_aligned/dl/features"
    
    print("=== Metal Label Addition Script ===")
    
    # Check if directories exist
    print(f"Checking paths...")
    print(f"CSV file exists: {os.path.exists(csv_path)}")
    print(f"PDB directory exists: {os.path.exists(pdb_dir)}")
    print(f"Features directory exists: {os.path.exists(features_dir)}")
    
    if not os.path.exists(csv_path):
        print(f"ERROR: CSV file not found: {csv_path}")
        return
    
    if not os.path.exists(features_dir):
        print(f"ERROR: Features directory not found: {features_dir}")
        return
    
    # Load the CSV data
    print("\nLoading metal binding sites data...")
    try:
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} rows from CSV")
        print(f"Columns: {list(df.columns)}")
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return
    
    # Show sample of data
    print("\nSample data:")
    print(df.head(3))
    
    # Filter for chain 'A' only
    print("\nFiltering for chain 'A'...")
    df['Binding Chains Parsed'] = df['Binding Chains'].apply(parse_binding_chains)
    df_chain_a = df[df['Binding Chains Parsed'].apply(lambda x: 'A' in x if x else False)]
    
    print(f"Found {len(df_chain_a)} metal binding sites in chain A")
    
    # Group by PDB ID
    grouped = df_chain_a.groupby('PDB ID')
    print(f"Unique PDB IDs with metal sites in chain A: {len(grouped.groups)}")
    
    # Get list of existing feature files
    print(f"\nChecking feature files in {features_dir}")
    try:
        feature_files = [f for f in os.listdir(features_dir) if f.endswith('.npz')]
        feature_ids = [f.replace('.npz', '') for f in feature_files]
        print(f"Found {len(feature_files)} feature files")
        
        if feature_files:
            print(f"Sample feature files: {feature_files[:5]}")
    except Exception as e:
        print(f"Error listing feature files: {e}")
        return
    
    if not feature_files:
        print("No feature files found!")
        return
    
    # Examine structure of first feature file
    print(f"\nExamining structure of first feature file...")
    first_file = os.path.join(features_dir, feature_files[0])
    try:
        data = np.load(first_file, allow_pickle=True)
        print(f"Keys in {feature_files[0]}: {list(data.keys())}")
        for key in data.keys():
            arr = data[key]
            print(f"  {key}: shape={arr.shape}, dtype={arr.dtype}")
    except Exception as e:
        print(f"Error examining first file: {e}")
        return
    
    # Process each feature file
    print(f"\nProcessing feature files...")
    updated_count = 0
    
    for i, pdb_id in enumerate(feature_ids[:5]):  # Process first 5 files as test
        print(f"\nProcessing {i+1}/{min(5, len(feature_ids))}: {pdb_id}")
        feature_path = os.path.join(features_dir, f"{pdb_id}.npz")
        
        # Load existing features
        try:
            data = np.load(feature_path, allow_pickle=True)
            features_dict = dict(data)
            
            # Check if metal information already exists
            if 'metal_labels' in features_dict:
                print(f"  Metal labels already exist, skipping...")
                continue
                
        except Exception as e:
            print(f"  Error loading {feature_path}: {e}")
            continue
        
        # Get metal data for this PDB ID
        pdb_id_upper = pdb_id.upper()
        if pdb_id_upper in grouped.groups:
            pdb_metals = grouped.get_group(pdb_id_upper)
            
            # Extract metal information
            metal_positions = []
            metal_types = []
            
            for _, row in pdb_metals.iterrows():
                pos = parse_metal_position(row['Metal Position'])
                if pos is not None:
                    metal_positions.append(pos)
                    metal_types.append(row['Metal Type'])
            
            print(f"  Found {len(metal_positions)} metal sites")
            
            # Add metal information to features
            if metal_positions:
                features_dict['metal_positions'] = np.array(metal_positions, dtype=np.float32)
                features_dict['metal_types'] = np.array(metal_types, dtype='U10')
                print(f"  Metal types: {metal_types}")
                print(f"  Metal positions: {metal_positions}")
            else:
                features_dict['metal_positions'] = np.array([], dtype=np.float32).reshape(0, 3)
                features_dict['metal_types'] = np.array([], dtype='U10')
                print(f"  No valid metal positions found")
        else:
            # PDB not found in metal database
            print(f"  No metal data found for {pdb_id}")
            features_dict['metal_positions'] = np.array([], dtype=np.float32).reshape(0, 3)
            features_dict['metal_types'] = np.array([], dtype='U10')
        
        # Determine number of residues for labels
        num_residues = 0
        if 'residue_coords' in features_dict:
            num_residues = len(features_dict['residue_coords'])
            print(f"  Found {num_residues} residues (from residue_coords)")
        elif 'coords' in features_dict:
            num_residues = len(features_dict['coords'])
            print(f"  Found {num_residues} residues (from coords)")
        elif 'atom_coords' in features_dict:
            num_residues = len(features_dict['atom_coords'])
            print(f"  Found {num_residues} atoms (from atom_coords)")
        
        # Initialize metal labels (0 = no metal binding, 1 = metal binding)
        if num_residues > 0:
            features_dict['metal_labels'] = np.zeros(num_residues, dtype=np.int32)
            print(f"  Created metal_labels array of size {num_residues}")
        else:
            features_dict['metal_labels'] = np.array([], dtype=np.int32)
            print(f"  Created empty metal_labels array")
        
        # Save updated features
        try:
            np.savez_compressed(feature_path, **features_dict)
            updated_count += 1
            print(f"  Successfully updated {pdb_id}")
        except Exception as e:
            print(f"  Error saving {feature_path}: {e}")
    
    print(f"\nTest completed! Updated {updated_count} feature files (first 5 only)")
    print("If this looks correct, remove the [:5] limit to process all files")

if __name__ == "__main__":
    main()
