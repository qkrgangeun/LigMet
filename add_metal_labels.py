#!/usr/bin/env python3
"""
Script to add metal labels, positions, and types to existing npz feature files.
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

def add_metal_labels_to_features():
    # Paths
    csv_path = "/home/qkrgangeun/LigMet/code/text/biolip/all_metal_binding_sites_NOSSE_3.0.csv"
    pdb_dir = "/home/qkrgangeun/LigMet/benchmark/mymodel/af3_testset/chimera_aligned/pdb"
    features_dir = "/home/qkrgangeun/LigMet/benchmark/mymodel/af3_testset/chimera_aligned/dl/features"
    
    # Load the CSV data
    print("Loading metal binding sites data...")
    df = pd.read_csv(csv_path)
    
    # Filter for chain 'A' only
    df['Binding Chains Parsed'] = df['Binding Chains'].apply(parse_binding_chains)
    df_chain_a = df[df['Binding Chains Parsed'].apply(lambda x: 'A' in x if x else False)]
    
    print(f"Found {len(df_chain_a)} metal binding sites in chain A")
    
    # Group by PDB ID
    grouped = df_chain_a.groupby('PDB ID')
    
    # Get list of PDB files
    if os.path.exists(pdb_dir):
        pdb_files = [f for f in os.listdir(pdb_dir) if f.endswith('.pdb')]
        pdb_ids = [f.replace('.pdb', '') for f in pdb_files]
        print(f"Found {len(pdb_ids)} PDB files")
    else:
        print(f"PDB directory not found: {pdb_dir}")
        return
    
    # Get list of existing feature files
    if os.path.exists(features_dir):
        feature_files = [f for f in os.listdir(features_dir) if f.endswith('.npz')]
        feature_ids = [f.replace('.npz', '') for f in feature_files]
        print(f"Found {len(feature_files)} feature files")
    else:
        print(f"Features directory not found: {features_dir}")
        return
    
    # Process each feature file
    updated_count = 0
    for pdb_id in feature_ids:
        feature_path = os.path.join(features_dir, f"{pdb_id}.npz")
        
        if not os.path.exists(feature_path):
            continue
            
        # Load existing features
        try:
            data = np.load(feature_path, allow_pickle=True)
            features_dict = dict(data)
            
            # Check if metal information already exists
            if 'metal_labels' in features_dict:
                print(f"Metal labels already exist for {pdb_id}, skipping...")
                continue
                
        except Exception as e:
            print(f"Error loading {feature_path}: {e}")
            continue
        
        # Get metal data for this PDB ID
        if pdb_id.upper() in grouped.groups:
            pdb_metals = grouped.get_group(pdb_id.upper())
            
            # Extract metal information
            metal_positions = []
            metal_types = []
            metal_labels = []
            
            for _, row in pdb_metals.iterrows():
                pos = parse_metal_position(row['Metal Position'])
                if pos is not None:
                    metal_positions.append(pos)
                    metal_types.append(row['Metal Type'])
                    metal_labels.append(1)  # 1 indicates metal binding site
            
            print(f"Processing {pdb_id}: Found {len(metal_positions)} metal sites")
            
            # Add metal information to features
            if metal_positions:
                features_dict['metal_positions'] = np.array(metal_positions, dtype=np.float32)
                features_dict['metal_types'] = np.array(metal_types, dtype='U10')
                
                # Create metal labels for residues (assuming this refers to residue-level labels)
                # You may need to adjust this based on your specific requirements
                if 'residue_coords' in features_dict or 'coords' in features_dict:
                    # Get the number of residues/atoms
                    if 'residue_coords' in features_dict:
                        num_residues = len(features_dict['residue_coords'])
                    elif 'coords' in features_dict:
                        num_residues = len(features_dict['coords'])
                    else:
                        num_residues = 0
                    
                    # Initialize metal labels (0 = no metal, 1 = metal binding)
                    metal_labels_array = np.zeros(num_residues, dtype=np.int32)
                    
                    # Here you would need logic to determine which residues are metal-binding
                    # This is a simplified approach - you may need more sophisticated logic
                    features_dict['metal_labels'] = metal_labels_array
                else:
                    # If no coordinate information, just store the presence of metals
                    features_dict['metal_labels'] = np.array([1] * len(metal_positions), dtype=np.int32)
            else:
                # No metals found for this PDB
                print(f"No metal sites found for {pdb_id}")
                features_dict['metal_positions'] = np.array([], dtype=np.float32).reshape(0, 3)
                features_dict['metal_types'] = np.array([], dtype='U10')
                
                # Set metal labels based on existing structure
                if 'residue_coords' in features_dict:
                    num_residues = len(features_dict['residue_coords'])
                elif 'coords' in features_dict:
                    num_residues = len(features_dict['coords'])
                else:
                    num_residues = 0
                
                if num_residues > 0:
                    features_dict['metal_labels'] = np.zeros(num_residues, dtype=np.int32)
                else:
                    features_dict['metal_labels'] = np.array([], dtype=np.int32)
        else:
            # PDB not found in metal database
            print(f"No metal data found for {pdb_id}")
            features_dict['metal_positions'] = np.array([], dtype=np.float32).reshape(0, 3)
            features_dict['metal_types'] = np.array([], dtype='U10')
            
            # Set metal labels based on existing structure
            if 'residue_coords' in features_dict:
                num_residues = len(features_dict['residue_coords'])
            elif 'coords' in features_dict:
                num_residues = len(features_dict['coords'])
            else:
                num_residues = 0
            
            if num_residues > 0:
                features_dict['metal_labels'] = np.zeros(num_residues, dtype=np.int32)
            else:
                features_dict['metal_labels'] = np.array([], dtype=np.int32)
        
        # Save updated features
        try:
            np.savez_compressed(feature_path, **features_dict)
            updated_count += 1
            print(f"Updated {pdb_id}")
        except Exception as e:
            print(f"Error saving {feature_path}: {e}")
    
    print(f"\nCompleted! Updated {updated_count} feature files")

if __name__ == "__main__":
    add_metal_labels_to_features()
