#!/usr/bin/env python3
"""
Script to examine the structure of existing npz feature files.
"""

import os
import numpy as np

def examine_npz_structure():
    features_dir = "/home/qkrgangeun/LigMet/benchmark/mymodel/af3_testset/chimera_aligned/dl/features"
    
    if not os.path.exists(features_dir):
        print(f"Features directory not found: {features_dir}")
        return
    
    # Get list of feature files
    feature_files = [f for f in os.listdir(features_dir) if f.endswith('.npz')]
    
    if not feature_files:
        print("No npz files found")
        return
    
    # Examine the first few files
    for i, filename in enumerate(feature_files[:3]):
        filepath = os.path.join(features_dir, filename)
        print(f"\n=== Examining {filename} ===")
        
        try:
            data = np.load(filepath, allow_pickle=True)
            print(f"Keys in file: {list(data.keys())}")
            
            for key in data.keys():
                arr = data[key]
                print(f"  {key}: shape={arr.shape}, dtype={arr.dtype}")
                if arr.size < 10:  # Only print small arrays
                    print(f"    values: {arr}")
                    
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
        
        if i >= 2:  # Only examine first 3 files
            break

if __name__ == "__main__":
    examine_npz_structure()
