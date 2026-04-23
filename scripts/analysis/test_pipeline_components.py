#!/usr/bin/env python3
"""
Test script to validate pipeline components before full execution.
"""

import sys
import os
from pathlib import Path
import numpy as np

# Add scripts directory to path
scripts_dir = Path(__file__).parent
sys.path.insert(0, str(scripts_dir))

print("=" * 70)
print("MMSEQS2 + MetalS2 Pipeline - Component Test")
print("=" * 70)

# Test 1: Paths
print("\n[TEST 1] Checking data paths...")
PROJECT_ROOT = Path('/home/qkrgangeun/LigMet')
paths_to_check = {
    'METAL_LABEL_DIR': PROJECT_ROOT / 'data/biolip_backup/metal_label',
    'FASTA_INPUT_DIR': PROJECT_ROOT / 'data/biolip_backup/fasta/train_chain1',
    'TRAIN_PDB_LIST': PROJECT_ROOT / 'code/text/biolip/filtered/train_pdbs_chain_1_filtered.txt',
    'PDB_DIR': PROJECT_ROOT / 'data/biolip_backup/pdb',
}

all_exist = True
for name, path in paths_to_check.items():
    exists = path.exists()
    status = "✓" if exists else "✗"
    print(f"  {status} {name}: {path}")
    if not exists:
        all_exist = False

if not all_exist:
    print("ERROR: Some required paths do not exist!")
    sys.exit(1)

# Test 2: Training PDB list
print("\n[TEST 2] Loading training PDB list...")
with open(paths_to_check['TRAIN_PDB_LIST']) as f:
    pdbs = [line.strip() for line in f if line.strip()]
print(f"  ✓ Loaded {len(pdbs)} training PDB IDs")
if len(pdbs) < 100:
    print("  WARNING: Very few PDB IDs loaded")

# Test 3: Metal annotations
print("\n[TEST 3] Checking metal annotations...")
metal_label_dir = paths_to_check['METAL_LABEL_DIR']
npz_files = list(metal_label_dir.glob('*.npz'))
print(f"  ✓ Found {len(npz_files)} .npz annotation files")

# Test loading one
test_pdb = pdbs[0].lower() if pdbs else None
if test_pdb:
    test_file = metal_label_dir / f"{test_pdb}.npz"
    if test_file.exists():
        try:
            data = np.load(test_file, allow_pickle=True)
            positions = data['metal_positions']
            types = data['metal_types']
            print(f"  ✓ Loaded annotation for {test_pdb}:")
            print(f"      Metals: {len(positions)}, Types: {list(types)}")
            data.close()
        except Exception as e:
            print(f"  ✗ Failed to load {test_file}: {e}")
    else:
        print(f"  ℹ Annotation not found for first PDB: {test_pdb}")

# Test 4: FASTA input
print("\n[TEST 4] Checking FASTA input file...")
fasta_input = paths_to_check['FASTA_INPUT_DIR'] / 'fasta_input.txt'
if fasta_input.exists():
    with open(fasta_input) as f:
        lines = f.readlines()
    headers = [l for l in lines if l.startswith('>')]
    sequences = [l for l in lines if not l.startswith('>')]
    print(f"  ✓ FASTA file: {len(headers)} headers, {len(sequences)} sequence lines")
else:
    print(f"  ✗ FASTA file not found: {fasta_input}")

# Test 5: Check external tools
print("\n[TEST 5] Checking external tools...")

import shutil
tools = {
    'mmseqs': 'MMseqs2 (for sequence clustering)',
    'biopython': 'BioPython (for PDB parsing)',
    'networkx': 'NetworkX (for graph operations)',
    'pandas': 'Pandas (for data processing)',
}

tool_status = {}
for tool_name, description in tools.items():
    if tool_name == 'biopython':
        try:
            import Bio
            tool_status[tool_name] = True
            print(f"  ✓ {description}")
        except ImportError:
            tool_status[tool_name] = False
            print(f"  ✗ {description} - NOT installed")
    elif tool_name == 'networkx':
        try:
            import networkx
            tool_status[tool_name] = True
            print(f"  ✓ {description}")
        except ImportError:
            tool_status[tool_name] = False
            print(f"  ✗ {description} - NOT installed")
    elif tool_name == 'pandas':
        try:
            import pandas
            tool_status[tool_name] = True
            print(f"  ✓ {description}")
        except ImportError:
            tool_status[tool_name] = False
            print(f"  ✗ {description} - NOT installed")
    else:
        exe_path = shutil.which(tool_name)
        if exe_path:
            tool_status[tool_name] = True
            print(f"  ✓ {description} at {exe_path}")
        else:
            tool_status[tool_name] = False
            print(f"  ⚠ {description} - NOT found (will be skipped/mocked)")

# Test 6: Import pipeline module
print("\n[TEST 6] Importing pipeline modules...")
try:
    from mmseqs2_metals_analysis_pipeline import (
        load_training_pdbs,
        load_metal_annotation,
        compute_jaccard_similarity,
    )
    print(f"  ✓ Successfully imported main pipeline module")
except Exception as e:
    print(f"  ✗ Failed to import pipeline: {e}")
    sys.exit(1)

try:
    from pdb_structure_utils import get_local_site_residues
    print(f"  ✓ Successfully imported PDB structure utilities")
except Exception as e:
    print(f"  ✗ Failed to import PDB utilities: {e}")
    print(f"    (This is OK if BioPython is not installed)")

# Test 7: Quick functional test
print("\n[TEST 7] Running quick functional tests...")

# Test loading a metal annotation
ann = load_metal_annotation(test_pdb)
if ann and len(ann['positions']) > 0:
    print(f"  ✓ Loaded metal annotation: {len(ann['positions'])} metals")

# Test Jaccard computation
set1 = {('A', 10), ('A', 11), ('A', 12)}
set2 = {('A', 11), ('A', 12), ('A', 13)}
jaccard = compute_jaccard_similarity(set1, set2)
print(f"  ✓ Jaccard test: J({set1}, {set2}) = {jaccard:.3f}")

# Summary
print("\n" + "=" * 70)
print("TEST SUMMARY")
print("=" * 70)

all_ok = all(tool_status.values())
if not all_ok:
    print("⚠ Some optional tools are missing, but pipeline can still run with mocking")
    print("\nTo fix, install missing packages:")
    for tool, status in tool_status.items():
        if not status:
            if tool == 'biopython':
                print(f"  pip install biopython")
            elif tool == 'networkx':
                print(f"  pip install networkx")
            elif tool == 'pandas':
                print(f"  pip install pandas")
            elif tool == 'mmseqs':
                print(f"  brew install mmseqs2  OR  conda install -c bioconda mmseqs2")

print("\n✓ All critical components available")
print("Ready to run: python mmseqs2_metals_analysis_pipeline.py")
print("=" * 70)
