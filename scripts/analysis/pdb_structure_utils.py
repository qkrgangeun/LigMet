#!/usr/bin/env python3
"""
PDB Structure Analysis Utilities

Functions for parsing PDB files and extracting metal-coordinating residues.
"""

import numpy as np
from pathlib import Path
from Bio import PDB
from Bio.PDB import PDBParser, PDBIO
import logging

logger = logging.getLogger(__name__)

PDB_DIR = Path('/home/qkrgangeun/LigMet/data/biolip_backup/pdb')


def load_pdb_structure(pdb_id, chain=None):
    """
    Load PDB structure using BioPython.
    
    Args:
        pdb_id: PDB ID (4 characters)
        chain: Optional chain ID filter
        
    Returns:
        Bio.PDB.Structure object or None if loading fails
    """
    pdb_file = PDB_DIR / f"{pdb_id.lower()}.pdb"
    
    if not pdb_file.exists():
        logger.warning(f"PDB file not found: {pdb_file}")
        return None
    
    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure(pdb_id, str(pdb_file))
        return structure
    except Exception as e:
        logger.error(f"Failed to parse {pdb_id}: {e}")
        return None


def get_atom_coordinates(structure, chain_id=None):
    """
    Get all atom coordinates from structure.
    
    Returns:
        dict: atom_name -> (x, y, z) coordinates
    """
    coords = {}
    
    for model in structure:
        for chain in model:
            if chain_id and chain.get_id() != chain_id:
                continue
                
            for residue in chain:
                for atom in residue:
                    atom_name = f"{chain.get_id()}_{residue.get_id()[1]}_{atom.get_id()}"
                    coords[atom_name] = atom.get_coord()
    
    return coords


def get_residues_within_radius(structure, metal_positions, radius, chain_id=None):
    """
    Get all residues within a certain radius of metal positions.
    
    Args:
        structure: Bio.PDB.Structure
        metal_positions: list of (x, y, z) coordinates
        radius: distance threshold in Ångströms
        chain_id: optional chain filter
        
    Returns:
        dict: metal_idx -> set of (chain_id, residue_number) tuples
    """
    local_residues = {}
    
    for model in structure:
        for chain in model:
            if chain_id and chain.get_id() != chain_id:
                continue
            
            for residue in chain:
                # Get residue center (Cα or centroid)
                try:
                    ca_atom = residue['CA']
                    res_coord = ca_atom.get_coord()
                except KeyError:
                    # Try using centroid if no CA
                    atoms = [atom.get_coord() for atom in residue if atom.element != 'H']
                    if not atoms:
                        continue
                    res_coord = np.mean(atoms, axis=0)
                
                res_key = (chain.get_id(), residue.get_id()[1])
                
                # Check distance to each metal
                for metal_idx, metal_pos in enumerate(metal_positions):
                    dist = np.linalg.norm(res_coord - np.array(metal_pos))
                    
                    if dist <= radius:
                        if metal_idx not in local_residues:
                            local_residues[metal_idx] = set()
                        local_residues[metal_idx].add(res_key)
    
    return local_residues


def get_coordinating_residues(structure, metal_positions, coord_radius=3.0, chain_id=None):
    """
    Get residues likely coordinating the metal (very close distance).
    
    Args:
        structure: Bio.PDB.Structure
        metal_positions: list of (x, y, z) coordinates  
        coord_radius: distance threshold for coordination (typically 2-3 Å)
        chain_id: optional chain filter
        
    Returns:
        dict: metal_idx -> set of (chain_id, residue_number) tuples
    """
    return get_residues_within_radius(structure, metal_positions, coord_radius, chain_id)


def get_local_site_residues(pdb_id, metal_positions, radius=5.0, chain_id=None):
    """
    Get all residues in local site around metal(s).
    Convenience wrapper combining structure loading and residue extraction.
    
    Args:
        pdb_id: PDB identifier
        metal_positions: array of shape (N_metals, 3)
        radius: Ångströms
        chain_id: optional chain filter
        
    Returns:
        dict: metal_idx -> set of (chain_id, resnum) tuples
    """
    structure = load_pdb_structure(pdb_id, chain_id)
    
    if structure is None:
        logger.warning(f"Could not load structure for {pdb_id}")
        return {}
    
    return get_residues_within_radius(structure, metal_positions, radius, chain_id)


if __name__ == '__main__':
    # Test example
    logging.basicConfig(level=logging.INFO)
    
    pdb_id = '6dq5'
    metal_positions = np.array([[-23.958, 18.803, 1.01]])  # From test data
    
    print(f"Testing with PDB {pdb_id}")
    local_residues = get_local_site_residues(pdb_id, metal_positions, radius=5.0)
    
    for metal_idx, residues in local_residues.items():
        print(f"  Metal {metal_idx}: {len(residues)} residues")
        print(f"    {residues}")
