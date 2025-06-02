import numpy as np
from scipy.spatial import cKDTree, query_ball_tree
from nuri.core import geometry as ngeo

def align_pdbs(pdb, af_model, m_pos):
    """
    pdb: should be renumbered pdb against af_model using pdb_renum.py 
    """
    pdb_dir = '/home/qkrgangeun/LigMet/data/biolip/renum_pdb'
    af_model_dir = '/home/qkrgangeun/LigMet/data/biolip/af2.3/DB' 
    pdb_id = pdb.split('/')[-1].split('.pdb')[0]
    
    # find metal binding residues(number and chain) in pdb
    atom_positions = []
    atom_resnums = []
    atom_resnames = []
    atom_names = []
    atom_chains = []
    atom_elements = []

    with open(f'{pdb_dir}/{pdb_id}') as f:
        for line in f:
            if line.startswith('ATOM'):
                atom_position = np.array(float(line[30:38]), float(line[38:46]), float(line[46:54]))
                atom_resnum = line[22:26].strip()
                atom_resname = line[17:20].strip()
                atom_name = line[12:16].strip()
                atom_chain = line[21].strip() 
                atom_element = line[76:78].strip()
                
                atom_positions.append(atom_position)
                atom_resnums.append(atom_resnum)
                atom_resnames.append(atom_resname)  
                atom_names.append(atom_name)
                atom_chains.append(atom_chain)
                atom_elements.append(atom_element)
                
    af_atom_positions = []
    af_atom_resnums = []
    af_atom_resnames = []
    af_atom_names = []
    af_atom_chains = []
    af_atom_elements = []
    with open(f'{af_model_dir}/{af_model}') as f:
        for line in f:
            if line.startswith('ATOM'):
                af_atom_position = np.array(float(line[30:38]), float(line[38:46]), float(line[46:54]))
                af_atom_resnum = line[22:26].strip()
                af_atom_resname = line[17:20].strip()
                af_atom_name = line[12:16].strip()
                af_atom_chain = line[21].strip() 
                af_atom_element = line[76:78].strip()
                
                af_atom_positions.append(af_atom_position)
                af_atom_resnums.append(af_atom_resnum)
                af_atom_resnames.append(af_atom_resname)  
                af_atom_names.append(af_atom_name)
                af_atom_chains.append(af_atom_chain)
                af_atom_elements.append(af_atom_element)
                
                
    accepted_elements = {'N', 'O', 'S', 'SE'}
    selected_resnums = set()

    atom_positions = np.array(atom_positions)        # (N, 3)
    atom_resnums = np.array(atom_resnums)            # (N,)
    atom_elements = np.array(atom_elements)          # (N,)

  # (3,) vector
    dists = np.linalg.norm(atom_positions - m_pos, axis=1)  # shape: (N,)
    within_3A = dists <= 3.0

    element_mask = np.isin(atom_elements, list(accepted_elements))
    final_mask = within_3A & element_mask

    selected_resnums.update(atom_resnums[final_mask])
    
    af_resnums_mask = np.isin(af_atom_resnums, selected_resnums)
    pdb_resnums_mask = np.isin(atom_resnums, selected_resnums)
    af_bb_mask = np.isin(af_atom_names, ['N', 'CA', 'C', 'O'])
    pdb_bb_mask = np.isin(atom_names, ['N', 'CA', 'C', 'O']) 
    transform_t = ngeo.align_points(af_atom_positions[af_resnums_mask], atom_positions[pdb_resnums_mask],)
    transformed_af_positions = ngeo.transform(af_atom_positions[af_resnums_mask], transform_t)

            
