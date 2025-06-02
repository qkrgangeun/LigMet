import numpy as np
import scipy
from ligmet.utils.constants import VAN_DER_WAALS_RADII
from ligmet.utils.constants import DIST_PROBE_ALPHA, DIST_PROBE_BETA, ANGLE_PAB
from ligmet.utils.constants import standard_residues
# from metalpred.data.biometall import DIST_PROBE_OXYGEN, ANGLE_POC

#lig
def near_lig(atom_coords, atom_elems, residues, grids, threshold):
    kd_grid = scipy.spatial.cKDTree(grids)
    kd_xyzs = scipy.spatial.cKDTree(atom_coords)
    neighs = kd_grid.query_ball_tree(kd_xyzs, threshold)
    n_lig_NOS = []
    n_lig_nion = []
    n_lig_etc = []
    nos_mask = np.isin(atom_elems, ['N','O','S','SE'])
    neg_ion_mask = np.isin(atom_elems,['BR','CL','I'])&np.isin(residues,['BR','CL','I'])
    etc_mask=np.isin(atom_elems,['P','TE','BR','CL','I'])&~np.isin(residues,['P','TE','BR','CL','I'])
    for n in neighs: 
        current_nos_mask = nos_mask[n]
        current_nion_mask = neg_ion_mask[n]
        current_etc_mask = etc_mask[n]        
        n_lig_NOS.append(current_nos_mask.sum())
        n_lig_nion.append(current_nion_mask.sum())
        n_lig_etc.append(current_etc_mask.sum())
    return n_lig_NOS, n_lig_nion, n_lig_etc
#protein
def near_res(residues, atom_names, elems, xyzs, grids, threshold =3.2):
    kd_grid = scipy.spatial.cKDTree(grids)
    kd_xyzs = scipy.spatial.cKDTree(xyzs)
    neighs = kd_grid.query_ball_tree(kd_xyzs, threshold)
    n_coords_res = []
    n_core_res = []
    n_bb_coords = []
    target_residues = ['ASP', 'HIS', 'GLU', 'CYS']
    # Precompute masks for N, O, S atoms
    nos_mask = np.isin(elems, ['N', 'O', 'S'])

    # Precompute masks for target residues and backbone coordinates
    res_mask = np.isin(residues, target_residues)
    bb_mask = (elems == 'O') | (atom_names == 'O') | (atom_names == 'OXT')

    # Initialize lists to store results
    n_coords_res = []
    n_core_res = []
    n_bb_coords = []

    for n in neighs:
        # Get the subset of the precomputed masks for the current neighborhood
        current_nos_mask = nos_mask[n]
        current_res_mask = res_mask[n]
        current_bb_mask = bb_mask[n]

        # Count N, O, S atoms
        n_coords_res.append(current_nos_mask.sum())

        # Count core residues (target residues that are also N, O, S atoms)
        n_core_res.append((current_res_mask & current_nos_mask).sum())

        # Count backbone coordinates (O, OXT)
        n_bb_coords.append(current_bb_mask.sum())

    return n_coords_res, n_core_res, n_bb_coords

#lig+protein
def binned_res(residues, atom_names, elems, xyzs, grids, lower=3.0, upper=5.0):
    kd_grid = scipy.spatial.cKDTree(grids)
    kd_xyzs = scipy.spatial.cKDTree(xyzs)
    
    inner = kd_grid.query_ball_tree(kd_xyzs, lower)
    outer = kd_grid.query_ball_tree(kd_xyzs, upper)
    
    target_residues = ['ASP', 'HIS', 'GLU', 'CYS']
    
    # Precompute masks for N, O, S atoms
    nos_mask = np.isin(elems, ['N', 'O', 'S'])

    # Precompute masks for target residues and backbone coordinates
    res_mask = np.isin(residues, target_residues)
    bb_mask = (elems == 'O') | (atom_names == 'O') | (atom_names == 'OXT') | (np.isin(residues, standard_residues))

    # Initialize lists to store results
    n_coords_res = []
    n_core_res = []
    n_bb_coords = []

    for inner_n, outer_n in zip(inner, outer):
        # Find neighbors within the 3-5 Å range
        neighs = np.setdiff1d(outer_n, inner_n)
        
        # Get the subset of the precomputed masks for the current neighborhood
        current_nos_mask = nos_mask[neighs]
        current_res_mask = res_mask[neighs]
        current_bb_mask = bb_mask[neighs]

        # Count N, O, S atoms
        n_coords_res.append(current_nos_mask.sum())

        # Count core residues (target residues that are also N, O, S atoms)
        n_core_res.append((current_res_mask & current_nos_mask).sum())
        
        # Count backbone coordinates (O, OXT)
        n_bb_coords.append(current_bb_mask.sum())

    return n_coords_res, n_core_res, n_bb_coords



def parse_pdb(structure):
    data = structure
    res_ids = data.residue_idxs  # 'chain:res_id' 형식
    residues = data.atom_residues
    atom_names = data.atom_names
    xyzs = data.atom_positions
    grids = data.grid_positions
    elems = data.atom_elements
    chain_ids = data.chain_ids
    # Initialize dictionary for residue index
    res_id_dict = {}
    index = 0  # 고유 인덱스 시작점

    # Iterate over 'res_ids' to assign unique index for each 'chain:res_id'
    for chain_id,res_id in zip(chain_ids, res_ids):
        if f'{chain_id}:{res_id}' not in res_id_dict:
            res_id_dict[f'{chain_id}:{res_id}'] = index  # 'chain:res_id'에 대해 고유 인덱스 생성
            index += 1  # 인덱스 증가

    max_num_res = len(res_id_dict)
    
    # Initialize arrays for atoms
    alphas = np.zeros((max_num_res, 3))
    betas = np.zeros((max_num_res, 3))
    carbons = np.zeros((max_num_res, 3))
    oxygens = np.zeros((max_num_res, 3))
    nitrogens = np.zeros((max_num_res, 3))
    res_names = np.empty((max_num_res,), dtype=object)
    atom_dict = {}

    # Iterate over the atoms and assign coordinates using the unique index
    for chain_id, res_id, res, atom_name, xyz in zip(chain_ids, res_ids, residues, atom_names, xyzs):
        idx = res_id_dict[f'{chain_id}:{res_id}']  # 'chain:res_id'에 대한 고유 인덱스를 가져옴
        
        if atom_name == 'CA':
            alphas[idx] = xyz
        elif atom_name == 'CB':
            betas[idx] = xyz
        elif atom_name == 'C':
            carbons[idx] = xyz
        elif atom_name == 'N':
            nitrogens[idx] = xyz
        elif atom_name == 'O':
            oxygens[idx] = xyz
        
        res_names[idx] = res

    # Store arrays in atom_dict
    atom_dict['alphas'] = alphas
    atom_dict['betas'] = betas
    atom_dict['carbons'] = carbons
    atom_dict['oxygens'] = oxygens
    atom_dict['nitrogens'] = nitrogens
    atom_dict['res'] = res_names  # 'chain:res_id'와 인덱스의 매핑
    
    return atom_dict, grids


def filter_by_biometall(grid, atom_dict):
    # Ensure all grid and atom_dict arrays are in float32
    grid = grid.astype(np.float32)
    
    alphas = atom_dict['alphas'].astype(np.float32)
    betas = atom_dict['betas'].astype(np.float32)
    # carbons = atom_dict['carbons'].astype(np.float32)
    # oxygens = atom_dict['oxygens'].astype(np.float32)
    # nitrogens = atom_dict['nitrogens'].astype(np.float32)
    res = np.array(atom_dict['res'])  # res는 문자열 데이터이므로 변환하지 않음
    
    core_res = ['HIS', 'GLU', 'ASP', 'CYS']

    # Calculate distances and angles (ensuring all operations are float32)
    alpha_beta_distances = np.linalg.norm(betas - alphas, axis=1).astype(np.float32)
    # oxygen_carbon_distances = np.linalg.norm(carbons - oxygens, axis=1).astype(np.float32)

    alpha_distances = np.linalg.norm(grid[:, np.newaxis, :] - alphas[np.newaxis, :, :], axis=2).astype(np.float32)
    beta_distances = np.linalg.norm(grid[:, np.newaxis, :] - betas[np.newaxis, :, :], axis=2).astype(np.float32)
    # carbon_distances = np.linalg.norm(grid[:, np.newaxis, :] - carbons[np.newaxis, :, :], axis=2).astype(np.float32)
    # nitrogen_distances = np.linalg.norm(grid[:, np.newaxis, :] - nitrogens, axis=2).astype(np.float32)
    # oxygen_distances = np.linalg.norm(grid[:, np.newaxis, :] - oxygens[np.newaxis, :, :], axis=2).astype(np.float32)

    PAB_angles = np.arccos(
        (np.square(alpha_distances) + np.square(alpha_beta_distances) - np.square(beta_distances)) /
        (2 * alpha_distances * alpha_beta_distances)
    ).astype(np.float32)

    # POC_angles = np.arccos(
    #     (np.square(oxygen_distances) + np.square(oxygen_carbon_distances) - np.square(carbon_distances)) /
    #     (2 * oxygen_distances * oxygen_carbon_distances)
    # ).astype(np.float32)

    grid_indices = []
    for res_name in core_res:
        index = np.where(
            (DIST_PROBE_ALPHA[res_name][0] <= alpha_distances) & (alpha_distances <= DIST_PROBE_ALPHA[res_name][1]) &
            (DIST_PROBE_BETA[res_name][0] <= beta_distances) & (beta_distances <= DIST_PROBE_BETA[res_name][1]) &
            (ANGLE_PAB[res_name][0] <= PAB_angles) & (PAB_angles <= ANGLE_PAB[res_name][1])
        )

        res_index = index[1]
        grid_index = index[0]
        restype_mask = np.isin(res[res_index], res_name)
        grid_index = grid_index[restype_mask]
        grid_indices.append(grid_index)
    
    # Concatenate all grid indices
    grid_indices = np.concatenate(grid_indices)

    # Use np.bincount to count how many times each grid index appears
    num_res_list = np.bincount(grid_indices, minlength=len(grid)).astype(np.float32)

    return num_res_list

#protein+ligand
def nearest_prot_carbon_dist(residues, atom_names, elems, xyzs, grids, aliphatic_carbons, aromatic_carbons):
    # Gather coordinates for normal (aliphatic + aromatic) carbons
    normal_carbons = []
    
    for i, (res, atom, xyz, elem) in enumerate(zip(residues, atom_names, xyzs, elems)):
        if res in aliphatic_carbons and atom in aliphatic_carbons[res]:
            normal_carbons.append(xyz.astype(np.float32))  # Cast coordinates to float32
  
        if res in aromatic_carbons and atom in aromatic_carbons[res]:
            normal_carbons.append(xyz.astype(np.float32))  # Cast coordinates to float32
        
        if (res not in standard_residues) and (elem == 'C'):
            normal_carbons.append(xyz.astype(np.float32))

    normal_carbons = np.array(normal_carbons, dtype=np.float32)  # Ensure the array is float32
    
    # Ensure grids are also float32 to save memory
    grids = grids.astype(np.float32)
    
    # Calculate distances from each grid point to the nearest normal carbon atom
    dist = np.sqrt(np.sum(np.square(grids[:, np.newaxis, :] - normal_carbons[np.newaxis, :, :]), axis=-1), dtype=np.float32)
    min_dist = np.min(dist, axis=-1)

    return min_dist

#protein
def nearest_bb_dist(atom_names, xyzs, grids):
    # Gather coordinates for bb carbon and nitrogens
    bb_cn = []
    for i, (atom, xyz) in enumerate(zip(atom_names,xyzs)):
        if atom in ['C','N']:
            bb_cn.append(xyz.astype(np.float32))

    bb_cn = np.array(bb_cn,dtype=np.float32)
    grids = np.array(grids,dtype=np.float32)
    # Calculate distances from each grid point to the nearest normal carbon atom
    dist = np.sqrt(np.sum(np.square(grids[:, np.newaxis, :] - bb_cn[np.newaxis, :, :]), axis=-1),dtype=np.float32)
    min_dist = np.min(dist, axis=-1)

    return min_dist

def sasa_grids(grids, n_samples):
    probe_radius = 1.4
    radii = np.ones_like(len(grids)) * 1.5 #avg metal vdw = 1.5
    m = len(grids)
    centers = grids
    inc = np.pi * (3 - np.sqrt(5))  # increment
    off = 2.0 / n_samples

    pts0 = []
    for k in range(n_samples):
        phi = k * inc
        y = k * off - 1 + (off / 2)
        r = np.sqrt(1 - y * y)
        pts0.append([np.cos(phi) * r, y, np.sin(phi) * r])
    pts0 = np.array(pts0)  # (g,3)
    pts0 = pts0[np.newaxis, :, :].repeat(m, axis=0)  # (m,g,3)

    sasa_grids = pts0 * (radii+ probe_radius) + centers[:, np.newaxis, :] 
    
    return sasa_grids

def filter_by_contact(grids, elems, xyzs):
    # Separate xyz arrays for each element
    unique_elems = np.array(elems)
    xyz_C = xyzs[unique_elems == 'C']
    xyz_N = xyzs[unique_elems == 'N']
    xyz_O = xyzs[unique_elems == 'O']
    xyz_S = xyzs[unique_elems == 'S']
    xyz_X = xyzs[~np.isin(unique_elems, ['C', 'N', 'O', 'S'])]  # Other elements
    
    # Create KD trees for each element
    tree_C = scipy.spatial.cKDTree(xyz_C) if xyz_C.size > 0 else None
    tree_N = scipy.spatial.cKDTree(xyz_N) if xyz_N.size > 0 else None
    tree_O = scipy.spatial.cKDTree(xyz_O) if xyz_O.size > 0 else None
    tree_S = scipy.spatial.cKDTree(xyz_S) if xyz_S.size > 0 else None
    tree_X = scipy.spatial.cKDTree(xyz_X) if xyz_X.size > 0 else None

    # Radii for each element
    radius_C = VAN_DER_WAALS_RADII['C']
    radius_N = VAN_DER_WAALS_RADII['N']
    radius_O = VAN_DER_WAALS_RADII['O']
    radius_S = VAN_DER_WAALS_RADII['S']
    radius_X = radius_C  # Use Carbon radius for other elements

    # Flatten the grids array
    num_metals, num_grids, _ = grids.shape
    flattened_grids = grids.reshape(-1, 3)
    grid_tree = scipy.spatial.cKDTree(flattened_grids)

    # Check for clashes using query_ball_tree
    clash_C = tree_C.query_ball_tree(grid_tree, radius_C) if tree_C else []
    clash_N = tree_N.query_ball_tree(grid_tree, radius_N) if tree_N else []
    clash_O = tree_O.query_ball_tree(grid_tree, radius_O) if tree_O else []
    clash_S = tree_S.query_ball_tree(grid_tree, radius_S) if tree_S else []
    clash_X = tree_X.query_ball_tree(grid_tree, radius_X) if tree_X else []
    
    # Combine all clashes
    all_clashes = clash_C + clash_N + clash_O + clash_S + clash_X
    clash_index = np.unique(np.concatenate(all_clashes)).astype(int) if all_clashes else []
    
    non_clashing_mask = np.ones(flattened_grids.shape[0], dtype=bool)
    non_clashing_mask[clash_index] = False
    
    # Reshape non_clashing_mask back to (metals, grids)
    non_clashing_mask = non_clashing_mask.reshape(num_metals, num_grids)
    remained_grids_list = np.sum(non_clashing_mask, axis=-1)

    return remained_grids_list


def RSA(grids,xyzs,elems,n_samples=50):
    n_samples = 50
    grids = sasa_grids(grids, n_samples)
    remained_grids_list = filter_by_contact(grids, elems, xyzs)
    sasa_scores = remained_grids_list / n_samples
    return sasa_scores
