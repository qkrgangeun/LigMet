import numpy as np
import torch  # type: ignore
import dgl  # type: ignore
from scipy.spatial import cKDTree  # type: ignore
from pathlib import Path
import torch.nn.functional as F  # type: ignore
from typing import Tuple, Union
from ligmet.featurizer import Features, Info, make_features # type: ignore
from ligmet.utils.constants import metals, standard_residues,ATOMIC_NUMBERS, atype2num,sec_struct_dict, sybyl_type_dict  # type: ignore
from ligmet.utils.pdb import read_pdb, StructureWithGrid
from ligmet.utils.grid import sasa_grids_thread, filter_by_clashmap
from dataclasses import asdict
import math
from tqdm import tqdm
import torch.nn as nn
class PreprocessedDataSet(torch.utils.data.Dataset):
    def __init__(self, data_file: str, features_dir: str, rf_result_dir: str,topk: int, edge_dist_cutoff: float, pocket_dist: float, rf_threshold: float, eps=1e-6):
        super().__init__()
        self.data_file=Path(data_file)
        self.features_dir=Path(features_dir)
        self.rf_result_dir=Path(rf_result_dir)
        self.topk = topk
        self.edge_dist_cutoff=edge_dist_cutoff
        self.pocket_dist=pocket_dist
        self.rf_threshold=rf_threshold
        self.pdbid_lists=[pdb.strip().split(".pdb")[0] for pdb in open(data_file) if (self.features_dir / f"{pdb.strip().split('.pdb')[0]}.npz").exists()]
        self.eps = eps
        self.alpha = 4/math.log(2) #5.77078
        self.metal_dir = Path('/home/qkrgangeun/LigMet/data/biolip/metal_label')
        self.relpos_embedding = nn.Embedding(65, 8)  # relative position embedding, dim=8
        print(self.features_dir)
    def __len__(self):
        return len(self.pdbid_lists)
    
    def __getitem__(self, index:int):
        G = []
        L = []
        pdb_id = self.pdbid_lists[index]
        feature_path = self.features_dir / f"{pdb_id}.npz"
        # print(feature_path)
        rf_result_path = self.rf_result_dir / f"{pdb_id}.npz"
        data = np.load(feature_path,allow_pickle=True)
        metal = np.load(self.metal_dir/f"{pdb_id}.npz", allow_pickle=True)
        # print(self.metal_dir/f"{pdb_id}.npz")
        features = Features(**data)
        features.metal_positions = metal["metal_positions"]
        features.metal_types = metal["metal_types"]
        ##metalpred## -> should # 2 above line
        
        if len(features.atom_positions) != len(features.atom_elements):
            raise Exception(feature_path)
        if features.bond_masks.shape != (len(features.atom_elements), len(features.atom_elements)):
        #memory issue: bond mask (l,k) neighbors -> n * n matrix
            bonds_mask = self.neigh_to_bondmask(features)
            features.bond_masks = bonds_mask
        
        grid_positions = features.grid_positions
        grid_data = np.load(rf_result_path)
        grid_probs = grid_data["prob"] 
        #metalpred-rf##
        # grid_positions = grid_data["grid_positions"]
        # grid_probs = grid_data["grid_probs"]
        #########
        grid_mask = grid_probs >= self.rf_threshold
        grids_after_rf = grid_positions[grid_mask]

        grids_after_rf = np.concatenate((grids_after_rf,features.metal_positions),axis=0)
        features.grid_positions = grids_after_rf
        # features_p, pocket_exist = self.find_pocket(features, grids_after_rf)
        
        # if pocket_exist is False:
        #     raise AttributeError("there is no grids after randomforest")
        
        # g = self.make_graph(features_p)
        g = self.make_graph(features)
        # l_prob, l_type, l_vector = self.make_label(features_p)
        l_prob, l_type, l_vector = self.make_label(features)
        labels = torch.cat([l_prob.unsqueeze(1), l_type.unsqueeze(1), l_vector], dim=1)  # shape [N,5]
        G.append(g)
        L.append(labels)
        if not G:
            raise AttributeError(f"{pdb_id} have none type graph")
        info = Info(
            pdb_id=np.array(pdb_id),
            grids_positions=torch.tensor(grids_after_rf, dtype=torch.float32),
            metal_positions=torch.tensor(features.metal_positions, dtype=torch.float32),
            metal_types=torch.tensor([metals.index(metal) for metal in features.metal_types]),
        )
        return G, L, info

    def neigh_to_bondmask(self, features:Features):
        # bond_mask : [[row], [col]]
        n_atom = len(features.atom_names)
        cov_bonds_mask = np.zeros((n_atom,n_atom))
        neigh = features.bond_masks
        cov_bonds_mask[neigh[0],neigh[1]] = 1
        cov_bonds_mask[neigh[1],neigh[0]] = 1
        return cov_bonds_mask
    
    def find_pocket(self, features: Features, grids: np.ndarray):
        c_grids = grids
        atom_pos = features.atom_positions
        size = len(features.metal_positions)
        # if len(features.metal_positions) > 10:
        perturb_size = np.random.uniform(0,3,size=features.metal_positions.shape[0])
        perturb_direct = np.random.randn(features.metal_positions.shape[0],3)
        normalized_direct = perturb_direct / np.linalg.norm(perturb_direct, axis=-1)[:,None]
        perturbation = normalized_direct * perturb_size[:,None]
        perturbed_metal_positions = features.metal_positions + perturbation
        #TODO size random
        random_size = np.random.randint(0,2*size)
        random_int = np.random.choice(len(grids), size=random_size, replace=False)
        combined_positions = np.vstack([c_grids[random_int], perturbed_metal_positions])

        gtree = cKDTree(c_grids)
        ptree = cKDTree(atom_pos)
        
        # all_positions = np.vstack([features.atom_positions,grids])
        # tree = cKDTree(all_positions)
        if len(features.metal_positions) > 10:
            mtree = cKDTree(perturbed_metal_positions)
            target = mtree
            # print('perturbed_metal_positions',len(perturbed_metal_positions))
        else:
            rtree = cKDTree(combined_positions)
            target = rtree
            # print('len(combined_positions), metal',len(combined_positions), size)

        # ii = target.query_ball_tree(ptree, self.pocket_dist)
        jj = target.query_ball_tree(gtree, self.pocket_dist)
        # idx = np.unique(np.concatenate([i for i in ii if i], axis=0)).astype(int)
        jdx = np.unique(np.concatenate([j for j in jj if j], axis=0)).astype(int)

        # if len(idx) == 0:
        #     return None, False
        # c_features = Features(
        #     atom_positions=atom_pos[idx],
        #     atom_names=features.atom_names[idx],
        #     atom_elements=features.atom_elements[idx],
        #     atom_residues=features.atom_residues[idx],
        #     residue_idxs=features.residue_idxs[idx],
        #     chain_ids=features.chain_ids[idx],
        #     is_ligand=features.is_ligand[idx],
        #     metal_positions=features.metal_positions, 
        #     metal_types=features.metal_types,
        #     grid_positions=c_grids[jdx],
        #     sasas=features.sasas[idx],
        #     qs=features.qs[idx],
        #     sec_structs=features.sec_structs[idx],
        #     gen_types=features.gen_types[idx],
        #     bond_masks=features.bond_masks[np.ix_(idx, idx)], 
        # )
        c_features = Features(
            atom_positions=atom_pos,
            atom_names=features.atom_names,
            atom_elements=features.atom_elements,
            atom_residues=features.atom_residues,
            residue_idxs=features.residue_idxs,
            chain_ids=features.chain_ids,
            is_ligand=features.is_ligand,
            metal_positions=features.metal_positions, 
            metal_types=features.metal_types,
            grid_positions=c_grids[jdx],
            sasas=features.sasas,
            qs=features.qs,
            sec_structs=features.sec_structs,
            gen_types=features.gen_types,
            bond_masks=features.bond_masks, 
        )
        return c_features, True
    
    def make_label(self, features:Features)->Union[torch.Tensor, torch.Tensor, torch.Tensor]:
        grid = np.array(features.grid_positions, dtype=np.float32)
        grids = torch.from_numpy(grid)
        metal_pos = torch.from_numpy(features.metal_positions)
        metal_types = torch.tensor([metals.index(metal) for metal in features.metal_types])

        diff = grids.unsqueeze(1) - metal_pos.unsqueeze(0)  # [g,m,3]
        dist = torch.sqrt(torch.sum(diff**2, dim=-1)) + self.eps  # [g,m]

        exp_dist = torch.exp(-(dist**2) / self.alpha)
        label_p, _ = torch.max(exp_dist, dim=-1)
        label_prob = torch.where(label_p <= 0.1, torch.tensor(0.0, dtype=label_p.dtype, device=label_p.device), label_p)

        min_dist, min_idx = torch.min(dist, dim=-1)  # [g,]
        label_type = torch.where(
            min_dist <= 2.0, metal_types[min_idx], torch.tensor(len(metals))
        )
        label_vector = diff[torch.arange(diff.size(0)), min_idx]
        
        return label_prob, label_type, label_vector
    
    def make_graph(self, features: Features) -> dgl.DGLGraph:
        xyz = torch.tensor(np.concatenate([features.atom_positions, features.grid_positions]))
        grid_mask = torch.ones(len(xyz))
        grid_mask[: len(features.sasas)] = 0
        n_feats, n_polar_vec = self.get_node_features(features)
        num_nodes = xyz.shape[0]
        edge_index_src, edge_index_dst, e_feats, rel_vec = self.make_edge(features)
        G = dgl.graph((edge_index_src.to(torch.int32), edge_index_dst.to(torch.int32)),num_nodes=num_nodes)
        G.ndata["xyz"] = xyz.to(torch.float32)
        G.ndata["L0"] = n_feats.to(torch.float32)
        G.ndata["L1"] = n_polar_vec.to(torch.float32)
        G.ndata["grid_mask"] = grid_mask.to(torch.float32)
        G.edata["L0"] = e_feats.to(torch.float32)
        G.edata["L1"] = rel_vec.to(torch.float32)
        # print('graph node개수:',len(xyz))
        # print('graph protein node개수:', len(features.atom_positions))
        # print('graph grid node개수:',len(features.grid_positions))
        # print('graph edge개수:',len(e_feats))
        return G
    
    def make_polarity_vector(self, features: Features) -> np.ndarray:
        xyz = torch.from_numpy(features.atom_positions)
        neigh_masks = torch.from_numpy(features.bond_masks)

        self_idx, nei_idx = torch.nonzero(neigh_masks, as_tuple=True)

        xyz_self = xyz * neigh_masks.sum(dim=1, keepdim=True)
        xyz_nei = -xyz[nei_idx].to(xyz_self.dtype)
        xyz_self.scatter_add_(0, self_idx[:, None].expand(-1, 3), xyz_nei)

        polar_vec = F.normalize(xyz_self, dim=1)
        polarity_vectors = torch.cat(
            [polar_vec, torch.zeros(features.grid_positions.shape)], dim=0
        ).numpy()
        return polarity_vectors

    def get_node_features(self, features: Features) -> Tuple[torch.Tensor, torch.Tensor]:
        num_grids = len(features.grid_positions)

        sasas = torch.from_numpy(features.sasas)
        qs = torch.from_numpy(features.qs)
        sec_structs = torch.from_numpy(features.sec_structs)
        atom_gentype = torch.from_numpy(features.gen_types)
        
        # One-hot features: aatype, atomtype, 2nd structures
        aatype = torch.Tensor([
            standard_residues.index(res) if res in standard_residues else len(standard_residues)
            for res in features.atom_residues
        ])
        grids_aatype = torch.ones(num_grids) * (len(standard_residues) + 1)
        aatype = torch.cat((aatype, grids_aatype))

        atomtype = torch.Tensor([ATOMIC_NUMBERS.get(elem, 119) for elem in features.atom_elements])
        grids_atomtype = torch.zeros(num_grids)
        atomtype = torch.cat([atomtype, grids_atomtype], dim=0)

        # Ligand gen_type
        grids_atomchemtype = torch.ones(num_grids) * (max(sybyl_type_dict.values()) + 1)
        atom_chem_type = torch.cat([atom_gentype, grids_atomchemtype], dim=0)

        grids_2nd = torch.ones(num_grids) * len(sec_struct_dict)
        sec_structs = torch.cat([sec_structs, grids_2nd])
        node_type = torch.cat([torch.from_numpy(features.is_ligand),torch.ones_like(grids_2nd)*2])
        # One-hot encoding
        aatype = F.one_hot(aatype.to(torch.int64), num_classes=len(standard_residues) + 2)
        atomtype = F.one_hot(atomtype.to(torch.int64), num_classes=len(ATOMIC_NUMBERS) + 2)
        sec_structs = F.one_hot(sec_structs.to(torch.int64), num_classes=len(sec_struct_dict) + 1)
        atom_chemtype = F.one_hot(atom_chem_type.to(torch.int64), num_classes=max(sybyl_type_dict.values()) + 2)
        node_type = F.one_hot(node_type.to(torch.int64), num_classes=3)
        # Real value features: sasas, qs (assign 0 for grids)
        grids_feat = torch.zeros(num_grids)
        sasas = torch.cat((sasas, grids_feat)).unsqueeze(-1)
        qs = torch.cat((qs, grids_feat)).unsqueeze(-1)
        sasas = sasas + self.eps

        # 🔹 NaN 값 변환 (모든 feature에 적용)
        # sasas = torch.nan_to_num(sasas, nan=0.0, posinf=0.0, neginf=0.0)
        # qs = torch.nan_to_num(qs, nan=0.0, posinf=0.0, neginf=0.0)
        # sec_structs = torch.nan_to_num(sec_structs, nan=0.0, posinf=0.0, neginf=0.0)
        # atom_chemtype = torch.nan_to_num(atom_chemtype, nan=0.0, posinf=0.0, neginf=0.0)
        # aatype = torch.nan_to_num(aatype, nan=0.0, posinf=0.0, neginf=0.0)
        # atomtype = torch.nan_to_num(atomtype, nan=0.0, posinf=0.0, neginf=0.0)

        # 모든 feature 합치기
        n_feats = torch.cat([aatype, atomtype, atom_chemtype, sec_structs, sasas, qs, node_type], dim=1)
        n_feats = torch.nan_to_num(n_feats, nan=0.0, posinf=0.0, neginf=0.0)
        # # NaN 값 체크 (Debug)
        # print(f"NaN in sasas: {torch.isnan(sasas).sum().item()}")
        # print(f"NaN in qs: {torch.isnan(qs).sum().item()}")
        # print(f"NaN in sec_structs: {torch.isnan(sec_structs).sum().item()}")
        # print(f"NaN in atom_gentype: {torch.isnan(atom_chemtype).sum().item()}")
        # print(f"NaN in aatype: {torch.isnan(aatype).sum().item()}")
        # print(f"NaN in atomtype: {torch.isnan(atomtype).sum().item()}")

        # Polarity vector 처리
        polarity_vectors = self.make_polarity_vector(features)
        polarity_vectors = torch.tensor(polarity_vectors)
        polarity_vectors = torch.nan_to_num(polarity_vectors, nan=0.0, posinf=0.0, neginf=0.0)

        return n_feats, polarity_vectors


    def onehot_edge_dist(self, dists: torch.Tensor) -> torch.Tensor:
        bin_edges = np.arange(0, self.edge_dist_cutoff + 0.5, 0.1)
        dist_binned = np.digitize(dists, bins=bin_edges) - 1
        one_hot_dist = F.one_hot(
            torch.from_numpy(dist_binned), num_classes=len(bin_edges)
        )
        return one_hot_dist

    def onehot_edge_type(
        self, edge_index_src: torch.Tensor, edge_index_dst: torch.Tensor, num_atom: int
    ) -> torch.Tensor:
        feat = np.zeros_like(edge_index_src)  # p to p :0
        feat[np.where((edge_index_src < num_atom) & (edge_index_dst >= num_atom))] = (
            1  # p to g :1
        )
        feat[np.where((edge_index_src >= num_atom) & (edge_index_dst < num_atom))] = (
            2  # g to p : 2
        )
        feat[np.where((edge_index_src >= num_atom) & (edge_index_dst >= num_atom))] = (
            3  # g to g :3
        )
        one_hot_feat = F.one_hot(torch.from_numpy(feat).to(torch.int64), num_classes=4)
        return one_hot_feat

    def cov_bond(
        self,
        edge_index_src: torch.Tensor,
        edge_index_dst: torch.Tensor,
        num_atom: int,
        features: Features,
    ) -> torch.Tensor:
        # shape (edge, )
        cov_bond = np.zeros(len(edge_index_src))
        prot_idx_mask = (edge_index_src < num_atom) & (edge_index_dst < num_atom)
        idx = (edge_index_src[prot_idx_mask], edge_index_dst[prot_idx_mask])
        cov_bond[prot_idx_mask] = features.bond_masks[tuple(idx)]
        cov_bond = torch.from_numpy(cov_bond)
        return cov_bond

    def make_edge(
        self, features: Features) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        num_atom = len(features.atom_positions)
        num_grids = len(features.grid_positions)
        num_nodes = num_atom + num_grids

        node_pos = np.concatenate([features.atom_positions, features.grid_positions], axis=0)
        k_nearest = min(self.topk + 1, num_nodes)

        tree = cKDTree(node_pos)
        dd, ii = tree.query(
            node_pos, k=k_nearest, distance_upper_bound=self.edge_dist_cutoff
        )
        node_pos = torch.from_numpy(node_pos).to(torch.float32)
        index_tensor = torch.arange(num_nodes, dtype=torch.int32)
        edge_index_src = torch.flatten(torch.from_numpy(ii)).to(torch.int32)
        edge_index_dst = torch.repeat_interleave(index_tensor, k_nearest)
        dists = torch.flatten(torch.from_numpy(dd))

        edge_mask = torch.logical_and(edge_index_src != edge_index_dst, edge_index_src != num_nodes)

        # else:
        #     mask = torch.logical_and(
        #         edge_index_src != edge_index_dst, edge_index_src != num_nodes
        #     )
        #     p2p_mask = (
        #         (edge_index_src < num_atom)
        #         & (edge_index_dst < num_atom)
        #         & (dists <= self.p_dist_cutoff)
        #     )
        #     g2g_mask = (
        #         (edge_index_src >= num_atom)
        #         & (edge_index_dst >= num_atom)
        #         & (dists <= self.g_dist_cutoff)
        #     )
        #     pg2gp_mask = (edge_index_src < num_atom) ^ (edge_index_dst < num_atom)
        #     edge_mask = torch.logical_and(mask, (p2p_mask | g2g_mask | pg2gp_mask))

        edge_index_src = edge_index_src[edge_mask].long()
        edge_index_dst = edge_index_dst[edge_mask].long()
        dists = dists[edge_mask]

        dist_bin = self.onehot_edge_dist(dists)
        onehot_type = self.onehot_edge_type(edge_index_src, edge_index_dst, num_atom)
        covalent_bond = self.cov_bond(edge_index_src, edge_index_dst, num_atom, features)
        covalent_bond = covalent_bond.unsqueeze(-1)
        # relative position
        e_vec = torch.tensor(
            node_pos[edge_index_dst.long()] - node_pos[edge_index_src.long()]
        )

        polarity_vectors = torch.tensor(
            self.make_polarity_vector(features), dtype=torch.float32
        )
        # edge_type을 설정: prot-to-prot, grid-to-grid, grid-to-prot, prot-to-grid 구분
        edge_type_prot_to_prot = (edge_index_src < num_atom) & (
            edge_index_dst < num_atom
        )
        edge_type_grid_to_grid = (edge_index_src >= num_atom) & (
            edge_index_dst >= num_atom
        )
        edge_type_grid_to_prot = (edge_index_src >= num_atom) & (
            edge_index_dst < num_atom
        )
        edge_type_prot_to_grid = (edge_index_src < num_atom) & (
            edge_index_dst >= num_atom
        )

        # 초기화
        start = torch.zeros((len(edge_index_src), 3), dtype=torch.float32)
        end = torch.zeros((len(edge_index_src), 3), dtype=torch.float32)

        # 1. prot to prot 또는 grid to grid
        mask = edge_type_prot_to_prot | edge_type_grid_to_grid
        start[mask] = polarity_vectors[edge_index_dst[mask].long()]
        end[mask] = polarity_vectors[edge_index_src[mask].long()]

        # 2. grid to prot
        mask = edge_type_grid_to_prot
        start[mask] = (
            node_pos[edge_index_src[mask].long()]
            - node_pos[edge_index_dst[mask].long()]
        )
        end[mask] = polarity_vectors[edge_index_dst[mask].long()]

        # 3. prot to grid
        mask = edge_type_prot_to_grid
        start[mask] = polarity_vectors[edge_index_src[mask].long()]
        end[mask] = (
            node_pos[edge_index_dst[mask].long()]
            - node_pos[edge_index_src[mask].long()]
        )

        cos = (
            torch.einsum(
                "ij,ij->i",
                start,
                end,
            ).unsqueeze(-1)
            + self.eps
        )
        sin = (
            torch.norm(
                torch.cross(
                    start,
                    end,
                ),
                dim=1,
                keepdim=True,
            )
            + self.eps
        )
        
        # 🔹 Relative residue index embedding (with chain consideration)
        residue_idx = torch.tensor(features.residue_idxs, dtype=torch.int64)
        chain_ids = torch.tensor([hash(c) % 997 for c in features.chain_ids], dtype=torch.int64)  # hash to int
        residue_idx_all = torch.cat([
            residue_idx,
            torch.full((len(features.grid_positions),), -999, dtype=torch.int64)
        ])
        chain_ids_all = torch.cat([
            chain_ids,
            torch.full((len(features.grid_positions),), -1, dtype=torch.int64)
        ])

        same_chain = chain_ids_all[edge_index_src] == chain_ids_all[edge_index_dst]
        rel_idx = residue_idx_all[edge_index_src.long()] - residue_idx_all[edge_index_dst.long()]

        rel_idx = torch.clamp(rel_idx, -32, 32) + 32
        rel_idx[~same_chain] = 64  # special index for inter-chain
        rel_emb = self.relpos_embedding(rel_idx)  # [E, 8]

        # 최종 edge feature
        e_feats = torch.cat([onehot_type, dist_bin, covalent_bond, cos, sin, rel_emb], dim=1)

        return edge_index_src, edge_index_dst, e_feats, e_vec
    
    @staticmethod
    def collate(samples: list) -> Tuple[dgl.DGLGraph, torch.Tensor, Info]:
        graphs, labels, g_pos, m_pos, m_types, pdb_ids = [], [], [], [], [], []

        for G, L, info in samples:
            if G is not None:
                graphs.extend(G)  # 각 샘플의 그래프 리스트를 하나의 리스트로 결합
                labels.extend(L)  # 각 샘플의 결합된 라벨 리스트를 하나의 리스트로 결합
                g_pos.append(info.grids_positions)
                m_pos.append(info.metal_positions)
                m_types.append(info.metal_types)
                pdb_ids.append(info.pdb_id)
        # 배치 그래프와 배치 라벨 생성
        batched_graphs = dgl.batch(graphs)  # shape [B*N]
        batched_labels = torch.cat(labels, dim=0)  # shape [B*N,2]
        g_poss = torch.cat(g_pos, dim=0)
        m_poss = torch.cat(m_pos, dim=0)
        m_typess = torch.cat(m_types, dim=0)
        pdb_idss = np.array(pdb_ids)
        batched_infos = Info(
            pdb_id=pdb_idss,
            grids_positions=g_poss,
            metal_positions=m_poss,
            metal_types=m_typess,
        )
        return batched_graphs, batched_labels, batched_infos

    
    
class OnTheFlyDataSet(PreprocessedDataSet):
    def __init__(self, data_file: str, pdb_dir: str, rf_model: str, topk: int, edge_dist_cutoff: float, pocket_dist: float, rf_threshold: float, eps=1e-6):
        dummy_dir = '/home/dummy'
        super().__init__(data_file, dummy_dir, dummy_dir, topk, edge_dist_cutoff, pocket_dist, rf_threshold, eps)
        self.pdb_dir = Path(pdb_dir)
        self.rf_model = rf_model  
        self.pdbid_lists = [pdb.strip().split(".pdb")[0] for pdb in open(data_file)]
        
    def __len__(self):
        return len(self.pdbid_lists)
    
    def __getitem__(self, index:int):
        G = []
        L = []
        pdb_id = self.pdbid_lists[index]
        pdb_path = self.pdb_dir / f'{pdb_id}.pdb'
        
        structure = read_pdb(pdb_path)
        grids = sasa_grids_thread(structure.atom_positions, structure.atom_elements)
        grids = filter_by_clashmap(grids)
        structure_dict = asdict(structure)
        structure_with_grid = StructureWithGrid(
            grid_positions= grids,
            **structure_dict  # structure_dict의 내용을 추가
        )
        features = make_features(pdb_path, structure_with_grid)
        # if features is None:
        #     return None, None, None
            
        grid_positions = features.grid_positions
        # print('원래 protein개수:', len(structure.atom_positions))
        # print('원래 grid개수:',len(grids))
        
        # Randomforest prediction # 
        # grid_probs = np.load(rf_result_path)
        # grid_mask = grid_probs >= self.rf_threshold
        # grids_after_rf = grid_positions[grid_mask]
        grids_after_rf = grid_positions
        features_p, pocket_exist = self.find_pocket(features, grids_after_rf)
        
        if pocket_exist is False:
            raise AttributeError("there is no grids after randomforest")
        
        g = self.make_graph(features_p)
        l_prob, l_type, l_vector = self.make_label(features_p)
        labels = torch.cat([l_prob.unsqueeze(1), l_type.unsqueeze(1), l_vector], dim=1)  # shape [N,5]
        G.append(g)
        L.append(labels)
        
        if not G:
            raise AttributeError(f"{pdb_id} have none type graph")
        
        info = Info(
            pdb_id=np.array(pdb_id),
            grids_positions=torch.tensor(grids_after_rf, dtype=torch.float32),
            metal_positions=torch.tensor(features.metal_positions, dtype=torch.float32),
            metal_types=torch.tensor([metals.index(metal) for metal in features.metal_types]),
        )
        return G, L, info
    
class TestDataSet(PreprocessedDataSet):
    def __init__(self, data_file: str, features_dir: str, rf_result_dir: str,topk: int, edge_dist_cutoff: float, pocket_dist: float, rf_threshold: float, eps=1e-6):
        super().__init__(
            data_file=data_file,  # Pass the data_file
            features_dir=features_dir,  # Pass the features_dir
            rf_result_dir=rf_result_dir,  # Pass the rf_result_dir
            topk=topk,  # Pass the topk
            edge_dist_cutoff=edge_dist_cutoff,  # Pass the edge_dist_cutoff
            pocket_dist=pocket_dist,  # Pass the pocket_dist
            rf_threshold=rf_threshold  # Pass the rf_threshold
        )
        self.data_file=Path(data_file)
        self.features_dir=Path(features_dir)
        self.rf_result_dir=Path(rf_result_dir)
        self.topk = topk
        self.edge_dist_cutoff=edge_dist_cutoff
        self.pocket_dist=pocket_dist
        self.rf_threshold=rf_threshold
        self.pdbid_lists=[pdb.strip().split(".pdb")[0] for pdb in open(data_file) if (self.features_dir / f"{pdb.strip().split('.pdb')[0]}.npz").exists()]
        self.eps = eps
        self.alpha = 4/math.log(2) #5.77078
        
    def __len__(self):
        return len(self.pdbid_lists)
    
    def __getitem__(self, index:int):
        G = []
        L = []
        pdb_id = self.pdbid_lists[index]
        feature_path = self.features_dir / f"{pdb_id}.npz"
        data = np.load(feature_path,allow_pickle=True)
        features = Features(**data)
        if features.bond_masks.shape != (len(features.atom_elements), len(features.atom_elements)):
            bonds_mask = self.neigh_to_bondmask(features)
            features.bond_masks = bonds_mask
        grid_positions = features.grid_positions

        grids_after_rf = grid_positions
        
        g = self.make_graph(features)
        l_prob, l_type, l_vector = self.make_label(features)
        labels = torch.cat([l_prob.unsqueeze(1), l_type.unsqueeze(1), l_vector], dim=1)  # shape [N,5]
        G.append(g)
        L.append(labels)
        
        if not G:
            raise AttributeError(f"{pdb_id} have none type graph")
        
        info = Info(
            pdb_id=np.array(pdb_id),
            grids_positions=torch.tensor(grids_after_rf, dtype=torch.float32),
            metal_positions=torch.tensor(features.metal_positions, dtype=torch.float32),
            metal_types=torch.tensor([metals.index(metal) for metal in features.metal_types]),
        )
        return G, L, info
    
    
class DataSetGraphCashe(PreprocessedDataSet):
    def __init__(self, data_file: str, features_dir: str, rf_result_dir: str,topk: int, edge_dist_cutoff: float, pocket_dist: float, rf_threshold: float, eps=1e-6): 
        super().__init__(
            data_file=data_file,  # Pass the data_file
            features_dir=features_dir,  # Pass the features_dir
            rf_result_dir=rf_result_dir,  # Pass the rf_result_dir
            topk=topk,  # Pass the topk
            edge_dist_cutoff=edge_dist_cutoff,  # Pass the edge_dist_cutoff
            pocket_dist=pocket_dist,  # Pass the pocket_dist
            rf_threshold=rf_threshold,  # Pass the rf_threshold
            eps=eps
        )
        self._g_cache = {}
        
    def __len__(self):
        return len(self.pdbid_lists)
    
    def __getitem__(self, index:int):
        G = []
        L = []
        pdb_id = self.pdbid_lists[index]
        
        if pdb_id in self._g_cache:
            print('*in_cashe', pdb_id)
            return self._g_cache[pdb_id]
        print('not_in_cashe', pdb_id)
        feature_path = self.features_dir / f"{pdb_id}.npz"
        print('\n')
        print(feature_path)
        rf_result_path = self.rf_result_dir / f"{pdb_id}.npz"
        data = np.load(feature_path,allow_pickle=True)
        metal = np.load(self.metal_dir/f"{pdb_id}.npz", allow_pickle=True)
        print(self.metal_dir/f"{pdb_id}.npz")
        features = Features(**data)
        features.metal_positions = metal["metal_positions"]
        features.metal_types = metal["metal_types"]
        if len(features.atom_positions) != len(features.atom_elements):
            raise Exception(feature_path)
        if features.bond_masks.shape != (len(features.atom_elements), len(features.atom_elements)):
        #memory issue: bond mask (l,k) neighbors -> n * n matrix
            bonds_mask = self.neigh_to_bondmask(features)
            features.bond_masks = bonds_mask
        
        grid_positions = features.grid_positions
        grid_data = np.load(rf_result_path)
        grid_probs = grid_data["prob"] 
        grid_mask = grid_probs >= self.rf_threshold
        grids_after_rf = grid_positions[grid_mask]
        grids_after_rf = np.concatenate((grids_after_rf,features.metal_positions),axis=0)
        features.grid_positions = grids_after_rf
        # features_p, pocket_exist = self.find_pocket(features, grids_after_rf)
        
        # if pocket_exist is False:
        #     raise AttributeError("there is no grids after randomforest")
        
        # g = self.make_graph(features_p)
        g = self.make_graph(features)
        # l_prob, l_type, l_vector = self.make_label(features_p)
        l_prob, l_type, l_vector = self.make_label(features)
        labels = torch.cat([l_prob.unsqueeze(1), l_type.unsqueeze(1), l_vector], dim=1)  # shape [N,5]
        G.append(g)
        L.append(labels)
        if not G:
            raise AttributeError(f"{pdb_id} have none type graph")
        info = Info(
            pdb_id=np.array(pdb_id),
            grids_positions=torch.tensor(grids_after_rf, dtype=torch.float32),
            metal_positions=torch.tensor(features.metal_positions, dtype=torch.float32),
            metal_types=torch.tensor([metals.index(metal) for metal in features.metal_types]),
        )
        self._g_cache[pdb_id] = (G, L, info)
        return G, L, info
        
class NPZCachedDataset(PreprocessedDataSet):
    def __init__(self, data_file: str, features_dir: str, rf_result_dir: str,
                 topk: int, edge_dist_cutoff: float, pocket_dist: float, rf_threshold: float, eps=1e-6):
        super().__init__(data_file, features_dir, rf_result_dir, topk,
                         edge_dist_cutoff, pocket_dist, rf_threshold, eps)

        self._cached_features = {}
        self._cached_metal = {}
        self._cached_rf = {}

        print("⏳ Caching .npz files into memory...")
        for pdb_id in tqdm(self.pdbid_lists):
            feat_path = self.features_dir / f"{pdb_id}.npz"
            if feat_path.exists():
                self._cached_features[pdb_id] = np.load(feat_path, allow_pickle=True)

            metal_path = self.metal_dir / f"{pdb_id}.npz"
            if metal_path.exists():
                self._cached_metal[pdb_id] = np.load(metal_path, allow_pickle=True)

            rf_path = self.rf_result_dir / f"{pdb_id}.npz"
            if rf_path.exists():
                self._cached_rf[pdb_id] = np.load(rf_path, allow_pickle=True)

        print(f"✅ Done caching {len(self._cached_features)} feature files.")
        print(f"🧠 Estimated NPZ cache memory: {self.format_bytes(self.get_total_cache_size())}")

    def __getitem__(self, index: int):
        G, L = [], []
        pdb_id = self.pdbid_lists[index]

        data = self._cached_features[pdb_id]
        metal = self._cached_metal[pdb_id]
        grid_data = self._cached_rf[pdb_id]

        features = Features(**data)
        features.metal_positions = metal["metal_positions"]
        features.metal_types = metal["metal_types"]

        if len(features.atom_positions) != len(features.atom_elements):
            raise Exception(f"[{pdb_id}] Mismatch in atom data.")
        if features.bond_masks.shape != (len(features.atom_elements), len(features.atom_elements)):
            features.bond_masks = self.neigh_to_bondmask(features)

        grid_positions = features.grid_positions
        grid_probs = grid_data["prob"]
        grid_mask = grid_probs >= self.rf_threshold
        grids_after_rf = grid_positions[grid_mask]
        grids_after_rf = np.concatenate((grids_after_rf, features.metal_positions), axis=0)
        features.grid_positions = grids_after_rf

        g = self.make_graph(features)
        l_prob, l_type, l_vector = self.make_label(features)
        labels = torch.cat([l_prob.unsqueeze(1), l_type.unsqueeze(1), l_vector], dim=1)

        G.append(g)
        L.append(labels)

        info = Info(
            pdb_id=np.array(pdb_id),
            grids_positions=torch.tensor(grids_after_rf, dtype=torch.float32),
            metal_positions=torch.tensor(features.metal_positions, dtype=torch.float32),
            metal_types=torch.tensor([metals.index(m) for m in features.metal_types]),
        )
        return G, L, info

    def get_total_cache_size(self) -> int:
        """Calculate total memory used by cached .npz files in bytes."""
        def get_npz_dict_size(npz_dict: dict) -> int:
            return sum(arr.nbytes for arr in npz_dict.values() if isinstance(arr, np.ndarray))

        total_bytes = 0
        for cache in [self._cached_features, self._cached_metal, self._cached_rf]:
            for npz_obj in cache.values():
                total_bytes += get_npz_dict_size(npz_obj)
        return total_bytes

    @staticmethod
    def format_bytes(num_bytes: int) -> str:
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if num_bytes < 1024:
                return f"{num_bytes:.2f} {unit}"
            num_bytes /= 1024
        return f"{num_bytes:.2f} PB"

class GraphCachedDataset(PreprocessedDataSet):
    def __init__(self, data_file: str, features_dir: str, rf_result_dir: str,
                 topk: int, edge_dist_cutoff: float, pocket_dist: float,
                 rf_threshold: float, eps=1e-6):
        super().__init__(data_file, features_dir, rf_result_dir,
                         topk, edge_dist_cutoff, pocket_dist, rf_threshold, eps)
        self._graph_cache = {}
        self._graph_size_cache = {}

        print("⏳ Caching all graphs into memory...")

        for pdb_id in self.pdbid_lists:
            try:
                feature_path = self.features_dir / f"{pdb_id}.npz"
                rf_result_path = self.rf_result_dir / f"{pdb_id}.npz"
                metal_path = self.metal_dir / f"{pdb_id}.npz"

                data = np.load(feature_path, allow_pickle=True)
                metal = np.load(metal_path, allow_pickle=True)
                rf_data = np.load(rf_result_path)

                features = Features(**data)
                features.metal_positions = metal["metal_positions"]
                features.metal_types = metal["metal_types"]

                if len(features.atom_positions) != len(features.atom_elements):
                    raise Exception(f"{pdb_id}: mismatched atom info")

                if features.bond_masks.shape != (len(features.atom_elements), len(features.atom_elements)):
                    features.bond_masks = self.neigh_to_bondmask(features)

                grid_probs = rf_data["prob"]
                grid_positions = features.grid_positions
                grid_mask = grid_probs >= self.rf_threshold
                grids_after_rf = grid_positions[grid_mask]
                grids_after_rf = np.concatenate([grids_after_rf, features.metal_positions], axis=0)
                features.grid_positions = grids_after_rf

                g = self.make_graph(features)
                l_prob, l_type, l_vector = self.make_label(features)
                labels = torch.cat([l_prob.unsqueeze(1), l_type.unsqueeze(1), l_vector], dim=1)

                info = Info(
                    pdb_id=np.array(pdb_id),
                    grids_positions=torch.tensor(grids_after_rf, dtype=torch.float32),
                    metal_positions=torch.tensor(features.metal_positions, dtype=torch.float32),
                    metal_types=torch.tensor([metals.index(m) for m in features.metal_types]),
                )

                self._graph_cache[pdb_id] = ([g], [labels], info)
                self._graph_size_cache[pdb_id] = self._estimate_sample_size([g], [labels])
                print(f"✅ Cached {pdb_id}")
            except Exception as e:
                print(f"⚠️ Failed to process {pdb_id}: {e}")

        print(f"✅ Cached {len(self._graph_cache)} / {len(self.pdbid_lists)} graphs")
        print(f"🧠 Total graph memory: {self.format_bytes(self.get_total_graph_cache_size())}")

    def __getitem__(self, index: int):
        pdb_id = self.pdbid_lists[index]
        return self._graph_cache[pdb_id]

    def _estimate_sample_size(self, G: list, L: list) -> int:
        size = 0
        for g in G:
            for key in g.ndata:
                size += g.ndata[key].element_size() * g.ndata[key].nelement()
            for key in g.edata:
                size += g.edata[key].element_size() * g.edata[key].nelement()
        for l in L:
            size += l.element_size() * l.nelement()
        return size

    def get_total_graph_cache_size(self) -> int:
        return sum(self._graph_size_cache.values())

    @staticmethod
    def format_bytes(num_bytes: int) -> str:
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if num_bytes < 1024:
                return f"{num_bytes:.2f} {unit}"
            num_bytes /= 1024
        return f"{num_bytes:.2f} PB"
    
class GraphFeatureCachedDataset(PreprocessedDataSet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.graph_cache = {}

        print("🔄 Caching node and edge features...")
        for pdb_id in tqdm(self.pdbid_lists):
            try:
                # Load and preprocess features
                feature_path = self.features_dir / f"{pdb_id}.npz"
                rf_result_path = self.rf_result_dir / f"{pdb_id}.npz"
                metal_path = self.metal_dir / f"{pdb_id}.npz"

                data = np.load(feature_path, allow_pickle=True)
                metal = np.load(metal_path, allow_pickle=True)
                rf_data = np.load(rf_result_path)

                features = Features(**data)
                features.metal_positions = metal["metal_positions"]
                features.metal_types = metal["metal_types"]
                if features is None:
                    continue

                if features.bond_masks.shape != (len(features.atom_elements), len(features.atom_elements)):
                    features.bond_masks = self.neigh_to_bondmask(features)

                rf_data = np.load(rf_result_path)
                grid_probs = rf_data["prob"]
                grid_mask = grid_probs >= self.rf_threshold
                grids_after_rf = features.grid_positions[grid_mask]
                grids_after_rf = np.concatenate((grids_after_rf, features.metal_positions), axis=0)
                features.grid_positions = grids_after_rf

                label_prob, label_type, label_vector = self.make_label(features)
                label = torch.cat([label_prob.unsqueeze(1), label_type.unsqueeze(1), label_vector], dim=1)

                # Features to cache
                node_feats, polar_vecs = self.get_node_features(features)
                edge_index_src, edge_index_dst, edge_feats, edge_rel_vecs = self.make_edge(features)
                grid_mask_tensor = torch.ones(len(node_feats))
                grid_mask_tensor[:len(features.sasas)] = 0

                self.graph_cache[pdb_id] = {
                    "node_feats": node_feats,
                    "polar_vecs": polar_vecs,
                    "grid_mask": grid_mask_tensor,
                    "edge_index_src": edge_index_src,
                    "edge_index_dst": edge_index_dst,
                    "edge_feats": edge_feats,
                    "edge_rel_vecs": edge_rel_vecs,
                    "label": label,
                    "grids_after_rf": torch.tensor(grids_after_rf, dtype=torch.float32),
                    "metal_positions": torch.tensor(features.metal_positions, dtype=torch.float32),
                    "metal_types": torch.tensor([metals.index(m) for m in features.metal_types]),
                }
            except Exception as e:
                print(f"❌ Failed to process {pdb_id}: {e}")
        print("✅ Graph feature caching complete.")

    def __getitem__(self, index: int):
        pdb_id = self.pdbid_lists[index]
        data = self.graph_cache[pdb_id]
        num_nodes = data["node_feats"].shape[0]

        g = dgl.graph((data["edge_index_src"], data["edge_index_dst"]), num_nodes=num_nodes)
        g.ndata["L0"] = data["node_feats"]
        g.ndata["L1"] = data["polar_vecs"]
        g.ndata["grid_mask"] = data["grid_mask"]
        g.edata["L0"] = data["edge_feats"]
        g.edata["L1"] = data["edge_rel_vecs"]

        info = Info(
            pdb_id=np.array(pdb_id),
            grids_positions=data["grids_after_rf"],
            metal_positions=data["metal_positions"],
            metal_types=data["metal_types"]
        )
        return [g], [data["label"]], info

import torch
import torch.nn.functional as F
import dgl
from torch.utils.data import Dataset
import numpy as np
from typing import Tuple, Dict
from ligmet.featurizer import Features, Info
from pathlib import Path
from ligmet.utils.constants import metals
from multiprocessing import Manager

class CachedEdgeBuilder:
    def __init__(self, edge_cache: Dict[str, dict], edge_dist_cutoff: float):
        self.edge_cache = edge_cache
        self.edge_dist_cutoff = edge_dist_cutoff

    def edge_type_index(self, src: torch.Tensor, dst: torch.Tensor, num_atom: int) -> torch.Tensor:
        edge_type = torch.zeros(len(src), dtype=torch.int64)
        edge_type[(src < num_atom) & (dst >= num_atom)] = 1
        edge_type[(src >= num_atom) & (dst < num_atom)] = 2
        edge_type[(src >= num_atom) & (dst >= num_atom)] = 3
        return edge_type

    def onehot_edge_dist(self, dists: torch.Tensor) -> torch.Tensor:
        bin_edges = torch.arange(0, self.edge_dist_cutoff + 0.5, 0.1)
        dist_binned = torch.bucketize(dists, bin_edges) - 1
        one_hot_dist = F.one_hot(dist_binned, num_classes=len(bin_edges))
        return one_hot_dist

    def build_graph_from_cache(self, pdb_id: str, num_nodes: int) -> dgl.DGLGraph:
        cached = self.edge_cache[pdb_id]
        dist_bin = self.onehot_edge_dist(cached["dist"])
        onehot_type = F.one_hot(cached["edge_type_idx"], num_classes=4)

        e_feats = torch.cat([
            onehot_type.to(torch.float32),
            dist_bin.to(torch.float32),
            cached["cov_bond"],
            cached["cos"],
            cached["sin"]
        ], dim=1)

        G = dgl.graph((cached["src"], cached["dst"]), num_nodes=num_nodes)
        G.edata["L0"] = e_feats
        G.edata["L1"] = cached["e_vec"]

        return G

    def cache_edges(self, pdb_id: str, src, dst, dist, cov_bond, cos, sin, e_vec, num_atom):
        edge_type_idx = self.edge_type_index(src, dst, num_atom)
        self.edge_cache[pdb_id] = {
            "src": src,
            "dst": dst,
            "dist": dist,
            "edge_type_idx": edge_type_idx,
            "cov_bond": cov_bond.unsqueeze(-1),
            "cos": cos,
            "sin": sin,
            "e_vec": e_vec
        }


class CachedEdgeDataset(PreprocessedDataSet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.edge_cache = Manager().dict()
        self.edge_builder = CachedEdgeBuilder(self.edge_cache, self.edge_dist_cutoff)

    def __len__(self):
        return len(self.pdbid_lists)

    def __getitem__(self, idx: int):
        pdb_id = self.pdbid_lists[idx]

        feature_path = self.features_dir / f"{pdb_id}.npz"
        rf_result_path = self.rf_result_dir / f"{pdb_id}.npz"
        metal_path = self.metal_dir / f"{pdb_id}.npz"

        data = np.load(feature_path, allow_pickle=True)
        metal = np.load(metal_path, allow_pickle=True)
        rf_result = np.load(rf_result_path)

        features = Features(**data)
        features.metal_positions = metal["metal_positions"]
        features.metal_types = metal["metal_types"]
        if features.bond_masks.shape != (len(features.atom_elements), len(features.atom_elements)):
        #memory issue: bond mask (l,k) neighbors -> n * n matrix
            bonds_mask = self.neigh_to_bondmask(features)
            features.bond_masks = bonds_mask
            
        grid_probs = rf_result["prob"]
        mask = grid_probs >= 0.5
        features.grid_positions = np.concatenate((features.grid_positions[mask], features.metal_positions), axis=0)

        node_pos = torch.tensor(np.concatenate([features.atom_positions, features.grid_positions]), dtype=torch.float32)
        num_atom = len(features.atom_positions)
        num_nodes = len(node_pos)

        if pdb_id not in self.edge_cache:
            from scipy.spatial import cKDTree
            tree = cKDTree(node_pos.numpy())
            dd, ii = tree.query(node_pos.numpy(), k=16, distance_upper_bound=self.edge_builder.edge_dist_cutoff)
            index_tensor = torch.arange(num_nodes, dtype=torch.int64)
            src = torch.flatten(torch.from_numpy(ii)).to(torch.int64)
            dst = torch.repeat_interleave(index_tensor, 16)
            dists = torch.flatten(torch.from_numpy(dd))
            mask = (src != dst) & (src != num_nodes)
            src = src[mask]
            dst = dst[mask]
            dists = dists[mask]
            e_vec = node_pos[dst] - node_pos[src]
            cos = F.cosine_similarity(e_vec, e_vec, dim=1, eps=1e-6).unsqueeze(-1)
            sin = torch.norm(torch.cross(e_vec, e_vec), dim=1, keepdim=True) + 1e-6
            cov_bond = torch.zeros(len(src))

            self.edge_builder.cache_edges(pdb_id, src, dst, dists, cov_bond, cos, sin, e_vec, num_atom)

        G = self.edge_builder.build_graph_from_cache(pdb_id, num_nodes)
        G.ndata["xyz"] = node_pos

        metal_pos = torch.tensor(features.metal_positions, dtype=torch.float32)
        metal_types = torch.tensor([metals.index(m) for m in features.metal_types], dtype=torch.long)
        grid = torch.tensor(features.grid_positions, dtype=torch.float32)

        diff = grid.unsqueeze(1) - metal_pos.unsqueeze(0)
        dist = torch.sqrt(torch.sum(diff**2, dim=-1)) + 1e-6
        exp_dist = torch.exp(-dist**2 / (4 / np.log(2)))
        label_prob, _ = torch.max(exp_dist, dim=1)
        label_prob[label_prob <= 0.1] = 0.0

        min_dist, min_idx = torch.min(dist, dim=1)
        label_type = torch.where(min_dist <= 2.0, metal_types[min_idx], torch.tensor(len(metals)))
        label_vector = diff[torch.arange(diff.size(0)), min_idx]

        labels = torch.cat([label_prob.unsqueeze(1), label_type.unsqueeze(1), label_vector], dim=1)

        G.ndata["grid_mask"] = torch.cat([torch.zeros(num_atom), torch.ones(len(grid))])
        n_feats, _ = self.get_node_features(features)
        G.ndata["L0"] = n_feats
        return [G], [labels], Info(
            pdb_id=np.array(pdb_id),
            grids_positions=grid,
            metal_positions=metal_pos,
            metal_types=metal_types
        )

