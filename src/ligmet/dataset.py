# ── Standard library
import argparse
import math
from dataclasses import asdict
from pathlib import Path
from typing import Tuple, Union

# ── Third-party
import numpy as np
import pandas as pd
import torch  # type: ignore
import torch.nn as nn
import torch.nn.functional as F  # type: ignore
import dgl  # type: ignore
from joblib import load
from scipy.spatial import cKDTree  # type: ignore
from tqdm import tqdm

# ── Local packages (ligmet)
from ligmet.featurizer import Features, Info, make_features  # type: ignore
from ligmet.utils.constants import (  # type: ignore
    ATOMIC_NUMBERS,
    aliphatic_carbons,
    aromatic_carbons,
    atype2num,
    metals,
    sec_struct_dict,
    standard_residues,
    sybyl_type_dict,
)
from ligmet.utils.grid import filter_by_clashmap, sasa_grids_thread
from ligmet.utils.pdb import StructureWithGrid, read_pdb  # type: ignore
from ligmet.utils.rf.label import label_grids  # type: ignore
from ligmet.utils.rf.rf_features import (  # type: ignore
    RSA,
    binned_res,
    filter_by_biometall,
    near_lig,
    near_res,
    nearest_bb_dist,
    nearest_prot_carbon_dist,
    parse_pdb,
)

class OnTheFlyDataSet(torch.utils.data.Dataset):
    def __init__(self, data_file: str, pdb_dir:str, rf_model_path:str, topk: int, edge_dist_cutoff: float, pocket_dist: float, rf_threshold: float, eps=1e-6):
        super().__init__()
        self.data_file=Path(data_file)
        self.pdb_dir = Path(pdb_dir)
        self.rf_model = load(rf_model_path)
        self.topk = topk
        self.edge_dist_cutoff=edge_dist_cutoff
        self.pocket_dist=pocket_dist
        self.rf_threshold=rf_threshold
        self.pdbid_lists=[pdb.strip().split(".pdb")[0] for pdb in open(data_file)]
        self.eps = eps
        self.alpha = 4/math.log(2) #5.77078
        self.relpos_embedding = nn.Embedding(65, 8)  # relative position embedding, dim=8

    def __len__(self):
        return len(self.pdbid_lists)
    
    def __getitem__(self, index:int):
        G = []
        L = []
        
        pdb_id = self.pdbid_lists[index]
        pdb_path = self.pdb_dir / f"{pdb_id}.pdb"
        dl_feat = self._make_dl_feature(pdb_path)
        rf_feat = self._make_rf_feature(dl_feat)
        rf_result = self._test_rf(rf_feat, self.rf_model, label_column='label_2.0')
            
        grid_positions = dl_feat.grid_positions
        grid_probs = rf_result["prob"] 

        grid_mask = grid_probs >= self.rf_threshold
        grids_after_rf = grid_positions[grid_mask]
        if dl_feat.metal_positions is not None:
            grids_after_rf = np.concatenate((grids_after_rf,dl_feat.metal_positions),axis=0)
        dl_feat.grid_positions = grids_after_rf

        g = self.make_graph(dl_feat)
        l_prob, l_type, l_vector = self.make_label(dl_feat)
        labels = torch.cat([l_prob.unsqueeze(1), l_type.unsqueeze(1), l_vector], dim=1)  # shape [N,5]
        G.append(g)
        L.append(labels)
        if not G:
            raise AttributeError(f"{pdb_id} have none type graph")
        info = Info(
            pdb_id=np.array(pdb_id),
            grids_positions=torch.tensor(grids_after_rf, dtype=torch.float32),
            metal_positions=(
            torch.tensor(dl_feat.metal_positions, dtype=torch.float32)
            if dl_feat.metal_positions is not None and len(dl_feat.metal_positions) > 0
            else None
            ),
            metal_types=(
            torch.tensor([metals.index(metal) for metal in dl_feat.metal_types])
            if dl_feat.metal_types is not None and len(dl_feat.metal_types) > 0
            else None
            ),
        )

        return G, L, info
    
    def _make_dl_feature(self, pdb_path:Path) -> Features:
        structure = read_pdb(pdb_path)
        if len(structure.atom_positions) > 50000:
            print("skip more than 50000")
            return
        else:
            grids = sasa_grids_thread(structure.atom_positions, structure.atom_elements)
            grids = filter_by_clashmap(grids)

            structure_dict = asdict(structure)
            structure_with_grid = StructureWithGrid(
                grid_positions=grids,
                **structure_dict
            )

            features = make_features(pdb_path, structure_with_grid)

            return features
    def _make_rf_feature(self, dl_features:Features) -> dict:
        protein_mask = dl_features.is_ligand == 0
        ligand_mask = dl_features.is_ligand == 1
        dists = [2.5, 2.8, 3.0, 3.2, 5.0]
        df = pd.DataFrame()
        
        for threshold in dists:
            p_coords_res, p_core_res, p_bb_coords = near_res(
                dl_features.atom_residues[protein_mask],
                dl_features.atom_names[protein_mask],
                dl_features.atom_elements[protein_mask],
                dl_features.atom_positions[protein_mask],
                dl_features.grid_positions,
                threshold
            )
            
            n_lig_NOS, n_lig_nion, n_lig_etc = near_lig(
                dl_features.atom_positions[ligand_mask],
                dl_features.atom_elements[ligand_mask],
                dl_features.atom_residues[ligand_mask],
                dl_features.grid_positions,
                threshold
            )
            
            df[f'p_coords_res_{threshold}'] = np.array(p_coords_res).astype(np.int8)
            df[f'p_core_res_{threshold}'] = np.array(p_core_res).astype(np.int8)
            df[f'p_bb_coords_{threshold}'] = np.array(p_bb_coords).astype(np.int8)
            df[f'n_lig_NOS_{threshold}'] = np.array(n_lig_NOS).astype(np.int8)
            df[f'n_lig_nion_{threshold}'] = np.array(n_lig_nion).astype(np.int8)
            df[f'n_lig_etc_{threshold}'] = np.array(n_lig_etc).astype(np.int8)
        
        n_coords_res_bin, n_core_res_bin, n_bb_coords_bin = binned_res(
            dl_features.atom_residues,
            dl_features.atom_names,
            dl_features.atom_elements,
            dl_features.atom_positions,
            dl_features.grid_positions,
            3,
            5
        )
        
        df['n_coords_res_bin'] = np.array(n_coords_res_bin).astype(np.int8)
        df['n_core_res_bin'] = np.array(n_core_res_bin).astype(np.int8)
        df['n_bb_coords_bin'] = np.array(n_bb_coords_bin).astype(np.int8)
        
        min_dist = nearest_prot_carbon_dist(
            dl_features.atom_residues,
            dl_features.atom_names,
            dl_features.atom_elements,
            dl_features.atom_positions,
            dl_features.grid_positions,
            aliphatic_carbons,
            aromatic_carbons
        )
        df['min_c_dist'] = np.array(min_dist).astype(np.float16)
        
        near_bb_dist_values = nearest_bb_dist(
            dl_features.atom_names[protein_mask],
            dl_features.atom_positions[protein_mask],
            dl_features.grid_positions
        )
        df['near_bb_dist'] = np.array(near_bb_dist_values).astype(np.float16)
        
        sasa = RSA(dl_features.grid_positions, dl_features.atom_positions, dl_features.atom_elements)
        df['sasa'] = np.array(sasa).astype(np.float16)
        
        atom_dict, grids = parse_pdb(dl_features)
        num_res_list = filter_by_biometall(grids, atom_dict)
        df['biometall'] = np.array(num_res_list).astype(np.int8)
        
        # #### Label ####
        labels = label_grids(dl_features.metal_positions, dl_features.grid_positions, 2.0)
        df['label_2.0'] = labels.astype('bool')
        
        return df
    

    def _test_rf(self, rf_feat_df:pd.DataFrame, rf_model, label_column='label_2.0') -> dict:
        X = rf_feat_df.drop(columns=[label_column]).values
        y = rf_feat_df[label_column].values
        y_prob = rf_model.predict_proba(X)[:, 1]
        return {"prob": y_prob}
    
    def neigh_to_bondmask(self, features:Features):
        # bond_mask : [[row], [col]]
        n_atom = len(features.atom_names)
        cov_bonds_mask = np.zeros((n_atom,n_atom))
        neigh = features.bond_masks
        cov_bonds_mask[neigh[0],neigh[1]] = 1
        cov_bonds_mask[neigh[1],neigh[0]] = 1
        return cov_bonds_mask
    
    
    def make_label(self, features: Features) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # grids: [G, 3]
        grid = np.array(features.grid_positions, dtype=np.float32)
        grids = torch.from_numpy(grid)  # CPU float32
        G = grids.shape[0]
        device = grids.device
        background_class = len(metals)  # 마지막 인덱스를 배경으로 사용

        # metal이 아예 없거나 길이가 0인 경우를 모두 처리
        no_metal = (
            features.metal_positions is None
            or (isinstance(features.metal_positions, np.ndarray) and features.metal_positions.size == 0)
        )

        if no_metal:
            label_prob   = torch.zeros(G, dtype=torch.float32, device=device)
            label_type   = torch.full((G,), background_class, dtype=torch.long, device=device)
            label_vector = torch.zeros(G, 3, dtype=torch.float32, device=device)
            return label_prob, label_type, label_vector

        # metal_pos: [M, 3], metal_types: [M]
        metal_pos_np = np.asarray(features.metal_positions, dtype=np.float32)
        M = metal_pos_np.shape[0]
        if M == 0:
            # 방어적: 혹시 위 조건을 통과했더라도 M==0이면 동일 처리
            label_prob   = torch.zeros(G, dtype=torch.float32, device=device)
            label_type   = torch.full((G,), background_class, dtype=torch.long, device=device)
            label_vector = torch.zeros(G, 3, dtype=torch.float32, device=device)
            return label_prob, label_type, label_vector

        metal_pos = torch.from_numpy(metal_pos_np).to(device=device)  # float32
        # metals: List[str] 라고 가정
        metal_types_idx = torch.tensor([metals.index(m) for m in features.metal_types],
                                    dtype=torch.long, device=device)  # [M]

        # diff/dist 계산
        diff = grids.unsqueeze(1) - metal_pos.unsqueeze(0)        # [G, M, 3]
        # self.eps는 아주 작은 양수여야 함
        dist = torch.sqrt(torch.sum(diff * diff, dim=-1)).clamp_min(0) + self.eps  # [G, M]

        # label_prob: exp(-d^2/alpha) 최대값
        # self.alpha > 0 가정
        exp_dist = torch.exp(-(dist * dist) / self.alpha)         # [G, M]
        label_p, _ = exp_dist.max(dim=-1)                         # [G]
        # 임계값 0.1 이하면 0으로
        label_prob = torch.where(label_p <= 0.1,
                                torch.zeros_like(label_p),
                                label_p)

        # type/vector: 최단 거리 metal 기준
        min_dist, min_idx = dist.min(dim=-1)                      # [G], [G]

        # 초기값(배경)
        label_type = torch.full((G,), background_class, dtype=torch.long, device=device)
        label_vector = torch.zeros(G, 3, dtype=torch.float32, device=device)

        # 유효 그리드만 갱신 (예: 2.0 Å 이내)
        thr = 2.0
        valid = min_dist <= thr
        if valid.any():
            v_idx = min_idx[valid]                                 # [G_valid]
            label_type[valid] = metal_types_idx[v_idx]             # 타입
            label_vector[valid] = diff[valid, v_idx, :]            # 방향 벡터 (정규화 원하면 아래 주석 해제)

            # 정규화가 필요하다면:
            # vec = diff[valid, v_idx, :]
            # vec = vec / (vec.norm(dim=-1, keepdim=True) + 1e-8)
            # label_vector[valid] = vec

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

        # 모든 feature 합치기
        # n_feats = torch.cat([aatype, atomtype, atom_chemtype, sec_structs, sasas, qs, node_type], dim=1)
        n_feats = torch.cat([aatype, atomtype, atom_chemtype, sec_structs, qs, node_type], dim=1)
        n_feats = torch.nan_to_num(n_feats, nan=0.0, posinf=0.0, neginf=0.0)

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
        rel_emb = self.relpos_embedding(rel_idx).detach()  # [E, 8]

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
                if info.metal_positions is not None and info.metal_positions.numel() > 0:
                    m_pos.append(info.metal_positions)
                else:
                    m_pos.append(torch.zeros((1, 3), dtype=torch.float32))
                if info.metal_types is not None and len(info.metal_types) > 0:
                    m_types.append(info.metal_types)
                else:
                    m_types.append(torch.full((1,), -1, dtype=torch.long))
                pdb_ids.append(info.pdb_id)
        # 배치 그래프와 배치 라벨 생성
        batched_graphs = dgl.batch(graphs)  # shape [B*N]
        batched_labels = torch.cat(labels, dim=0)  # shape [B*N,2]
        g_poss = torch.cat(g_pos, dim=0)
        print(m_pos, m_types)
        m_poss = torch.cat(m_pos, dim=0) 
        m_typess = torch.cat(m_types, dim=0) 
        pdb_idss = np.array(pdb_ids)
        batched_infos = Info(
            pdb_id=pdb_idss,
            grids_positions=g_poss.detach(),
            metal_positions=m_poss.detach(),
            metal_types=m_typess.detach(),
        )
        return batched_graphs, batched_labels, batched_infos