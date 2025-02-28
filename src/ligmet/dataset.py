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
        self.pdbid_lists=[pdb.strip().split(".pdb")[0] for pdb in open(data_file)]
        self.eps = eps
        self.alpha = 4/math.log(2) #5.77078
        
    def __len__(self):
        return len(self.pdbid_lists)
    
    def __getitem__(self, index:int):
        G = []
        L = []
        pdb_id = self.pdbid_lists[index]
        feature_path = self.features_dir / f"{pdb_id}.npz"
        rf_result_path = self.rf_result_dir / f"{pdb_id}.npz"
        data = np.load(feature_path,allow_pickle=True)
        features = Features(**data)
        grid_positions = features.grid_positions
        grid_probs = np.load(rf_result_path)
        grid_mask = grid_probs >= self.rf_threshold
        grids_after_rf = grid_positions[grid_mask]
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

    def find_pocket(self, features: Features, grids: np.ndarray):
        c_grids = grids
        atom_pos = features.atom_positions
        size = len(features.metal_positions)
        random_int = np.random.choice(len(grids), size=size, replace=False)
        combined_positions = np.vstack([c_grids[random_int], features.metal_positions])
        rtree = cKDTree(combined_positions)
        mtree = cKDTree(features.metal_positions)
        gtree = cKDTree(c_grids)
        ptree = cKDTree(atom_pos)
        all_positions = np.vstack([features.atom_positions,grids])
        tree = cKDTree(all_positions)
        # ii = gtree.query_ball_tree(ptree, self.pocket_dist)
        ii = rtree.query_ball_tree(ptree, self.pocket_dist)
        jj = rtree.query_ball_tree(gtree, self.pocket_dist)
        idx = np.unique(np.concatenate([i for i in ii if i], axis=0)).astype(int)
        jdx = np.unique(np.concatenate([j for j in jj if j], axis=0)).astype(int)
        if len(idx) == 0:
            return None, False

        c_features = Features(
            atom_positions=atom_pos[idx],
            atom_names=features.atom_names[idx],
            atom_elements=features.atom_elements[idx],
            atom_residues=features.atom_residues[idx],
            residue_idxs=features.residue_idxs[idx],
            chain_ids=features.chain_ids[idx],
            is_ligand=features.is_ligand[idx],
            metal_positions=features.metal_positions, 
            metal_types=features.metal_types,
            grid_positions=c_grids[jdx],
            sasas=features.sasas[idx],
            qs=features.qs[idx],
            sec_structs=features.sec_structs[idx],
            gen_types=features.gen_types[idx],
            bond_masks=features.bond_masks[np.ix_(idx, idx)], 
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
        label_prob = torch.where(label_p <= 0.1, torch.tensor(0.0), label_p)

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
        print('graph node개수:',len(xyz))
        print('graph protein node개수:', len(features.atom_positions))
        print('graph grid node개수:',len(features.grid_positions))
        print('graph edge개수:',len(e_feats))
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
        # num_res = len(features.atom_names)
        num_grids = len(features.grid_positions)

        sasas = torch.from_numpy(features.sasas)
        qs = torch.from_numpy(features.qs)
        sec_structs = torch.from_numpy(features.sec_structs)
        atom_gentype = torch.from_numpy(features.gen_types)
        
        # one hot features: aatype, atomtype, 2nd structures
        # assign max int for grids
        aatype = torch.Tensor([standard_residues.index(res) if res in standard_residues else len(standard_residues) for res in features.atom_residues ])
        grids_aatype = torch.ones(num_grids) * len(standard_residues)+1
        aatype = torch.cat((aatype, grids_aatype))

        atomtype = torch.Tensor([ATOMIC_NUMBERS.get(elem,119) for elem in features.atom_elements])
        grids_atomtype = torch.zeros(num_grids)
        atomtype = torch.cat([atomtype, grids_atomtype], dim=0)
        
        ##TODO: ligand gentype
        grids_atomchemtype = torch.ones(num_grids) *(max(sybyl_type_dict.values())+1)
        atom_chem_type = torch.cat([atom_gentype, grids_atomchemtype], dim=0)
        
        grids_2nd = torch.ones(num_grids) * len(sec_struct_dict)
        sec_structs = torch.cat([sec_structs, grids_2nd])
        # one-hot encoding
        aatype = F.one_hot(aatype.to(torch.int64), num_classes=len(standard_residues) + 2)
        atomtype = F.one_hot(atomtype.to(torch.int64), num_classes=len(ATOMIC_NUMBERS) + 2)
        sec_structs = F.one_hot(
            sec_structs.to(torch.int64), num_classes=len(sec_struct_dict) + 1
        )
        atom_chemtype = F.one_hot(
            atom_chem_type.to(torch.int64), num_classes=max(sybyl_type_dict.values())+2
        )
        # real value features: sasas, qs
        # assign 0 for grids
        grids_feat = torch.zeros(num_grids)
        sasas = torch.cat((sasas, grids_feat)).unsqueeze(-1)
        qs = torch.cat((qs, grids_feat)).unsqueeze(-1)
        sasas = sasas + self.eps

        n_feats = torch.cat(
            [aatype, atomtype, atom_chemtype, sec_structs, sasas, qs], dim=1
        )
        print(f"NaN in sasas: {torch.isnan(sasas).sum().item()}")
        print(f"NaN in qs: {torch.isnan(qs).sum().item()}")
        print(f"NaN in features.qs: {torch.isnan(torch.tensor(features.qs)).sum().item()}")
        print(f"NaN in sec_structs: {torch.isnan(sec_structs).sum().item()}")
        print(f"NaN in atom_gentype: {torch.isnan(atom_chemtype).sum().item()}")
        print(f"NaN in aatype: {torch.isnan(aatype).sum().item()}")
        print(f"NaN in atomtype: {torch.isnan(atomtype).sum().item()}")
        polarity_vectors = self.make_polarity_vector(features)
        polarity_vectors = torch.tensor(polarity_vectors)
        return n_feats, polarity_vectors

    def onehot_edge_dist(self, dists: torch.Tensor) -> torch.Tensor:
        bin_edges = np.arange(0, self.edge_dist_cutoff + 0.5, 0.5)
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

        edge_index_src = edge_index_src[edge_mask]
        edge_index_dst = edge_index_dst[edge_mask]
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
        e_feats = torch.cat([onehot_type, dist_bin, covalent_bond, cos, sin], dim=1)

        return edge_index_src, edge_index_dst, e_feats, e_vec
    
    @staticmethod
    def collate(samples: list) -> Tuple[dgl.DGLGraph, torch.Tensor, Info]:
        graphs, labels, g_pos, m_pos, m_types, pdb_ids = [], [], [], [], [], []

        for G, L, info in samples:
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
        grid_positions = features.grid_positions
        print('원래 protein개수:', len(structure.atom_positions))
        print('원래 grid개수:',len(grids))
        
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
    
def get_dataset_class(config):
    dataset_type = config["dataset"]["type"]
    
    if dataset_type == "preprocessed":
        return PreprocessedDataSet(**config["dataset"]["preprocessed"])
    elif dataset_type == "on_the_fly":
        return OnTheFlyDataSet(**config["dataset"]["onthefly"])
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")