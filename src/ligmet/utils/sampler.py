import pandas as pd
from torch.utils.data import Sampler, WeightedRandomSampler
import pandas as pd
from torch.utils.data import Sampler, WeightedRandomSampler
import math

class WeightedSampler(Sampler):
    def __init__(self, dataset, shuffle=True, total_samples=None):
        """
        dataset: PreprocessedDataSet 객체
        shuffle: 샘플링을 무작위로 할지 여부 (현재는 사용되지 않음)
        total_samples: 샘플링할 총 개수 (기본값: 데이터셋 크기)
        """
        self.dataset = dataset
        self.shuffle = shuffle
        self.total_samples = total_samples if total_samples is not None else len(self.dataset)

        self.metal_site_path = '/home/qkrgangeun/LigMet/code/text/biolip/all_metal_binding_sites_NOSSE_3.0.csv'
        self.pdb_ids = self.dataset.pdbid_lists  # index에 대응하는 pdb_id 리스트
        self.pdb_weights = self._compute_weights()

        self.sampler = WeightedRandomSampler(
            weights=self.pdb_weights,
            num_samples=self.total_samples,
            replacement=True
        )

    def _compute_weights(self):
        # metal binding site 데이터 로드
        metal_df = pd.read_csv(self.metal_site_path, converters={
            'Metal Position': eval,
            'Binding Residues': eval,
            'Binding Chains': eval,
            'Cluster ID': eval
        })

        # metal 데이터 중 해당하는 PDB ID만 필터링
        filtered_metal_df = metal_df[metal_df['PDB ID'].isin(self.pdb_ids)]
        metal_counts = filtered_metal_df['Metal Type'].value_counts().to_dict()

        # 각 PDB ID에 대해 포함된 metal type 리스트
        pdb_metal_map = filtered_metal_df.groupby('PDB ID')['Metal Type'].apply(list).to_dict()

        # 가장 희귀한 metal 기준으로 weight 계산
        pdb_to_weight = {}
        for pdb_id, metal_list in pdb_metal_map.items():
            weights = [1 / metal_counts[metal] for metal in metal_list if metal in metal_counts]
            # weights = [1 / math.sqrt(metal_counts[metal]) for metal in metal_list if metal in metal_counts]
            pdb_to_weight[pdb_id] = max(weights) if weights else 0.0

        # Dataset 순서에 맞게 weight 리스트 생성
        weights = [pdb_to_weight.get(pdb_id, 0.0) for pdb_id in self.pdb_ids]

        return weights

    def __iter__(self):
        return iter(self.sampler)

    def __len__(self):
        return self.total_samples


# class MetalSamplerBuilder:
#     def __init__(self, metal_to_pdbs_path: str, pdb_id_to_metals_path: str, total_samples: int = 15000):
#         self.metal_to_pdbs_path = metal_to_pdbs_path
#         self.pdb_id_to_metals_path = pdb_id_to_metals_path
#         self.total_samples = total_samples

#         # Load data
#         self.metal_to_pdbs = self._load_pickle(self.metal_to_pdbs_path)
#         self.pdb_id_to_metals = self._load_pickle(self.pdb_id_to_metals_path)

#         # Initialize internal data
#         self.metal_counts = self._compute_metal_counts()
#         self.scaled_target_count = self._compute_scaled_target_count()
#         self.pdb_ids = list(self.pdb_id_to_metals.keys())
#         self.pdb_weights = self._compute_pdb_weights()

#     def _load_pickle(self, path: str):
#         with open(path, 'rb') as f:
#             return pickle.load(f)

#     def _compute_metal_counts(self):
#         return {metal: len(set(pdbs)) for metal, pdbs in self.metal_to_pdbs.items()}

#     def _compute_scaled_target_count(self):
#         n_max = max(self.metal_counts.values())
#         cube_root_n_max = n_max ** (1/3)

#         metal_sampling_ratio = {
#             metal: (count ** (1/3)) / cube_root_n_max
#             for metal, count in self.metal_counts.items()
#         }

#         total_ratio = sum(metal_sampling_ratio.values())

#         return {
#             metal: round(ratio / total_ratio * self.total_samples)
#             for metal, ratio in metal_sampling_ratio.items()
#         }

#     def _compute_pdb_weights(self):
#         weights = []
#         for pdb_id in self.pdb_ids:
#             metals = self.pdb_id_to_metals[pdb_id]
#             metal_weights = []
#             for metal in metals:
#                 current = self.metal_counts.get(metal, 1)
#                 target = self.scaled_target_count.get(metal, 0)
#                 metal_weights.append(target / current)
#             weight = max(metal_weights) if metal_weights else 0
#             weights.append(weight)
#         return weights

#     def build_sampler(self):
#         return WeightedRandomSampler(
#             weights=self.pdb_weights,
#             num_samples=self.total_samples,
#             replacement=True
#         )

#     def get_pdb_ids(self):
#         return self.pdb_ids

#     def get_weights(self):
#         return self.pdb_weights

#     def get_scaled_target_count(self):
#         return self.scaled_target_count
