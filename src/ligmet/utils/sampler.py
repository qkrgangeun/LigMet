import pandas as pd
from torch.utils.data import Sampler, WeightedRandomSampler
import pandas as pd
from torch.utils.data import Sampler, WeightedRandomSampler
import math
import numpy as np
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

    # def _compute_weights(self):
    #     # metal binding site 데이터 로드
    #     metal_df = pd.read_csv(self.metal_site_path, converters={
    #         'Metal Position': eval,
    #         'Binding Residues': eval,
    #         'Binding Chains': eval,
    #         'Cluster ID': eval
    #     })

    #     # metal 데이터 중 해당하는 PDB ID만 필터링
    #     filtered_metal_df = metal_df[metal_df['PDB ID'].isin(self.pdb_ids)]
    #     metal_counts = filtered_metal_df['Metal Type'].value_counts().to_dict()

    #     # 각 PDB ID에 대해 포함된 metal type 리스트
    #     pdb_metal_map = filtered_metal_df.groupby('PDB ID')['Metal Type'].apply(list).to_dict()

    #     # 가장 희귀한 metal 기준으로 weight 계산
    #     pdb_to_weight = {}
    #     for pdb_id, metal_list in pdb_metal_map.items():
    #         weights = [1 / metal_counts[metal] for metal in metal_list if metal in metal_counts]
    #         # weights = [1 / math.sqrt(metal_counts[metal]) for metal in metal_list if metal in metal_counts]
    #         pdb_to_weight[pdb_id] = max(weights) if weights else 0.0

    #     # Dataset 순서에 맞게 weight 리스트 생성
    #     weights = [pdb_to_weight.get(pdb_id, 0.0) for pdb_id in self.pdb_ids]

    #     return weights

#log scale weight
    def _compute_weights(self):
        metal_df = pd.read_csv(self.metal_site_path, converters={
            'Metal Position': eval,
            'Binding Residues': eval,
            'Binding Chains': eval,
            'Cluster ID': eval
        })

        filtered_metal_df = metal_df[metal_df['PDB ID'].isin(self.pdb_ids)]
        metal_counts = filtered_metal_df['Metal Type'].value_counts().to_dict()

        # Step 1: log(count + 1)
        metal_log = {m: np.log1p(c) for m, c in metal_counts.items()}
        log_sum = sum(metal_log.values())

        # Step 2: target ratio and weight
        metal_weights = {
            m: metal_log[m] / (metal_counts[m] * log_sum)
            for m in metal_counts
        }

        # Step 3: map PDB ID to max weight among its metals
        pdb_metal_map = filtered_metal_df.groupby('PDB ID')['Metal Type'].apply(list).to_dict()
        pdb_to_weight = {
            pdb_id: max([metal_weights.get(m, 0.0) for m in metal_list], default=0.0)
            for pdb_id, metal_list in pdb_metal_map.items()
        }

        # Step 4: final weight list
        weights = [pdb_to_weight.get(pdb_id, 0.0) for pdb_id in self.pdb_ids]
        return weights
    
    def __iter__(self):
        return iter(self.sampler)

    def __len__(self):
        return self.total_samples
