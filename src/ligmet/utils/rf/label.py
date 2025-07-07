import numpy as np

def label_grids(metal_coords: np.ndarray, 
                grid_coords: np.ndarray, 
                t: float = 2.0) -> np.ndarray:
    """
    metal_coords: (M,3) array of metal ion positions  
    grid_coords:  (N,3) array of grid point positions  
    t:            cutoff distance  
    """
    # metal_coords가 비어 있으면 모두 False 반환
    if metal_coords.size == 0:
        return np.zeros(grid_coords.shape[0], dtype=bool)

    # (N,1,3) - (1,M,3) → (N,M,3)
    dists = np.linalg.norm(grid_coords[:, None, :] - metal_coords[None, :, :], axis=-1)
    # 각 grid에 대해 가장 가까운 metal 거리
    min_dists = np.min(dists, axis=-1)
    # cutoff 이하이면 True
    return min_dists <= t
