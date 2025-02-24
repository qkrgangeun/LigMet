import pandas as pd
import numpy as np

# 메탈 좌표와 그리드 좌표 간의 거리 계산 및 레이블링 함수
def label_grids(metal_coords, grid_coords, t =2.0):
    dists = np.linalg.norm(grid_coords[:,None,:] - metal_coords[None,:,:], axis=-1)
    dists = np.min(dists, axis=-1)
    labels = dists <= t
    return labels
