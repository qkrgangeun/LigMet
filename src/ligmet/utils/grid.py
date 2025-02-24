import numpy as np
import scipy
import scipy.spatial
from multiprocessing import Pool
def compute_sasa_for_atom(args):
    i, neigh, center, radius, xyz, radii, probe_radius, n_samples, pts0 = args

    neigh.remove(i)
    n_neigh = len(neigh)
    pts = pts0 * (radius + probe_radius) + center
    x_neigh = xyz[neigh][None, :, :].repeat(n_samples, axis=0)
    pts_expand = pts.repeat(n_neigh, 0).reshape(n_samples, n_neigh, 3)
    d2 = np.sum((pts_expand - x_neigh) ** 2, axis=2)
    r2 = (radii[neigh] + probe_radius) ** 2
    r2 = np.stack([r2] * n_samples)
    outsiders = np.all(d2 >= (r2 * 0.99), axis=1)  # 0.99는 수치적 오류를 보정
    return pts[np.where(outsiders)[0]]

def sasa_grids(atom_coords, atom_elems, probe_radius=0.4, n_samples=20, num_processes=4):
    atomic_radii = {
        "C": 2.0,
        "N": 1.5 * 0.8,
        "O": 1.4 * 0.8,
        "S": 1.85 * 0.8,
        "H": 0.0,
        "F": 1.47,
        "Cl": 1.75,
        "Br": 1.85,
        "I": 2.0,
        "P": 1.8,
    }
    centers = atom_coords
    radii = np.array([atomic_radii.get(e,2.0) for e in atom_elems if e in atom_elems])
    inc = np.pi * (3 - np.sqrt(5))  # increment
    off = 2.0 / (n_samples // 2)
    pts0 = []
    for k in range(n_samples // 2):
        phi = k * inc
        y = k * off - 1 + (off / 2)
        r1 = np.sqrt(1 - y * y)
        r2 = 2 * np.sqrt(1 - y * y)
        pts0.append([np.cos(phi) * r1, y, np.sin(phi) * r1])
        pts0.append([np.cos(phi) * r2, y, np.sin(phi) * r2])

    pts0 = np.array(pts0)
    kd = scipy.spatial.cKDTree(atom_coords)
    neighs = kd.query_ball_tree(kd, 8.0)

    # 각 원자에 대한 작업을 병렬로 수행
    with Pool(processes=num_processes) as pool:
        results = pool.map(
            compute_sasa_for_atom,
            [
                (i, neigh, center, radius, atom_coords, radii, probe_radius, n_samples, pts0)
                for i, (neigh, center, radius) in enumerate(zip(neighs, centers, radii))
            ],
        )

    # 결과를 병합
    pts_out = np.concatenate(results)

    return pts_out

def filter_by_clashmap(pts_out, threshold=150000):
    # KD-Tree를 사용하여 충돌하는 점 쌍 찾기
    kd = scipy.spatial.cKDTree(pts_out)
    clash_pairs = kd.query_pairs(r=0.6)

    # 충돌 쌍을 (i, j)와 (j, i)로 확장
    clash_pairs = np.array(list(clash_pairs))
    clash_pairs = np.concatenate([clash_pairs, clash_pairs[:, [1, 0]]])

    # 충돌을 기록할 배열 초기화
    clash_counts = np.zeros(len(pts_out), dtype=int)

    # 각 점의 충돌 횟수 계산
    np.add.at(clash_counts, clash_pairs[:, 0], 1)

    # 충돌이 있는 동안 반복해서 제거
    incl = np.ones(len(pts_out), dtype=bool)
    while np.any(clash_counts > 0):
        # 충돌이 가장 많은 점부터 제거
        max_clash_idx = np.argmax(clash_counts)
        incl[max_clash_idx] = False
        clash_counts[max_clash_idx] = 0

        # 제거된 점과 관련된 충돌을 업데이트
        clash_counts[clash_pairs[clash_pairs[:, 0] == max_clash_idx, 1]] -= 1
        clash_counts[clash_pairs[clash_pairs[:, 1] == max_clash_idx, 0]] -= 1

    grids = pts_out[incl]

    return grids