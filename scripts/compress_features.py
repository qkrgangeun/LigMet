import os
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

input_dir = "/home/qkrgangeun/LigMet/data/biolip/dl/features"
output_dir = "/home/qkrgangeun/LigMet/data/biolip/dl/features"
error_log_path = "error_log.txt"

os.makedirs(output_dir, exist_ok=True)

def bondmask_to_neighidx(bond_mask: np.ndarray):
    rows, cols = np.where(np.triu(bond_mask) > 0)
    return np.stack([rows, cols], axis=0).astype(np.int32)

def compress_memory_worker(fname):
    input_path = os.path.join(input_dir, fname)
    output_path = os.path.join(output_dir, fname)

    try:
        data = np.load(input_path)
        compressed = {}

        for key in data:
            arr = data[key]

            if key == "bond_masks":
                compressed[key] = bondmask_to_neighidx(arr)
            elif key in ["atom_positions", "metal_positions", "grid_positions", "sasas", "qs"]:
                compressed[key] = arr.astype(np.float32)
            elif key in ["residue_idxs"]:
                compressed[key] = arr.astype(np.int32)
            elif key in ["sec_structs", "gen_types"]:
                compressed[key] = arr.astype(np.int16)
            elif key == "is_ligand":
                compressed[key] = arr.astype(np.bool_)
            elif arr.dtype.kind == "U":
                maxlen = max(len(str(s)) for s in arr)
                compressed[key] = arr.astype(f"<U{maxlen}")
            else:
                compressed[key] = arr

        np.savez(output_path, **compressed)
        return None  # 성공 시 None 반환
    except Exception as e:
        print(f"[ERROR] {fname}: {e}")
        return fname  # 실패한 파일명 반환

def main():
    npz_files = [f for f in os.listdir(input_dir) if f.endswith(".npz")]

    error_files = []
    with Pool(cpu_count()) as pool:
        for result in tqdm(pool.imap_unordered(compress_memory_worker, npz_files), total=len(npz_files), desc="Compressing .npz files"):
            if result is not None:
                error_files.append(result)

    # 실패한 파일 로그 저장
    if error_files:
        with open(error_log_path, "w") as f:
            for fname in error_files:
                f.write(f"{fname}\n")
        print(f"\n❌ {len(error_files)} errors logged to: {error_log_path}")
    else:
        print("\n✅ All files compressed successfully.")

if __name__ == "__main__":
    main()
