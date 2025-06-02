import os
import glob

features_dir = '/home/qkrgangeun/LigMet/data/biolip/dl/features'
train_val_txt = '/home/qkrgangeun/LigMet/code/text/biolip/paper/train_val_filtered.txt'
test_txt = '/home/qkrgangeun/LigMet/code/text/biolip/paper/test_pdbs_filtered.txt'

def get_pdbids_from_file(path):
    with open(path, 'r') as f:
        return set(line.strip() for line in f if line.strip())

def get_all_npz_files(directory):
    files = glob.glob(os.path.join(directory, '*.npz'))
    pdb_to_file = {os.path.splitext(os.path.basename(f))[0]: f for f in files}
    return pdb_to_file

def get_total_size(file_paths):
    total_bytes = sum(os.path.getsize(f) for f in file_paths if os.path.exists(f))
    return total_bytes

def format_size(bytes_size):
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_size < 1024:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024
    return f"{bytes_size:.2f} PB"

def main():
    train_val_ids = get_pdbids_from_file(train_val_txt)
    test_ids = get_pdbids_from_file(test_txt)
    used_ids = train_val_ids | test_ids

    all_npz = get_all_npz_files(features_dir)

    train_val_files = [path for pdbid, path in all_npz.items() if pdbid in train_val_ids]
    test_files = [path for pdbid, path in all_npz.items() if pdbid in test_ids]
    used_files = train_val_files + test_files
    unused_files = [path for pdbid, path in all_npz.items() if pdbid not in used_ids]

    total_size = get_total_size(all_npz.values())
    train_val_size = get_total_size(train_val_files)
    test_size = get_total_size(test_files)
    used_size = train_val_size + test_size
    unused_size = get_total_size(unused_files)

    print("=== NPZ 파일 용량 통계 ===")
    print(f"전체 파일 수        : {len(all_npz)}")
    print(f"Train/Val 파일 수    : {len(train_val_files)}")
    print(f"Test 파일 수         : {len(test_files)}")
    print(f"사용되지 않은 파일 수 : {len(unused_files)}")
    print()
    print(f"전체 용량          : {format_size(total_size)}")
    print(f"Train/Val 용량      : {format_size(train_val_size)}")
    print(f"Test 용량           : {format_size(test_size)}")
    print(f"사용된 총 용량      : {format_size(used_size)}")
    print(f"사용되지 않은 용량   : {format_size(unused_size)}")

if __name__ == "__main__":
    main()
