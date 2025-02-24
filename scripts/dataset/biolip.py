import os
from pathlib import Path
from multiprocessing import Pool, cpu_count

def process_pdb(pdb):
    metals = {"MG", "ZN", "MN", "CA", "FE", "NI", "CO", "CU", "K", "NA"}
    
    if not pdb.endswith(".pdb"):  # 잘못된 파일 필터링
        return
    
    pdb_id = pdb.replace(".pdb", "")
    
    biolip_path = biolip_dir / f"{pdb_id}_merged.pdb"
    pdb_path = pdb_dir / f"{pdb_id}.pdb"
    processed_path = processed_dir / f"{pdb_id}.pdb"
    
    if not pdb_path.exists():  # 다운로드 실패 방지
        print(f"Warning: {pdb_id}.pdb not found in {pdb_dir}")
        return
    
    try:
        with open(biolip_path, 'r') as b, open(pdb_path, 'r') as p, open(processed_path, 'w') as file:
            biolip_text = {line[30:81].strip() for line in b if line.startswith(('HETATM','ATOM'))}
            
            for line in p:
                if line.startswith('ATOM'):
                    if line[30:81].strip() in biolip_text:
                        file.write(line)
                elif line.startswith('HETATM'):
                    atom_elem = line[76:78].strip()  # 원소명 위치 수정
                    res_name = line[17:20].strip()  # 리간드명 위치 수정
                    
                    if atom_elem in metals:
                        if line[30:81].strip() in biolip_text:
                            file.write(line)
                    elif res_name != 'HOH':
                        file.write(line)
        print(f"Processed: {processed_path}")
    except Exception as e:
        print(f"Error processing {pdb_id}: {e}")

if __name__ == "__main__":
    biolip_dir = Path("/home/qkrgangeun/LigMet/data/merge")
    pdb_dir = Path("/home/qkrgangeun/LigMet/data/pdb")
    processed_dir = Path("/home/qkrgangeun/LigMet/data/processed")
    
    os.makedirs(processed_dir, exist_ok=True)
    
    pdb_list = [pdb for pdb in os.listdir(pdb_dir) if pdb.endswith(".pdb")]
    
    with Pool(processes=cpu_count()) as pool:
        pool.map(process_pdb, pdb_list)