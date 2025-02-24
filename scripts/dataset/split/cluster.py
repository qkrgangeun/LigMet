import os
from pathlib import Path
from multiprocessing import Pool, cpu_count
import pandas as pd

def extract_release_date_from_pdb(pdb_id, pdb_dir):
    pdb_file = os.path.join(pdb_dir, f"{pdb_id}.pdb")
    if not os.path.exists(pdb_file):
        return "Unknown"
    
    try:
        with open(pdb_file, "r") as f:
            for line in f:
                if line.startswith("REVDAT   1"):
                    date_parts = line[10:23].strip().split("-")
                    if len(date_parts) == 3:
                        day, month, year = date_parts
                        month_dict = {
                            "JAN": "01", "FEB": "02", "MAR": "03", "APR": "04", "MAY": "05", "JUN": "06",
                            "JUL": "07", "AUG": "08", "SEP": "09", "OCT": "10", "NOV": "11", "DEC": "12"
                        }
                        formatted_year = "19" + year if int(year) >= 25 else "20" + year  # 00-49 -> 2000s, 50-99 -> 1900s
                        return formatted_year + month_dict.get(month.upper(), "00") + day.zfill(2)
        return "Unknown"
    except Exception as e:
        return "Unknown"

def process_fasta(fasta_file, fasta_dir, cluster_df, pdb_dir):
    pdb_id = fasta_file.replace(".fasta", "")
    fasta_path = os.path.join(fasta_dir, fasta_file)
    
    with open(fasta_path, "r") as f:
        lines = f.readlines()
        chains = [line.split(":")[-1].strip() for line in lines if line.startswith(">")]
    
    cluster_data = cluster_df[cluster_df["chain"].str.startswith(pdb_id)]
    clusters = cluster_data["cluster"].tolist()
    
    release_date = extract_release_date_from_pdb(pdb_id, pdb_dir)
    
    return [pdb_id, chains, clusters, release_date]

def extract_fasta_info(fasta_dir, cluster_file, pdb_dir):
    fasta_files = [f for f in os.listdir(fasta_dir) if f.endswith(".fasta")]
    # fasta_files = ['8a8d.fasta']
    cluster_df = pd.read_csv(cluster_file, sep="\t")
    
    with Pool(processes=cpu_count()) as pool:
        pdb_info_list = pool.starmap(process_fasta, [(f, fasta_dir, cluster_df, pdb_dir) for f in fasta_files])
    
    return pdb_info_list

def main():
    fasta_dir = Path("/home/qkrgangeun/LigMet/data/fasta")
    cluster_file = "/home/qkrgangeun/LigMet/code/scripts/dataset/split/cluster.tsv"
    pdb_dir = "/home/qkrgangeun/LigMet/data/pdb"
    output_csv = "/home/qkrgangeun/LigMet/code/scripts/dataset/split/pdb_info.csv"
    
    pdb_info = extract_fasta_info(fasta_dir, cluster_file, pdb_dir)
    
    # DataFrame으로 변환 후 저장
    df = pd.DataFrame(pdb_info, columns=["pdb_id", "chains", "clusters", "release_date"])
    df.to_csv(output_csv, index=False)
    
    print(f"CSV 파일 저장 완료: {output_csv}")

if __name__ == "__main__":
    main()