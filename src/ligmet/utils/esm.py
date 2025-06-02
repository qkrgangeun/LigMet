import torch
import esm
from Bio.PDB import PDBParser

# GPU 사용 가능 여부 체크
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ESM-3 모델 불러오기
model, alphabet = esm.pretrained.esmfold_v1()
model = model.eval().to(device)

# PDB 파일 읽기
def read_pdb_sequence(pdb_file, chain_id='A'):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_file)
    sequence = ''
    for model in structure:
        for chain in model:
            if chain.id == chain_id:
                for residue in chain:
                    if residue.get_resname() in esm.data.ALPHABET_PROTEIN_THREE_LETTER:
                        sequence += esm.data.ALPHABET_PROTEIN_THREE_LETTER[residue.get_resname()]
    return sequence

# 단백질 서열 추출
pdb_file_path = '/home/qkrgangeun/LigMet/data/biolip/pdb/2c9p.pdb'
chain_id = 'A'
sequence = read_pdb_sequence(pdb_file_path, chain_id)

print(f"Extracted sequence: {sequence}")

# 서열을 모델 입력 포맷으로 변환
batch_converter = alphabet.get_batch_converter()
data = [("protein", sequence)]
batch_labels, batch_strs, batch_tokens = batch_converter(data)
batch_tokens = batch_tokens.to(device)

# ESM-3로 서열을 임베딩하여 특성 추출
with torch.no_grad():
    results = model(batch_tokens)
    representations = results['s_s']  # 잔기별 특성
    mean_representation = representations.mean(1)  # 단백질 수준의 특성

print("Residue-level representations shape:", representations.shape)
print("Protein-level representation shape:", mean_representation.shape)
