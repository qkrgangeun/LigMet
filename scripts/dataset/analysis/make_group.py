import pandas as pd

# 원본 CSV 파일 경로
input_path = '/home/qkrgangeun/LigMet/code/text/biolip/metal_binding_sites3.csv'
# 저장할 CSV 파일 경로
output_path = '/home/qkrgangeun/LigMet/code/text/biolip/metal_binding_sites3_grouped.csv'

# CSV 파일 읽기
df = pd.read_csv(input_path)

# 문자열로 저장된 리스트들을 실제 리스트로 변환 및 정규화
df['Binding Residues'] = df['Binding Residues'].apply(eval).apply(lambda x: tuple(sorted(x)))
df['Cluster ID List'] = df['Cluster ID'].apply(eval)

# Cluster ID 내 None 제거하고 tuple로 변환 (grouping용)
def clean_cluster_id(cluster_list):
    if not isinstance(cluster_list, list):
        return tuple()
    cleaned = [c for c in cluster_list if c is not None]
    return tuple(sorted(cleaned))

df['Cluster ID Key'] = df['Cluster ID List'].apply(clean_cluster_id)

# 결측값 제거 (필수 컬럼에 대해서만)
df = df.dropna(subset=['Binding Residues', 'Metal Type'])

# group_id 부여
df['group_id'] = df.groupby(['Binding Residues', 'Metal Type', 'Cluster ID Key']).ngroup().astype(int)

# group_id 기준 정렬
df = df.sort_values(by='group_id').reset_index(drop=True)

# Cluster ID 원래 값 복원
df['Cluster ID'] = df['Cluster ID List']
df.drop(columns=['Cluster ID List', 'Cluster ID Key'], inplace=True)

# 저장
df.to_csv(output_path, index=False)

print(f"None을 무시한 Cluster ID 기준으로 group_id를 부여하고 저장 완료:\n{output_path}")
