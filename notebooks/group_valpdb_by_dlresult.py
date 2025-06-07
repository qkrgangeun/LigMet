import re

# 로그 파일 경로
log_path = "/home/qkrgangeun/LigMet/benchmark/test_chain1_pre3.log"

# 그룹 분류용 딕셔너리 초기화
group_precision_recall = {
    "A": [],  # recall > 0.7 and precision > 0.5
    "B": [],  # recall > 0.7 and precision <= 0.5
    "C": [],  # recall <= 0.7 and precision > 0.5
    "D": [],  # recall <= 0.7 and precision <= 0.5
}

group_type_accuracy = {
    "HIGH": [],  # type_accuracy > 0.5
    "LOW": []    # type_accuracy <= 0.5
}

with open(log_path, "r") as f:
    lines = f.readlines()

pdb_id = None
for i, line in enumerate(lines):
    # PDB ID 찾기
    if line.startswith("=== PDB:"):
        pdb_id_match = re.search(r"\['(.+?)'\]", line)
        if pdb_id_match:
            pdb_id = pdb_id_match.group(1)

    # precision & recall 값 찾기
    if pdb_id and "threshold 0.5 | precision:" in line:
        pr_match = re.search(r"precision: ([0-9.]+) \| recall: ([0-9.]+)", line)
        if pr_match:
            precision = float(pr_match.group(1))
            recall = float(pr_match.group(2))
            # 그룹 분류
            if recall > 0.7:
                if precision > 0.5:
                    group_precision_recall["A"].append(pdb_id)
                else:
                    group_precision_recall["B"].append(pdb_id)
            else:
                if precision > 0.5:
                    group_precision_recall["C"].append(pdb_id)
                else:
                    group_precision_recall["D"].append(pdb_id)

    # type_accuracy 값 찾기
    if pdb_id and "threshold 0.5 | type_accuracy:" in line:
        acc_match = re.search(r"type_accuracy: ([0-9.]+)", line)
        if acc_match:
            type_acc = float(acc_match.group(1))
            if type_acc > 0.5:
                group_type_accuracy["HIGH"].append(pdb_id)
            else:
                group_type_accuracy["LOW"].append(pdb_id)

# 결과 출력
print("=== Precision/Recall 그룹 ===")
for group, ids in group_precision_recall.items():
    print(f"Group {group}: {ids}")

print("\n=== Type Accuracy 그룹 ===")
for group, ids in group_type_accuracy.items():
    print(f"{group}: {ids}")

