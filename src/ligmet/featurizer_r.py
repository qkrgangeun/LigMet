from dataclasses import dataclass
import numpy as np
from Bio.PDB import PDBParser, DSSP
from typing import Optional,Union
import freesasa
from collections import defaultdict
import tempfile
from pathlib import Path
import re
import torch
from ligmet.utils.pdb import StructureWithGrid, read_pdb
import io
from openbabel import pybel
from ligmet.utils.constants import (
    partial_q,
    sec_struct_dict,
    bonds,
    standard_residues,
    atype2num,
    atypes,
    sybyl_type_dict,
)
from rdkit import Chem
from rdkit.Chem import AllChem

@dataclass
class Features():
    atom_positions: np.ndarray  #(n_atoms,3)
    atom_names: np.ndarray  #(n_atoms,1)
    atom_elements: np.ndarray   #(n_atoms,1)
    atom_residues: np.ndarray   #(n_atoms,1)
    residue_idxs: np.ndarray    #(n_atoms,1)
    chain_ids: np.ndarray   #(n_atoms,1)
    is_ligand: np.ndarray   #(n_atoms,1)
    metal_positions: np.ndarray     #(n_atoms,1)
    metal_types: np.ndarray     #(n_atoms,1)
    grid_positions:np.ndarray   #(n_atoms,1)
    sasas: np.ndarray   #(n_atoms,1)
    qs: np.ndarray  #(n_atoms,1)
    sec_structs: np.ndarray     #(n_atoms,1)
    gen_types: np.ndarray
    bond_masks: np.ndarray  #(n_atoms,n_atoms)

@dataclass
class Info():
    pdb_id: np.array
    grids_positions: torch.Tensor  # [g, 3]
    metal_positions: torch.Tensor # [m, 3]
    metal_types: torch.Tensor # [m]

# def make_pdb(protein: StructureWithGrid) -> tuple[io.StringIO, io.StringIO, io.StringIO]:
#     pdb_io = io.StringIO()
#     protein_io = io.StringIO()
#     ligand_io = io.StringIO()

#     for idx, (chain_id, res_idx, res_name, atom_name, atom_pos, atom_elem, is_lig) in enumerate(zip(
#             protein.chain_ids, protein.residue_idxs, protein.atom_residues, protein.atom_names, 
#             protein.atom_positions, protein.atom_elements, protein.is_ligand)):
        
#         # 🔹 res_idx에서 숫자 부분만 추출 (Insertion Code 제거)
#         res_idx_clean = int(''.join(filter(str.isdigit, str(res_idx)))) if any(c.isdigit() for c in str(res_idx)) else res_idx
    
#         pdb_line = f"{'HETATM' if is_lig else 'ATOM  '}{idx+1:5d}  {atom_name:<3} {res_name:>3} {chain_id:>1}{res_idx_clean:>4d}    {atom_pos[0]:8.3f}{atom_pos[1]:8.3f}{atom_pos[2]:8.3f}  1.00  0.00          {atom_elem:>2}\n"
#         pdb_io.write(pdb_line)

#         if is_lig:
#             ligand_io.write(pdb_line)
#         else:
#             protein_io.write(pdb_line)

#     pdb_io.seek(0)
#     protein_io.seek(0)
#     ligand_io.seek(0)
#     return pdb_io, protein_io, ligand_io

# def make_pdb(protein: StructureWithGrid) -> tuple[io.StringIO, io.StringIO, io.StringIO]:
#     pdb_io = io.StringIO()
#     protein_io = io.StringIO()
#     ligand_io = io.StringIO()

#     for idx, (chain_id, res_idx, res_name, atom_name, atom_pos, atom_elem, is_lig) in enumerate(zip(
#             protein.chain_ids, protein.residue_idxs, protein.atom_residues, protein.atom_names, 
#             protein.atom_positions, protein.atom_elements, protein.is_ligand)):
#         try:
#             # 🔹 res_idx에서 숫자 부분만 추출 (숫자가 없으면 0으로 설정)
#             res_idx_clean = int(''.join(filter(str.isdigit, str(res_idx)))) if any(c.isdigit() for c in str(res_idx)) else 0
#         except ValueError as e:
#             print(f"⚠️ [ERROR] res_idx 변환 실패: {res_idx} (chain_id: {chain_id}, res_name: {res_name})")
#             raise e

#         # 🔹 원자명 (Atom Name) 포맷팅
#         if len(atom_name) == 4:  # 4글자 원자명은 왼쪽 정렬
#             atom_name_fixed = f"{atom_name:<4}"
#         elif len(atom_elem) == 1:  # 단일 원소 기호 (예: C, N, O)
#             if len(atom_name) > 1:
#                 atom_name_fixed = f"{atom_elem:>2}{atom_name[1:]:<2}"  # 원소 기호 오른쪽 정렬 + 뒤쪽 문자는 왼쪽 정렬
#             else:
#                 atom_name_fixed = f" {atom_elem:>1}  "  # 단독 원소는 13-14 컬럼 정렬, 뒤쪽 공백
#         elif len(atom_elem) == 2:  # 두 글자 원소 (예: FE, SE)
#             atom_name_fixed = f"{atom_elem:>2}  "  # 원소 기호만 배치, 나머지는 공백
#         else:
#             raise ValueError(f"⚠️ 알 수 없는 원소 형식: {atom_elem}")

#         # 🔹 PDB Format에 맞춘 정렬
#         pdb_line = (
#             f"{'HETATM' if is_lig==1 else 'ATOM  '}"  # (1-6) Record Type
#             f"{idx+1:>5} "                         # (7-11) Atom Serial Number (오른쪽 정렬)
#             f"{atom_name_fixed:<4}"                 # (13-16) Atom Name (PDB 포맷 적용, 4글자면 왼쪽 정렬)
#             f" "                                    # (17) Alternate Location
#             f"{res_name:>3} "                       # (18-20) Residue Name (우측 정렬)
#             f"{chain_id:>1}"                        # (22) Chain Identifier
#             f"{res_idx_clean:>4d}"                  # (23-26) Residue Sequence Number (우측 정렬)
#             f" "                                    # (27) Insertion Code
#             f"   "                                  # (28-30) Unused Columns (공백)
#             f"{atom_pos[0]:>8.3f}"                  # (31-38) X Coordinate (우측 정렬)
#             f"{atom_pos[1]:>8.3f}"                  # (39-46) Y Coordinate (우측 정렬)
#             f"{atom_pos[2]:>8.3f}"                  # (47-54) Z Coordinate (우측 정렬)
#             f"{1.00:>6.2f}"                         # (55-60) Occupancy (우측 정렬, 기본값 1.00)
#             f"{0.00:>6.2f}"                         # (61-66) Temperature Factor (우측 정렬, 기본값 0.00)
#             f"          "                          # (73-76) Segment Identifier (공백)
#             f"{atom_elem:>2}"                       # (77-78) Element Symbol (우측 정렬)
#             f"  "                                   # (79-80) Charge (공백)
#             f"\n"
#         )

#         pdb_io.write(pdb_line)

#         if is_lig:
#             ligand_io.write(pdb_line)
#         else:
#             protein_io.write(pdb_line)

#     pdb_io.seek(0)
#     protein_io.seek(0)
#     ligand_io.seek(0)
#     return pdb_io, protein_io, ligand_io


def make_pdb(protein: StructureWithGrid) -> tuple[io.StringIO, io.StringIO, io.StringIO]:
    pdb_io = io.StringIO()
    protein_io = io.StringIO()
    ligand_io = io.StringIO()

    unique_residues = {}  # (chain_id, original_res_idx, insertion_code) -> new_res_idx
    res_counter = defaultdict(int)  # chain_id 별 residue index 관리

    for idx, (chain_id, res_idx, res_name, atom_name, atom_pos, atom_elem, is_lig) in enumerate(zip(
            protein.chain_ids, protein.residue_idxs, protein.atom_residues, protein.atom_names, 
            protein.atom_positions, protein.atom_elements, protein.is_ligand)):
        try:
            # 🔹 res_idx를 문자열로 변환 후 숫자와 insertion code 분리
            res_idx_str = str(res_idx)
            numeric_part = ''.join(filter(str.isdigit, res_idx_str))  # 숫자만 추출
            insertion_code = ''.join(filter(str.isalpha, res_idx_str))  # 문자만 추출

            res_idx_clean = int(numeric_part) if numeric_part else 0  # 숫자가 없으면 기본값 0
        except ValueError as e:
            print(f"⚠️ [ERROR] res_idx 변환 실패: {res_idx} (chain_id: {chain_id}, res_name: {res_name})")
            raise e

        # 🔹 같은 chain 내에서 residue index 정렬 유지
        res_key = (chain_id, res_idx_clean, insertion_code)
        if res_key not in unique_residues:
            res_counter[chain_id] += 1
            unique_residues[res_key] = res_counter[chain_id]
        new_res_idx = unique_residues[res_key]

        # 🔹 원자명 (Atom Name) 포맷팅
        if len(atom_name) == 4:  # 4글자 원자명은 왼쪽 정렬
            atom_name_fixed = f"{atom_name:<4}"
        elif len(atom_elem) == 1:  # 단일 원소 기호 (예: C, N, O)
            if len(atom_name) > 1:
                atom_name_fixed = f"{atom_elem:>2}{atom_name[1:]:<2}"  # 원소 기호 오른쪽 정렬 + 뒤쪽 문자 왼쪽 정렬
            else:
                atom_name_fixed = f" {atom_elem:>1}  "  # 단독 원소는 13-14 컬럼 정렬, 뒤쪽 공백
        elif len(atom_elem) == 2:  # 두 글자 원소 (예: FE, SE)
            atom_name_fixed = f"{atom_elem:>2}  "  # 원소 기호만 배치, 나머지는 공백
        else:
            raise ValueError(f"⚠️ 알 수 없는 원소 형식: {atom_elem}")

        # 🔹 PDB Format에 맞춘 정렬
        pdb_line = (
            f"{'HETATM' if is_lig==1 else 'ATOM  '}"  # (1-6) Record Type
            f"{idx+1:>5} "                         # (7-11) Atom Serial Number (오른쪽 정렬)
            f"{atom_name_fixed:<4}"                 # (13-16) Atom Name (PDB 포맷 적용, 4글자면 왼쪽 정렬)
            f" "                                    # (17) Alternate Location
            f"{res_name:>3} "                       # (18-20) Residue Name (우측 정렬)
            f"{chain_id:>1}"                        # (22) Chain Identifier
            f"{new_res_idx:>4d}"                    # (23-26) Residue Sequence Number (우측 정렬, 유니크한 index)
            f" "                                    # (27) Insertion Code
            f"   "                                  # (28-30) Unused Columns (공백)
            f"{atom_pos[0]:>8.3f}"                  # (31-38) X Coordinate (우측 정렬)
            f"{atom_pos[1]:>8.3f}"                  # (39-46) Y Coordinate (우측 정렬)
            f"{atom_pos[2]:>8.3f}"                  # (47-54) Z Coordinate (우측 정렬)
            f"{1.00:>6.2f}"                         # (55-60) Occupancy (우측 정렬, 기본값 1.00)
            f"{0.00:>6.2f}"                         # (61-66) Temperature Factor (우측 정렬, 기본값 0.00)
            f"          "                          # (73-76) Segment Identifier (공백)
            f"{atom_elem:>2}"                       # (77-78) Element Symbol (우측 정렬)
            f"  "                                   # (79-80) Charge (공백)
            f"\n"
        )

        pdb_io.write(pdb_line)

        if is_lig:
            ligand_io.write(pdb_line)
        else:
            protein_io.write(pdb_line)

    pdb_io.seek(0)
    protein_io.seek(0)
    ligand_io.seek(0)
    return pdb_io, protein_io, ligand_io



def process_pdb(pdb_io)->str:
    with tempfile.NamedTemporaryFile(suffix=".pdb", mode="w", delete=False) as temp_pdb:
        temp_pdb.write(pdb_io.getvalue())
        temp_pdb.flush()
        temp_pdb_path = temp_pdb.name
    
    return temp_pdb_path

class CustomClassifier(freesasa.Classifier):
    """
    FreeSASA의 사용자 정의 Classifier
    - 원자의 유형을 특정 기준으로 분류
    - 알 수 없는 원자의 반지름을 1.5Å로 설정
    """
    purePython = True  # 필수 설정

    def classify(self, res_name: str, atom_name: str) -> str:
        """ 원자의 유형을 분류하는 함수 """
        if re.match(r'\s*N', atom_name): return 'Nitrogen'
        if re.match(r'\s*C', atom_name): return 'Carbon'
        if re.match(r'\s*O', atom_name): return 'Oxygen'
        if re.match(r'\s*S', atom_name): return 'Sulfur'
        return 'Unknown'  # 알 수 없는 원자

    def radius(self, res_name: str, atom_name: str) -> float:
        """ 
        원자의 반지름을 반환하는 함수
        - 기본적인 원소에 대한 반지름 설정
        - 알 수 없는 원소는 1.5Å로 설정
        """
        if re.match(r'\s*N', atom_name): return 1.6  # Nitrogen
        if re.match(r'\s*C', atom_name): return 1.7  # Carbon
        if re.match(r'\s*O', atom_name): return 1.4  # Oxygen
        if re.match(r'\s*S', atom_name): return 1.8  # Sulfur
        return 1.5  # Unknown atoms → 1.5Å 설정

def calculate_sasa(pdb_path: str) -> np.ndarray:
    """
    사용자 정의 Classifier를 적용한 SASA 계산 (리간드 포함)
    - 알 수 없는 원소 반지름을 1.5Å로 설정
    """
    # PDB 파일을 BioPython으로 파싱
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_path)

    # 사용자 정의 Classifier 적용
    classifier = CustomClassifier()
    options = {'hetatm': True, 'skip-unknown': False}  # 리간드 포함, unknown 스킵 X

    try:
        result, sasa_classes = freesasa.calcBioPDB(structure, classifier=classifier, options=options)
    except Exception as e:
        print(f"[ERROR] FreeSASA 계산 중 오류 발생: {e}")
        
        atom_count = len(list(structure.get_atoms()))
        return np.zeros(atom_count)  # 오류 발생 시 0 배열 반환

    sasa = np.array([result.atomArea(i) for i in range(result.nAtoms())]) / 50
    return sasa


def q_per_atom(ligand_mol, structure: StructureWithGrid):
    ligand_mask = structure.is_ligand == 1
    protein_mask = ~ligand_mask
    qs = np.zeros(len(structure.atom_names))

    # 단백질 부분 전하 계산
    prot_qs = [
        partial_q[res][atom] if res in partial_q and atom in partial_q[res]
        else partial_q[atom] if atom == "OXT"
        else (print(f"Missing charge data for Residue: {res}, Atom: {atom}") or 0.0)
        for res, atom in zip(structure.atom_residues[protein_mask], structure.atom_names[protein_mask])
    ]

    # 리간드 부분 전하 계산
    ligand_qs = []
    if ligand_mol is not None:
        try:
            ligand_mol.addh()  # ✅ 수소 추가
            ligand_mol.calccharges("Gasteiger")  # ✅ Gasteiger 전하 계산
            ligand_mol.removeh()  # ✅ 다시 원래 리간드 형태로 수소 제거
            for atom in ligand_mol.atoms:
                charge = atom.charge  # ✅ OpenBabel에서 Gasteiger 전하 값 가져오기
                ligand_qs.append(charge)
        except Exception as e:
            print(f"⚠️ Gasteiger charge calculation failed: {e}")
            ligand_qs = [0.0] * np.sum(ligand_mask)
    else:
        ligand_qs = [0.0] * np.sum(ligand_mask)

    # numpy 배열 변환
    prot_qs = np.array(prot_qs, dtype=np.float32)
    ligand_qs = np.array(ligand_qs, dtype=np.float32)

    # NaN 및 Inf 값 변환
    prot_qs = np.nan_to_num(prot_qs, nan=0.0, posinf=0.0, neginf=0.0)
    ligand_qs = np.nan_to_num(ligand_qs, nan=0.0, posinf=0.0, neginf=0.0)

    # 🔹 개수 검증 (Assertion)
    assert ligand_mask.sum() == len(ligand_qs), f"Mismatch: ligand_mask.sum()={ligand_mask.sum()} != len(ligand_qs)={len(ligand_qs)}"
    assert protein_mask.sum() == len(prot_qs), f"Mismatch: protein_mask.sum()={protein_mask.sum()} != len(prot_qs)={len(prot_qs)}"

    # qs 업데이트
    qs[np.where(protein_mask)] = prot_qs
    qs[np.where(ligand_mask)] = ligand_qs

    return qs

# def q_per_atom(ligand_mol, structure: StructureWithGrid):
#     ligand_mask = structure.is_ligand == 1
#     protein_mask = ~ligand_mask
#     qs = np.zeros(len(structure.atom_names))

#     # 단백질 부분 전하 계산
#     prot_qs = [
#         partial_q[res][atom] if res in partial_q and atom in partial_q[res]
#         else partial_q[atom] if atom == "OXT"
#         else (print(f"Missing charge data for Residue: {res}, Atom: {atom}") or 0.0)
#         for res, atom in zip(structure.atom_residues[protein_mask], structure.atom_names[protein_mask])
#     ]

#     # 리간드 부분 전하 계산
#     ligand_qs = []
#     if ligand_mol is not None:
#         for atom in ligand_mol.GetAtoms():
#             try:
#                 charge = atom.GetDoubleProp("_GasteigerCharge")
#             except KeyError:
#                 charge = 0.0  # 전하 값을 가져올 수 없으면 0으로 설정
#             ligand_qs.append(charge)
#     else:
#         ligand_qs = [0.0] * np.sum(ligand_mask)

#     # numpy 배열 변환
#     prot_qs = np.array(prot_qs, dtype=np.float32)
#     ligand_qs = np.array(ligand_qs, dtype=np.float32)

#     # NaN 및 Inf 값 확인 및 변환
#     prot_qs = np.nan_to_num(prot_qs, nan=0.0, posinf=0.0, neginf=0.0)
#     ligand_qs = np.nan_to_num(ligand_qs, nan=0.0, posinf=0.0, neginf=0.0)
#     # **🔹 개수 검증 (Assertion)**
#     assert ligand_mask.sum() == len(ligand_qs), f"Mismatch: ligand_mask.sum()={ligand_mask.sum()} != len(ligand_qs)={len(ligand_qs)}"
#     assert protein_mask.sum() == len(prot_qs), f"Mismatch: protein_mask.sum()={protein_mask.sum()} != len(prot_qs)={len(prot_qs)}"

#     # qs 업데이트
#     qs[np.where(protein_mask)] = prot_qs
#     qs[np.where(ligand_mask)] = ligand_qs

#     # NaN 체크 및 출력
#     print(f"NaN in prot_qs: {np.isnan(prot_qs).sum()}")
#     print(f"NaN in ligand_qs: {np.isnan(ligand_qs).sum()}")
#     print(f"Inf in prot_qs: {np.isinf(prot_qs).sum()}")
#     print(f"Inf in ligand_qs: {np.isinf(ligand_qs).sum()}")

#     if np.isnan(ligand_qs).sum() > 0 or np.isinf(ligand_qs).sum() > 0:
#         print("Invalid values in ligand_qs:", ligand_qs)

#     return qs


def secondary_struct(pdb_path: str,structure) -> np.ndarray:
    parser = PDBParser()
    structure_pdb = parser.get_structure('protein', pdb_path)
    model = structure_pdb[0]
    dssp = DSSP(model, pdb_path)

    secondary_structure_by_atom = []
    for chain in structure_pdb.get_chains():
        for residue in chain.get_residues():
            res_id = residue.get_id()
            chain_id = chain.get_id()
            try:
                dssp_info = dssp[(chain_id, res_id)]
                sec_structure = dssp_info[2]  # DSSP에서 이차 구조 정보 가져오기
            except KeyError:
                sec_structure = '-'  # DSSP 정보가 없을 경우 기본값('-') 설정

            for _ in residue.get_atoms():
                secondary_structure_by_atom.append(sec_structure)

    secs = [sec_struct_dict[sec] for sec in secondary_structure_by_atom]
    secs_array = np.array(secs)

    # 리간드의 이차구조는 'L'로 설정
    ligand_mask = structure.is_ligand == 1
    secs_array[ligand_mask] = sec_struct_dict['L']
    
    return secs_array

def cov_bonds_mask(structure: StructureWithGrid, ligand_mol):
    cov_bonds_mask = np.zeros((len(structure.atom_names), len(structure.atom_names)))
    #protein
    for i, (chain, res_idx, res_name, atom_name) in enumerate(zip(structure.chain_ids,structure.residue_idxs, structure.atom_residues, structure.atom_names)):
        if res_name in standard_residues:
            for (atom, neigh) in bonds[res_name]:
                if atom == atom_name:
                    mask = (structure.residue_idxs == res_idx) & (structure.atom_residues == res_name) & (structure.atom_names == neigh)
                    cov_bonds_mask[i][mask] = 1
            if atom_name == "N":
                bb_mask = (structure.residue_idxs == res_idx-1) & (structure.chain_ids == chain) & (structure.atom_names == 'C') & (structure.is_ligand == 0)
                cov_bonds_mask[i][bb_mask] = 1
            elif atom_name == 'C':
                bb_mask = (structure.residue_idxs == res_idx+1) & (structure.chain_ids == chain) & (structure.atom_names == 'N') & (structure.is_ligand == 0)
                cov_bonds_mask[i][bb_mask] = 1
    #ligand            
    ligand_mask = structure.is_ligand == 1
    ligand_indices = np.where(ligand_mask)[0]  # 리간드 원자 인덱스

    if ligand_mol:
        for bond in ligand_mol.GetBonds():
            atom1_idx = bond.GetBeginAtomIdx()
            atom2_idx = bond.GetEndAtomIdx()

            # RDKit의 인덱스를 Structure의 인덱스로 매핑
            if atom1_idx < len(ligand_indices) and atom2_idx < len(ligand_indices):
                global_idx1 = ligand_indices[atom1_idx]
                global_idx2 = ligand_indices[atom2_idx]

                cov_bonds_mask[global_idx1, global_idx2] = 1
                cov_bonds_mask[global_idx2, global_idx1] = 1  # 대칭 행렬

    return cov_bonds_mask

def make_gentype(structure:Features, ligand_mol):
    print('리간드 atom 개수',sum(structure.is_ligand))
    prot_gentype = []
    lig_gentype = []
    gentype = np.ones((len(structure.atom_elements)))
    for res, atm, is_lig in zip(structure.atom_residues, structure.atom_names, structure.is_ligand):
        if is_lig == 0:
            if res in standard_residues:
                if atm == 'OXT':
                    prot_gentype.append(6)
                else:
                    prot_gentype.append(atype2num[atypes.get((res,atm),'X')])
            else:
                prot_gentype.append(60)
    count = 0
    print('outside ligandmol2')
    print(ligand_mol)
    if ligand_mol:
        print('inside ligandmol2')
        print(ligand_mol)
        for atom in ligand_mol.GetAtoms():
            count+=1
            sybyl_type = atom.GetProp("_TriposAtomType") if atom.HasProp("_TriposAtomType") else "Du"
            lig_gentype.append(sybyl_type_dict.get(sybyl_type, 60))      
    print('리간드 count',count)
    gentype[structure.is_ligand==0] = np.array(prot_gentype)
    gentype[structure.is_ligand==1] = np.array(lig_gentype)
    
    return gentype
         

# def make_features(pdb_path: Optional[str], structure: StructureWithGrid) -> Optional[Features]:
#     print('pdb_path', pdb_path)
    
#     pdb_io, protein_io, ligand_io = make_pdb(structure)
#     ligand_pdb_str = ligand_io.getvalue()
#     ligand_mol = None
    
#     print('ligand_mol', ligand_pdb_str)
#     if ligand_pdb_str.strip():
#         print('inside ligand_mol')
#         ligand_mol = Chem.MolFromPDBBlock(ligand_pdb_str, removeHs=False)

#         if ligand_mol is None:
#             print("⚠️ RDKit failed to parse the ligand. Logging to error.txt")
#             with open("/home/qkrgangeun/LigMet/code/text/ligand_error.txt", "a") as error_file:  
#                 error_file.write(f"{pdb_path}\n")
#             return None

#     new_pdb_path = process_pdb(pdb_io)
#     new_structure = read_pdb(new_pdb_path)

#     # Feature 계산
#     sasas = calculate_sasa(new_pdb_path)
#     qs = q_per_atom(ligand_mol, new_structure)

#     # 🔹 DSSP 오류 처리 추가
#     try:
#         sec_structs = secondary_struct(new_pdb_path, new_structure)
#     except Exception as e:
#         print(f"⚠️ DSSP failed: {e} - Logging to dssp_error.txt")
#         with open("/home/qkrgangeun/LigMet/code/text/dssp_error.txt", "a") as error_file:
#             error_file.write(f"{pdb_path}\n")
#         return None

#     bond_masks = cov_bonds_mask(new_structure, ligand_mol)
#     gen_types = make_gentype(new_structure, ligand_mol)

#     # Features 객체 반환
#     features = Features(
#         **{k: v for k, v in new_structure.__dict__.items() if k not in ["metal_positions", "metal_types"]},
#         metal_positions=structure.metal_positions,  # structure의 값 유지
#         metal_types=structure.metal_types,  # structure의 값 유지
#         grid_positions=structure.grid_positions,
#         gen_types=gen_types,
#         sasas=sasas,
#         qs=qs,
#         sec_structs=sec_structs,
#         bond_masks=bond_masks
#     )

#     return features


from openbabel import pybel

def make_features(pdb_path: Optional[str], structure: StructureWithGrid) -> Optional[Features]:
    print('pdb_path', pdb_path)

    # PDB 생성
    pdb_io, protein_io, ligand_io = make_pdb(structure)
    ligand_pdb_str = ligand_io.getvalue()
    ligand_mol = None

    # 리간드가 존재하는 경우 OpenBabel로 변환
    print('ligand_mol', ligand_pdb_str)
    if ligand_pdb_str.strip():
        print('inside ligand_mol')

        try:
            ob_mol = pybel.readstring("pdb", ligand_pdb_str)
            ligand_mol = ob_mol  # ✅ Gasteiger 전하는 q_per_atom에서 처리
        except Exception as e:
            print(f"⚠️ OpenBabel failed to parse the ligand: {e}. Logging to error.txt")
            with open("/home/qkrgangeun/LigMet/code/text/ligand_error.txt", "a") as error_file:
                error_file.write(f"{pdb_path}\n")
            return None

    # PDB 파일을 처리하여 새로운 구조 생성
    new_pdb_path = process_pdb(pdb_io)
    new_structure = read_pdb(new_pdb_path)

    # Feature 계산
    sasas = calculate_sasa(new_pdb_path)
    qs = q_per_atom(ligand_mol, new_structure)  # ✅ Gasteiger 전하 계산은 q_per_atom에서 수행

    # 🔹 DSSP 오류 처리 추가
    try:
        sec_structs = secondary_struct(new_pdb_path, new_structure)
    except Exception as e:
        print(f"⚠️ DSSP failed: {e} - Logging to dssp_error.txt")
        with open("/home/qkrgangeun/LigMet/code/text/dssp_error.txt", "a") as error_file:
            error_file.write(f"{pdb_path}\n")
        return None

    bond_masks = cov_bonds_mask(new_structure, ligand_mol)
    gen_types = make_gentype(new_structure, ligand_mol)

    # Features 객체 반환
    features = Features(
        **{k: v for k, v in new_structure.__dict__.items() if k not in ["metal_positions", "metal_types"]},
        metal_positions=structure.metal_positions,  # structure의 값 유지
        metal_types=structure.metal_types,  # structure의 값 유지
        grid_positions=structure.grid_positions,
        gen_types=gen_types,
        sasas=sasas,
        qs=qs,
        sec_structs=sec_structs,
        bond_masks=bond_masks
    )

    return features
