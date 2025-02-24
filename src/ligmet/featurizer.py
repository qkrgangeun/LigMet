from dataclasses import dataclass
import numpy as np
from Bio.PDB import PDBParser, DSSP
from typing import Optional,Union
import freesasa
import tempfile
from pathlib import Path
import re
import torch
from ligmet.utils.pdb import StructureWithGrid
import io
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

def make_pdb(protein: StructureWithGrid) -> tuple[io.StringIO, io.StringIO, io.StringIO]:
    pdb_io = io.StringIO()
    protein_io = io.StringIO()
    ligand_io = io.StringIO()

    for idx, (chain_id, res_idx, res_name, atom_name, atom_pos, atom_elem, is_lig) in enumerate(zip(
            protein.chain_ids, protein.residue_idxs, protein.atom_residues, protein.atom_names, 
            protein.atom_positions, protein.atom_elements, protein.is_ligand)):
        pdb_line = f"{'HETATM' if is_lig else 'ATOM  '}{idx+1:5d}  {atom_name:<3} {res_name:>3} {chain_id:>1}{res_idx:>4d}    {atom_pos[0]:8.3f}{atom_pos[1]:8.3f}{atom_pos[2]:8.3f}  1.00  0.00          {atom_elem:>2}\n"
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

def q_per_atom(ligand_mol, structure:StructureWithGrid):
    ligand_mask = structure.is_ligand == 1
    protein_mask = ~ligand_mask
    qs = np.zeros(len(structure.atom_names))
    prot_qs = [
        partial_q[res][atom] if res in partial_q and atom in partial_q[res]
        else (print(f"Missing charge data for Residue: {res}, Atom: {atom}") or 0.0)
        for res, atom in zip(structure.atom_residues[protein_mask], structure.atom_names[protein_mask])
    ]

    ligand_qs = []

    if ligand_mol is not None:
        AllChem.ComputeGasteigerCharges(ligand_mol)
        ligand_qs = [
            atom.GetDoubleProp("_GasteigerCharge") for atom in ligand_mol.GetAtoms()
        ]
    else: 
        ligand_qs = [0.0]*np.sum(ligand_mask)
        
    qs[np.where(protein_mask)] = np.array(prot_qs)  
    qs[np.where(ligand_mask)] = np.array(ligand_qs)
    return qs

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
    if ligand_mol:
        for atom in ligand_mol.GetAtoms():
            sybyl_type = atom.GetProp("_TriposAtomType") if atom.HasProp("_TriposAtomType") else "Du"
            lig_gentype.append(sybyl_type_dict.get(sybyl_type, 60))      
    gentype[structure.is_ligand==0] = np.array(prot_gentype)
    gentype[structure.is_ligand==1] = np.array(lig_gentype)
    
    return gentype
         
def make_features(pdb_path: Optional[str], structure:StructureWithGrid) -> Features:
    # structure = np.load(structure_path, allow_pickle=True).item()
    # assert isinstance(structure, StructureWithGrid), "structure must be an instance of StructureWithGrid"

    pdb_io, protein_io, ligand_io = make_pdb(structure)
    
    ligand_pdb_str = ligand_io.getvalue()
    ligand_mol = None
    if ligand_pdb_str.strip():
        ligand_mol = Chem.MolFromPDBBlock(ligand_pdb_str, removeHs=False)
    pdb_path = process_pdb(pdb_io)
    # Feature 계산
    sasa = calculate_sasa(pdb_path)  
    qs = q_per_atom(ligand_mol, structure)
    sec_structs = secondary_struct(pdb_path, structure)  
    bond_masks = cov_bonds_mask(structure, ligand_mol)
    gen_type = make_gentype(structure, ligand_mol)

    # Features 객체 반환
    features = Features(
        atom_positions=structure.atom_positions,
        atom_names=structure.atom_names,
        atom_elements=structure.atom_elements,
        atom_residues=structure.atom_residues,
        residue_idxs=structure.residue_idxs,
        chain_ids=structure.chain_ids,
        is_ligand=structure.is_ligand,
        metal_positions=structure.metal_positions,
        metal_types=structure.metal_types,
        grid_positions=structure.grid_positions,
        gen_types=gen_type,
        sasas=sasa,
        qs=qs,
        sec_structs=sec_structs,
        bond_masks=bond_masks
    )

    return features