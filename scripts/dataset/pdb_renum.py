import os
import glob
from pathlib import Path
import subprocess as sp


one_letter_code = {
        'ALA':'A',
        'GLY':'G',
        'ILE':'I',
        'LEU':'L',
        'PRO':'P',
        'VAL':'V',
        'PHE':'F',
        'TRP':'W',
        'TYR':'Y',
        'ASP':'D',
        'GLU':'E',
        'ARG':'R',
        'HIS':'H',
        'LYS':'K',
        'SER':'S',
        'THR':'T',
        'CYS':'C',
        'MET':'M',
        'ASN':'N',
        'GLN':'Q',
        }

# pdb including receptor residues of which index are larger than 1000
EXCEPTION = [
            'ADRB1.LA',
            'ADRB1.LI',
            'ADRB2.LA',
        ]

# error from input seq error
# gpcr : residue to remove
PDB_MINOR_ERROR = {
        }

def renumbering(pdb, af_pdb, rec_id, out_fn):
    """
    input: pdb file to be renumbered
           AF pdb file (used as AF input) to be used as reference indexing
           receptor chain id
           output file name
    output: new pdb file of which residue number is
            matched to AF pdb index (receptor chain id is set to 'A')
    >> renumbering('../set/pdbs/GRM2/7TMS.pdb',
                    '../receptor/receptor_only/clean.GRM2.LA/relaxed_model_1.pdb',
                    'A',
                    './renum_pdb/GRM2.LA.pdb')
    """
    # check AF pdb exists
    if not os.path.exists(af_pdb):
        print("AF model doesn't exists: %s"%af_pdb)
        return

    tmp_dir = '/home/qkrgangeun/LigMet/data/biolip/junk/'
    # extract seq from pdb
    pdb_seq = []
    pdb_resn = []
    pre_res_n = ''
    
    with open(pdb) as f:
        for l in f:
            if l.startswith('ATOM') and (l[21]==rec_id):
                cur_res = l[17:20]
                cur_res_n = l[22:26]
                # skip inserted binding partener
                # of which residue number usually starts from 1000
                if (int(cur_res_n)> 1000) and (not is_exception):
                    continue
                if cur_res_n != pre_res_n:
                    pdb_seq.append(one_letter_code[cur_res])
                    pdb_resn.append(int(cur_res_n))
                    pre_res_n = cur_res_n
    pdbn_to_idx = dict(zip(pdb_resn, list(range(1,len(pdb_resn)+1))))
    

    os.makedirs(tmp_dir,exist_ok=True)
    tmp_fasta_fn = tmp_dir + out_fn.split('/')[-1].replace('.pdb','_tmp.fasta')
    pdb_seq_str = ''.join(pdb_seq)
    with open(tmp_fasta_fn,'w') as f:
        f.write('>%s\n'%pdb)
        f.write(pdb_seq_str+'\n')

    # extract seq from AF pdb
    af_seq = []
    pre_res_n = ''
    with open(af_pdb) as f:
        for l in f:
            if l.startswith('ATOM'):
                if 'receptor/' in af_pdb:
                    # chain 'A' of AF model is receptor chain
                    MATCH_CHAIN = (l[21] == 'A')
                else:
                    # chain 'A' or 'R' are receptor chain of original pdb
                    MATCH_CHAIN = (l[21]=='A') or (l[21]=='R')
                
                if MATCH_CHAIN:
                    cur_res = l[17:20]
                    cur_res_n = l[22:26]
                    if cur_res_n != pre_res_n:
                        af_seq.append(one_letter_code[cur_res])
                        pre_res_n = cur_res_n


    af_fasta = tmp_dir + af_pdb.split('/')[-1].replace('.pdb.','_af')+'_tmp.fasta'
    # '../junk/af.GRM2.LA_tmp.fasta'

    af_seq_str = ''.join(af_seq)
    with open(af_fasta,'w') as f:
        f.write('>%s\n'%af_pdb)
        f.write(af_seq_str+'\n')


    # HHalign
    tmp_hhr_fn = tmp_dir + out_fn.split('/')[-1].replace('.pdb','_tmp.hhr')
    os.system('/applic/hhsuite/current/bin/hhalign -i %s -t %s -o %s'%(
                tmp_fasta_fn, af_fasta, tmp_hhr_fn))

    # mapping
    with open(tmp_hhr_fn) as f:
        lines = f.readlines()
    """
    hhr result 
    10: query HMM[76:83] template HMM[86:93]
    ---- block of alignment result ---
    16:Q query head_res_n seq tail_res_n (seq_len)
    17:Q Consensus  ....
    18:aignment type for each residue [22:]
    19:T Consensus  ....
    20:T templ  query head_res_n seq tail_res_n (seq_len) 
    21:Confidence   ....
    22:
    23:
    ----- 8 lines consist a block ----
    """
    idx_to_fastan = {}
    i = 1
    while i < len(lines):
        # skip header
        if i < 16:
            i += 1
            continue

        # read alignment block
        if (i-16)%8 == 0:
            # pdb seq
            pdb_l = lines[i-1].strip().split()
            pdb_st_n = int(pdb_l[2])
            pdb_seq = pdb_l[3]
            # fasta seq
            fasta_l = lines[i+4-1].strip().split()
            fasta_st_n = int(fasta_l[2])
            fasta_seq = fasta_l[3]
            pdb_gap = 0
            fasta_gap = 0
            for r in range(len(pdb_seq)):
                is_gap = False
                # skip missing residue
                if pdb_seq[r]=='-':
                    pdb_gap += 1
                    is_gap = True
                # skip expression tag
                if fasta_seq[r]=='-':
                    fasta_gap += 1
                    is_gap = True
                if is_gap:
                    continue
                pdb_n = pdb_st_n+r-pdb_gap
                fasta_n = fasta_st_n+r-fasta_gap
                idx_to_fastan[pdb_n] = fasta_n 

            # move to next block
            i += 8

    # write result
    wrt = []
    with open(pdb) as f:
        for l in f:
            if l.startswith('ATOM') and (l[21]==rec_id):
                res_n = int(l[22:26])
                # skip expression tag at the ends
                # idx_to_fastan contains only residues 
                # which exists in both pdb and input fasta
                _idx = pdbn_to_idx.get(res_n, None)
                if _idx is None:
                    continue
                new_res_n = idx_to_fastan.get(_idx, None)
                if new_res_n is None:
                    continue
                # renumbering + fixing receptor chain id to 'A'
                new_l = l[:21] +'A'+ '%4d'%new_res_n + l[26:]
                wrt.append(new_l)
    out_fn_dir = os.path.dirname(out_fn)
    os.makedirs(out_fn_dir, exist_ok=True)
    with open(out_fn,'w') as f:
        f.write('MODEL %s %s\n'%(out_fn.split('/')[-1][:-4], rec_id))
        f.writelines(wrt)
        f.write('TER\n')
        f.write('ENDMDL\n')


def crop_af(pdb_fn, af_fn, out_fn):
    """
    input: renumbered pdb (pdb with only receptor domain)
           AF pdb
           output pdb file name
    extract residue numbers from renumbered pdb
    and then filtering AF pdb with those residues
    """
    if not os.path.exists(pdb_fn):
        print("PDB doesn't exist: %s"%pdb_fn)
        return
        
    res_n = []
    with open(pdb_fn) as f:
        for l in f:
            if l.startswith('ATOM'):
                res_n.append(int(l[22:26]))
    res_n = set(res_n)
    
    wrt = []
    with open(af_fn) as f:
        for l in f:
            # A chain of AF pdb is receptor chain
            if l.startswith('ATOM') and (l[21]=='A'):
                af_res_n = int(l[22:26])
                if af_res_n in res_n:
                    wrt.append(l)

    with open(out_fn,'w') as f:
        f.writelines(wrt)
            

def check_pdb_pdb(pdb, af_pdb):
    # extract seq from pdb
    seq = []
    res_n = []
    pre_res_n = ''
    with open(pdb) as f:
        for i,l in enumerate(f):
            if l.startswith('ATOM'):
                cur_res = l[17:20]
                cur_res_n = l[22:26]
                if cur_res_n != pre_res_n:
                    seq.append(one_letter_code[cur_res])
                    res_n.append(int(cur_res_n))
                    pre_res_n = cur_res_n
    # extract seq from af pdb
    af_seq = []
    af_res_n = []
    pre_res_n = ''
    with open(af_pdb) as f:
        for l in f:
            if l.startswith('ATOM'):
                cur_res = l[17:20]
                cur_res_n = l[22:26]
                if cur_res_n != pre_res_n:
                    af_seq.append(one_letter_code[cur_res])
                    af_res_n.append(int(cur_res_n))
                    pre_res_n = cur_res_n
    result = ''
    af_res_d = dict(zip(af_res_n, af_seq))
    for i in res_n:
        corr = af_res_d.get(i,'-')
        result += corr
    pdb_st = ''.join(seq).lstrip('-')[0]
    af_st = result.lstrip('-')[0]
    if (len(seq) != len(af_seq)) or (pdb_st!=af_st):
        print('%s (%d), %s (%d)'%(pdb, len(seq), af_pdb, len(af_seq)))
        print(''.join(seq))
        print(result)


#======================
# For temporary analysis
#======================
def check_old_new_fa():
    """
    Check whether the new cropped seq matches
    to the previously cropped seq which was used as AF input
    """
    different = []
    old_fasta_s = glob.glob('../old_input_seq/*.fa')
    print('num of old_fasta: %d'%len(old_fasta_s))
    for old_f in old_fasta_s:
        gpcr_name = '.'.join(old_f.split('/')[-1].split('.')[1:3]) 
        with open(old_f) as f:
            old_header = f.readline().strip().split(' ')[-1]
            old_seq = f.readline().strip()
        new_f = '../input_seq/%s'%old_f.split('/')[-1]
        
        if not os.path.exists(new_f):
            print('ERROR: Not found %s'%new_f)
            continue
        with open(new_f) as f:
            new_header = f.readline().strip().split(' ')[-1]
            new_seq = f.readline().strip()

        if old_seq!=new_seq:
            print('Miss-matchted seq %15s, %s, %s'%
                    (old_f.split('/')[-1], old_header, new_header))
            different.append(gpcr_name)

    different = set(different)
    print('Number of miss-match gpcr: %d'%len(different))
    for i in different:
        print(i)

#===============================
# for meiler,lim and cross
# 1) renumer according to af model
# 2) crop renumbered pdb accordin to original renum_pdb
#===============================
def meiler_renum():
    # renumbering Meiler inactive model
    meilers = glob.glob('../receptor/receptor_meiler/*.Inactive.pdb')
    for meiler_pdb in meilers:
        gpcr = meiler_pdb.split('/')[-1].split('.')[0] + '.LI'
        
        if gpcr == 'GABR2.LI': # meiler model has only 7TM, not ligand binding domain
            continue 
        
        af_pdb = '../receptor/receptor_only/clean.%s.single/relaxed_model_1.pdb'%gpcr
        if not os.path.exists(af_pdb):
            af_pdb = '../receptor/receptor_only/clean.%s/relaxed_model_1.pdb'%gpcr
        if not os.path.exists(af_pdb):
            print('No single sturecuture: %s'%gpcr)
            af_pdb = '../receptor/receptor_complex/clean.%s.alpha/relaxed_model_1_multimer.pdb'%gpcr
        rec = 'A' # receptor chain id of meiler model = 'A'
        out_fn = './renum_pdb/meiler.%s.pdb'%gpcr

        renumbering(meiler_pdb, af_pdb, rec, out_fn)

        ref_pdb = './renum_pdb/%s.pdb'%gpcr
        crop_af(ref_pdb, out_fn, out_fn)
        exit()

def Lim_renum():
    """
    if model in lim_db, it is used
    else generated model is used
    """
    lim_db = glob.glob('/home/seeun/GPCR_bench/rec/alphafold-multistate/human_gpcr/af-msa+db_*/*')
    lim_db += glob.glob('/home/seeun/GPCR_bench/rec/alphafold-multistate/new_structure/af-msa+db_*/*')
    uni_lim_d = {}
    for i in lim_db:
        uni = i.split('/')[-1].split('_')[0]
        uni = uni.split('.')[0]
        uni = uni.split('-')[0]
        uni_lim_d[uni] = i
    
    in_db = {}
    for gpcr, pdb in name_pdb.items():
        # check gpcr is in lim_db
        uni = pdb_uniprot[pdb]
        if uni_lim_d.get(uni, None):
           lim_pdb = uni_lim_d[uni]
           in_db[gpcr] = lim_pdb
        else:
            if '.LA' in gpcr:
                lim_pdb  = '../receptor/receptor_lim/clean.%s.single_active/relaxed_model_1.pdb'%gpcr
                if not os.path.exists(lim_pdb):
                    lim_pdb = '../receptor/receptor_lim/clean.%s_active/relaxed_model_1.pdb'%gpcr
            else:
                lim_pdb = '../receptor/receptor_lim/clean.%s_inactive/relaxed_model_1.pdb'%gpcr
                if not os.path.exists(lim_pdb):
                    lim_pdb = '../receptor/receptor_lim/clean.%s.single_inactive/relaxed_model_1.pdb'%gpcr
        # check pdb is empty
        with open(lim_pdb) as f:
            lines = len(f.readlines())
        if lines == 0:
            print('Empty pdb: %s'%lim_pdb)
       
        af_pdb = '../receptor/receptor_only/clean.%s.single/relaxed_model_1.pdb'%gpcr
        if not os.path.exists(af_pdb):
            af_pdb = '../receptor/receptor_only/clean.%s/relaxed_model_1.pdb'%gpcr
        if not os.path.exists(af_pdb):
            print('No single sturecuture: %s'%gpcr)
            af_pdb = '../receptor/receptor_complex/clean.%s.alpha/relaxed_model_1_multimer.pdb'%gpcr
        
        # IRRITATING renumbering error... :(
        out_fn = './renum_pdb/lim.%s.pdb'%gpcr
        rec= 'A' # receptor chain id of limdb = 'A'
        renumbering(lim_pdb, af_pdb, rec, out_fn)

        ref_pdb = './renum_pdb/%s.pdb'%gpcr
        crop_af(ref_pdb, out_fn, out_fn)
    
    for k, v in in_db.items():
        print('%s: %s'%(k,v))
    print('Number of gpcr in LimDB : %d (/%d)'%(len(in_db), len(name_pdb)))
    

def cross_renum():
    # renumbering Cross models
    CROSS_DIR = Path(f'/home/sumin/GPCR_bench/receptor/receptor_cross')
    cross_s = CROSS_DIR.glob('*.pdb')
    for cross_pdb in cross_s:
        gpcr_name = cross_pdb.stem.strip('cross.')
        af_pdb = Path(f'../receptor/receptor_only/clean.{gpcr_name}.single/relaxed_model_1.pdb')
        if not af_pdb.exists():
            af_pdb = Path(f'../receptor/receptor_only/clean.{gpcr_name}/relaxed_model_1.pdb')
        if not af_pdb.exists():
            print(f'No single chain AF sturcutre: {gpcr_name}')
            af_pdb = Path(f'../receptor/receptor_complex/clean.{gpcr_name}.alpha/relaxed_model_1_multimer.pdb')
        rec = cross_rec_chain[gpcr_name] # receptor chain id
        out_fn = Path(f'./renum_pdb/cross.{gpcr_name}.pdb')

        renumbering(str(cross_pdb), str(af_pdb), rec, str(out_fn))
        
        ref_pdb = Path(f'./renum_pdb/{gpcr_name}.pdb')
        crop_af(str(ref_pdb), str(out_fn), str(out_fn))


def Help_prep_recycle_get_rec_type(recycle_result_dir):
    if 'alpha' in recycle_result_dir.name:
        return 'alpha'
    elif 'single' in recycle_result_dir.name:
        return 'single'
    elif '.LI' in recycle_result_dir.name:
        return 'single'
    else:
        if (recycle_result_dir/'unrelaxed_model_5.pdb').exists():
            return 'single'
        elif (recycle_result_dir/'unrelaxed_model_5_multimer.pdb').exists():
            return 'complex'


def Help_prep_recycle_check_match(ref_af_pdb, af_pdb):
    # check residue numbering is correct
    # af_res_n = ref_af_res_n + diff

    # extract seq from pdb
    seq = []
    res_n = []
    pre_res_n = ''
    with open(ref_af_pdb) as f:
        for i,l in enumerate(f):
            if l.startswith('ATOM'):
                cur_res = l[17:20]
                cur_res_n = l[22:26]
                if cur_res_n != pre_res_n:
                    seq.append(one_letter_code[cur_res])
                    res_n.append(int(cur_res_n))
                    pre_res_n = cur_res_n

    # extract seq from af pdb
    af_seq = []
    af_res_n = []
    pre_res_n = ''
    with open(af_pdb) as f:
        for l in f:
            if l.startswith('ATOM'):
                cur_res = l[17:20]
                cur_res_n = l[22:26]
                if cur_res_n != pre_res_n:
                    af_seq.append(one_letter_code[cur_res])
                    af_res_n.append(int(cur_res_n))
                    pre_res_n = cur_res_n
 
    ref_seq_str = ''.join(seq[:5])
    af_seq_str = ''.join(af_seq)
    diff = af_seq_str.find(ref_seq_str)+1-res_n[0]
    return diff


def Help_prep_recycle_crop(ref_af_pdb, af_pdb, diff, out_fn):
    if not os.path.exists(ref_af_pdb):
        print("PDB doesn't exsit: %s"%ref_af_pdb)
        return
    res_n = []
    with open(ref_af_pdb) as f:
        for l in f:
            if l.startswith('ATOM'):
                res_n.append(int(l[22:26]))
    wrt = []
    with open(af_pdb) as f:
        for l in f:
            if l.startswith('ATOM') and (l[21]=='A'):
                af_res_n = int(l[22:26])
                new_res_n = af_res_n - diff
                if new_res_n in res_n:
                    new_l = l[:22] + f'{new_res_n:4d}' + l[26:]
                    wrt.append(new_l)
    if len(wrt)==0: #DEBUG multimer have no A chain (why??)
        # instead, first chain is B
        with open(af_pdb) as f:
            for l in f:
                if l.startswith('ATOM') and (l[21]=='B'):
                    af_res_n = int(l[22:26])
                    new_res_n = af_res_n - diff
                    if new_res_n in res_n:
                        new_l = l[:22] + f'{new_res_n:4d}' + l[26:]
                        wrt.append(new_l)

    with open(out_fn,'w') as f:
        f.writelines(wrt)


def prep_recycle():
    result_dir_s = list(Path('/home/sumin/GPCR_bench/receptor/recycle').glob('recycle_*/*'))
    n = 0#DEBUG
    for result_dir in result_dir_s:
        recycle_n = result_dir.parents[0].name.split('_')[-1]
        gpcr = '.'.join(result_dir.name.split('.')[1:3])
        rec_type = Help_prep_recycle_get_rec_type(result_dir)
        
        #check result pdb exists
        RESULT_EXISTS = False
        if len(list(result_dir.glob('relaxed_model_*.pdb'))) == 5:
            RESULT_EXISTS = True
        elif (result_dir/'unrelaxed_model_5.pdb').exists():
            UNRESULT_EXISTS = True
        elif (result_dir/'unrelaxed_model_5_multimer.pdb').exists():
            UNRESULT_EXISTS = True
        if not (RESULT_EXISTS or UNRESULT_EXISTS):
            n += 1 #DEBUG
            print(f'{n}: {result_dir}')

        # extract main chain and cropping
        ref_af_pdb = f'./renum_pdb/af.{gpcr}.single.1.pdb'
        if RESULT_EXISTS:
            af_pdb_s = result_dir.glob('relaxed_model_*.pdb')
        else:
            af_pdb_s = result_dir.glob('unrelaxed_model_*.pdb')

        for af_pdb in af_pdb_s:
            model_id  = af_pdb.stem.split('_')[2]
            diff = Help_prep_recycle_check_match(ref_af_pdb, af_pdb)
            out_fn = f'./renum_pdb/re_{recycle_n}.{gpcr}.{rec_type}.{model_id}.pdb'
            Help_prep_recycle_crop(ref_af_pdb,af_pdb,diff,out_fn)
            if not Path(out_fn).exists(): #DEBUG
                print(f'No A chain in : {af_pdb}')
                continue
            check_pdb_pdb(ref_af_pdb, out_fn)


def main():
    for gpcr, pdb in tmp_name_pdb.items():
        # pdb renumbering and cropping
        pdb = '../set/pdbs/%s/%s.pdb'%(gpcr.split('.')[0],pdb)
        af_pdb = '../receptor/receptor_only/clean.%s.single/relaxed_model_1.pdb'%gpcr
        if not os.path.exists(af_pdb):
            af_pdb = '../receptor/receptor_only/clean.%s/relaxed_model_1.pdb'%gpcr
        if not os.path.exists(af_pdb):
            print('No sinlge structure: %s'%gpcr)
            af_pdb = '../receptor/receptor_complex/clean.%s.alpha/relaxed_model_1_multimer.pdb'%gpcr
        rec = name_rec[gpcr]
        out_fn = './renum_pdb/%s.pdb'%gpcr
        
        if os.path.exists(out_fn):
            pass
        else:
            renumbering(pdb, af_pdb, rec, out_fn)
        
        MAX_MODEL_NUM = 5
        af_pdb_s = glob.glob('../receptor/receptor_only/clean.%s*/relaxed_model_*.pdb'%gpcr)
        af_pdb_s += glob.glob('../receptor/receptor_complex/clean.%s*/relaxed_model_*.pdb'%gpcr)
        
        # AF model cropping
        for p in af_pdb_s:
            model_id = int(p.split('/')[-1].split('_')[2].strip('.pdb'))
            if model_id > MAX_MODEL_NUM:
                continue

            elif 'receptor_only' in p:
                af_out_fn = './renum_pdb/af.%s.single.%d.pdb'%(gpcr, model_id)

            elif 'alpha' in p:
                af_out_fn = './renum_pdb/af.%s.alpha.%d.pdb'%(gpcr, model_id)
            
            elif 'receptor_complex' in p:
                af_out_fn = './renum_pdb/af.%s.complex.%d.pdb'%(gpcr, model_id)
                 
            """
            EXCEPTION: 
                OX2R.LA.alpha: unrelaxed model, receptor chain is B not A
            """
            if 'OX2R.LA.alpha' in p:
                wrt = []
                with open(p) as f:
                    for l in f:
                        if l.startswith('ATOM') and l[21]=='B':
                            wrt.append(l[:21]+'A'+l[22:])
                with open(p,'w') as f:
                    f.writelines(wrt)

            crop_af(out_fn, p, af_out_fn)
            check_pdb_pdb(out_fn, af_out_fn)


def temp_alpha_renumber():
    """
    original af.alpha.pdb --> renumbered and cropped to match renumbered af.single.pdb
    AF alpha model pdb renumbering : due to input_seq error
    alpha residue is one greater than single or complex number
    """
    tmp_dir = '../junk/'
    alpha_s = glob.glob('./renum_pdb/af.*.alpha.[0-9].pdb')
    for alpha in alpha_s:
        gpcr_name = '.'.join(alpha.split('/')[-1].split('.')[1:3])
        ref = './renum_pdb/af.%s.single.1.pdb'%gpcr_name
        if not os.path.exists(ref):
            print('%s sinlge pdb does not exist.'%gpcr_name)
            continue
   
        # extract seq from alpha
        alpha_seq = []
        alpha_res_n = []
        pre_res_n = ''
        with open(alpha) as f:
            for l in f:
                if l.startswith('ATOM') and (l[21]=='A'):
                    cur_res = l[17:20]
                    cur_res_n = l[22:26]
                    if cur_res_n != pre_res_n:
                        alpha_seq.append(one_letter_code[cur_res])
                        alpha_res_n.append(int(cur_res_n))
                        pre_res_n = cur_res_n

        # extract seq from ref
        ref_seq = []
        ref_res_n = []
        pre_res_n = ''
        with open(ref) as f:
            for l in f:
                # chain A of AF is receptor chain 
                if l.startswith('ATOM') and (l[21]=='A'):
                    cur_res = l[17:20]
                    cur_res_n = l[22:26]
                    if cur_res_n != pre_res_n:
                        ref_seq.append(one_letter_code[cur_res])
                        ref_res_n.append(int(cur_res_n))
                        pre_res_n = cur_res_n

        
        # alpha residue number is different from single and complex
        # find the extent of difference
        alpha_seq_str = ''.join(alpha_seq)
        ref_seq_str = ''.join(ref_seq)
        res_diff = alpha_seq_str.find(ref_seq_str[:5])
        
        PRINT_CHECK = True
        if alpha_res_n == cur_res_n:
            continue
            if PRINT_CHECK:
                print('%s ALGINED CORRECTLY'%alpha)
        
        if PRINT_CHECK:
            print(alpha)
            print('    ALPHA_SEQ (%d): '%len(alpha_seq) + ' '.join(list(map(str,alpha_res_n[:10]))))
            print('    SINGLE_SEQ (%d): '%len(ref_seq) + ' '.join(list(map(str,ref_res_n[:10]))))
            print('    RESI_N_MATH: '+str(alpha_res_n==ref_res_n))
            print('    SEQ_MATCH: '+ str( alpha_seq_str == ref_seq_str))
            print('    '+''.join(alpha_seq[:8]))
            print('    '+''.join(ref_seq[:8]))
            print('    NUM_DIFF: %d'%res_diff)
            
        # no need to modify. it is correct numbering
        if res_diff == 0:
            continue

        # comparing alpha_AF_model - single_AF_model
        model_id = int(alpha.split('.')[-2])
        original_alpha = '../receptor/receptor_complex/clean.%s.alpha/relaxed_model_%d_multimer.pdb'%(gpcr_name, model_id)
        
        wrt = []
        with open(original_alpha) as f:
            for l in f:
                if l.startswith('ATOM') and (l[21]=='A'):
                    # renumbering
                    new_res_n = int(l[22:26]) - res_diff
                    
                    # cropping
                    if new_res_n in ref_res_n:
                        new_l = l[:22] + '%4d'%new_res_n + l[26:]
                        wrt.append(new_l)
    
        out_fn = alpha # overwirte
        with open(out_fn,'w') as f:
            f.writelines(wrt)
        
        print('%s has been renumbered'%alpha)


def fix_af_renum():
    """
    correct residue numbers according to ref_renum_pdb
    NEED_TO_BE_FIXED[af_model_name] = diff(af - ref)
    new_af_n(ref_n) = af_n - diff
    """
    NEED_TO_BE_FIXED = {
            'af.S1PR1.LA.alpha':1,
            'af.AA1R.LA.alpha':1,
            'af.SMO.LA.alpha':1,
            'af.ACM1.LA.alpha':1,
            'af.5HT2A.LA.complex':-1
                    }

    for k, v in NEED_TO_BE_FIXED.items():
        if not os.path.exists('./renum_pdb/before.%s.1.pdb'%k):
            renum_pdb_s = glob.glob('./renum_pdb/%s.[0-9].pdb'%k)
            for renum_pdb in renum_pdb_s:
                before_pdb = './renum_pdb/before.%s'%renum_pdb.split('/')[-1]
                os.system('mv %s %s'%(renum_pdb, before_pdb))

        before_fix_s = glob.glob('./renum_pdb/before.%s.[0-9].pdb'%k)
        for before_fix_fn in before_fix_s:
            wrt = []
            with open(before_fix_fn) as f:
                for l in f:
                    if l.startswith('ATOM'):
                        new_res_n = int(l[22:26]) - v
                        wrt.append(l[:22]+'%4d'%new_res_n+l[26:])
                    else:
                        wrt.append(l)

            fixed_pdb = './renum_pdb/%s'%before_fix_fn.split('/')[-1][7:] # remove before
            with open(fixed_pdb,'w') as f:
                f.writelines(wrt)



def crop_domain():
    """
    due to orientation issue,
    TMalign should be applied to cropped model
    gpcr_name, [start_residue, end_residue]
    """
    NEED_TO_BE_CROPPED = {
            'GRM2.LA':[542,813],
            'CASR.LA':[604,860],
            'GABR2.LA':[153,560],
            'GABR2.LI':[151,560],
            'OXYR.LI':[1,297],
            }

    for k, v in NEED_TO_BE_CROPPED.items():
        af_model_s = glob.glob('./renum_pdb/%s.pdb'%k)
        af_model_s += glob.glob('./renum_pdb/af.%s.*.pdb'%k)
        for af_model in af_model_s:
            wrt = []
            st_res_n = v[0]
            ed_res_n = v[1]
            with open(af_model) as f:
                for l in f:
                    if l.startswith('ATOM'):
                        res_n = int(l[22:26])
                        if (st_res_n <= res_n) and (ed_res_n >= res_n):
                            wrt.append(l)
            
            with open(af_model, 'w') as f:
                f.writelines(wrt)


def check_recycle_renum():
    af_s = list(Path('../receptor/recycle').glob('recycle_*/*/unrelaxed_model_*.pdb'))
    renum_s = list(map(str,Path('renum_pdb').glob('re_*.pdb')))
    print(f'Num of afs: {len(af_s)}')
    print(f'Num of renums: {len(renum_s)}')
    n = 0
    for af in af_s:
        recycle = af.parents[1].name.split('_')[-1]
        gpcr = '.'.join(af.parents[0].name.split('.')[1:3])
        rec_type = Help_prep_recycle_get_rec_type(af.parents[0])
        model_id = af.stem.split('_')[2]
        renum_name = f'renum_pdb/re_{recycle}.{gpcr}.{rec_type}.{model_id}.pdb'
        if renum_name not in renum_s:
            n += 1
            print(f'{n}: {af}')


if __name__ == '__main__':
    #main()
    #crop_domain()
    #fix_af_renum()
    #meiler_renum()
    #Lim_renum()
    
    #cross_renum()
    prep_recycle()
    #check_recycle_renum()
