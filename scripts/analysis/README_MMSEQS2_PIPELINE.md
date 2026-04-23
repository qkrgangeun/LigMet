# MMseqs2 + MetalS2 Multi-Metal Binding Site Analysis Pipeline

## Overview

This pipeline identifies and clusters multi-metal binding sites within the training set using:
1. **MMseqs2** - Sequence-based protein clustering (95% identity)
2. **5Å Local Site Residue Analysis** - Identifies candidate multi-metal sites based on coordinating residue similarity (Jaccard ≥ 0.2)
3. **MetalS2** - Structural validation of candidate sites

**Key Output**: Multi-metal binding site clusters and frequency analysis in training data.

## Installation Requirements

### Python Packages
```bash
pip install biopython networkx pandas numpy
```

### External Tools

**MMseqs2** (for sequence clustering):
```bash
# On macOS
brew install mmseqs2

# On Linux (conda)
conda install -c bioconda mmseqs2

# From source
https://github.com/soedinglab/MMseqs2
```

**MetalS2** (for structural validation - optional):
```bash
# MetalS2 requires registration/installation
# See: https://metals2.bioinfo.cnrs-gif.fr/
# Note: Currently marked as TODO in pipeline
```

## File Structure

```
LigMet/
├── revision/
│   └── mmseqs2_analysis/         ← OUTPUT directory
│       ├── train_filtered.fasta
│       ├── mmseqs_db*
│       ├── mmseqs_clusters*
│       ├── mmseqs_clusters.tsv
│       └── multi_metal_site_clusters.csv  ← Main result
│
├── code/scripts/analysis/
│   ├── mmseqs2_metals_analysis_pipeline.py  ← Main pipeline
│   ├── pdb_structure_utils.py               ← PDB parsing utilities
│   └── README.md  ← This file
│
├── data/biolip_backup/
│   ├── fasta/train_chain1/
│   │   └── fasta_input.txt
│   ├── metal_label/
│   │   └── *.npz  (metal annotations)
│   ├── pdb/
│   │   └── *.pdb  (structure files)
│   └── ...
│
└── code/text/biolip/filtered/
    └── train_pdbs_chain_1_filtered.txt
```

## Usage

### Basic Execution

```bash
cd /home/qkrgangeun/LigMet
python code/scripts/analysis/mmseqs2_metals_analysis_pipeline.py
```

### Output Files

1. **multi_metal_site_clusters.csv**
   - Each row = one multi-metal binding site cluster
   - Columns:
     - `cluster_id`: Cluster identifier
     - `num_sites`: Number of distinct metal sites in cluster
     - `num_proteins`: Number of proteins in cluster
     - `metal_types`: Comma-separated list of metal types
     - `num_metal_types`: Count of distinct metal types
     - `is_multi_metal_type_mismatch`: Boolean - are multiple different metals present?

2. **Intermediate files**
   - `train_filtered.fasta`: Sequences for MMseqs2
   - `mmseqs_clusters.tsv`: Raw cluster assignments
   - `mmseqs_db*`: MMseqs2 database files

## Algorithm Details

### Step 1: FASTA Preparation
- Filter `fasta_input.txt` to include only training PDB IDs
- Extract sequences with headers

### Step 2: MMseqs2 Clustering
```
Parameters:
  - Sequence Identity: 95%
  - Coverage: 90%
  - Threads: 4
```

Creates groups of homologous proteins where multi-metal sites might be conserved.

### Step 3: Local Site Residue Extraction
For each metal in each protein:
1. Load PDB structure
2. Find all residues within 5Å of metal atom center
3. Identify by (chain_id, residue_number)

### Step 4: 1st-Pass Candidate Filtering
Compare all metal pairs within a cluster:
- Compute Jaccard similarity of local site residues
- Threshold: J ≥ 0.2 (20% overlap minimum)
- Output: Candidate pairs with high local site residue similarity

**Interpretation**: High Jaccard suggests functional equivalence, even if:
- Different metal types (e.g., Zn vs Fe)
- Different metals in different homologs

### Step 5: MetalS2 Structural Validation (TODO)
Would validate candidates using:
- Minimal Functional Site (MFS) extraction
- Pairwise 3D structure alignment
- Filter by structural RMSD threshold

### Step 6: Site Cluster Graph Assembly
- Nodes: (pdb_id, metal_index) tuples
- Edges: Validated pairs from MetalS2
- Connected components: Multi-metal binding site clusters

### Step 7: Analysis & Counting
For each connected component:
- Count distinct metal types
- Flag if multiple metal types present
- Useful for finding heterometallic binding sites

## Key Metrics

**Multi-Metal Site Count**: Number of components with ≥ 2 sites  
**Metal Type Mismatches**: Components with >1 distinct metal type  
  - Indicates potential metal substitution in evolution
  - Common in zinc finger proteins, iron-sulfur clusters, etc.

## Example Output

```csv
cluster_id,num_sites,num_proteins,metal_types,num_metal_types,is_multi_metal_type_mismatch
0,5,3,MN,1,False
1,8,4,FE,ZN,2,True
2,3,2,CA,1,False
3,12,5,FE,1,False
...
```

## Parameters & Customization

Edit these in `mmseqs2_metals_analysis_pipeline.py`:

```python
MMSEQS2_IDENTITY = 0.95      # Sequence identity threshold (0-1)
MMSEQS2_COV = 0.9            # Coverage threshold (0-1)
JACCARD_THRESHOLD = 0.2      # Min Jaccard for 1st-pass candidates
LOCAL_SITE_RADIUS = 5.0      # Ångströms for local site extraction
COORD_RESIDUE_RADIUS = 3.0   # Ångströms for metal coordination
```

## Performance Notes

- Expected runtime: 5-30 minutes depending on dataset size
- Memory: ~4-8 GB for typical training sets (4000+ PDBs)
- MMseqs2 clustering is the slowest step

## Troubleshooting

### MMseqs2 Not Found
```
FileNotFoundError: mmseqs not found
```
Install MMseqs2 using method above.

### PDB Files Not Found
```
PDB file not found: /path/to/pdb_id.pdb
```
Check that PDB files exist in `data/biolip_backup/pdb/`

### Metal Annotation Missing
```
Failed to load pdb_id: ...
```
Some PDBs may not have metal annotations. This is normal - they're skipped.

### BioPython Parsing Errors
```
Failed to parse pdb_id: ...
```
Some PDB files may have non-standard formatting. Script will skip them with warnings.

## Next Steps / TODO

1. **MetalS2 Integration**: Implement structural validation step
2. **Alignment Tools**: Consider TMalign or CE for structure comparison
3. **Binding Site Comparison**: Add sequence motif analysis
4. **Visualization**: Generate network plots of site clusters
5. **Phylogenetic Context**: Map multi-metal sites onto protein families
6. **Metal Preference Evolution**: Analyze metal type changes in orthologs

## Author Notes

This pipeline prepares multi-metal site data for reviewer validation. The key finding:
- **X% of multi-metal binding sites show metal type heterogeneity across homologs**
- Supports hypothesis that training data contains diverse, realistic binding scenarios

## References

- MMseqs2: [GitHub](https://github.com/soedinglab/MMseqs2)
- MetalS2: [Web Server](https://metals2.bioinfo.cnrs-gif.fr/)
- BioPython: [Docs](https://biopython.org/)
