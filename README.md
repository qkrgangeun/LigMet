# LigMet: Ligand-Protein Binding Prediction

LigMet is a machine learning framework for ligand-protein binding prediction using deep learning and traditional ML approaches.

## Features

- **Multiple Model Architectures**: Deep Learning (DL) and Random Forest implementations
- **PyTorch Lightning Integration**: Streamlined training and evaluation pipeline
- **Flexible Dataset Support**: Built-in dataset and featurizer modules
- **Parameter Management**: Pre-trained parameters available via Hugging Face

## Installation

### Prerequisites

- Python 3.9 or higher
- CUDA 11.3 (for GPU support)
- Conda package manager

### Step 1: Clone the Repository

```bash
git clone https://github.com/momil/LigMet.git
cd LigMet/code
```

### Step 2: Create and Activate Conda Environment

```bash
conda env create -f environment.yml
conda activate promet
```

This creates a `promet` environment with all required dependencies including:
- PyTorch 1.11
- DGL (Deep Graph Library) with CUDA 11.3 support
- RDKit for molecular processing
- PyTorch Lightning for training
- And more scientific computing libraries

### Step 3: Install LigMet Package

```bash
pip install -e .
```

### Step 4: Download Pre-trained Parameters (Optional)

Pre-trained model parameters are hosted on Hugging Face:

```bash
# Using huggingface-hub CLI
huggingface-cli download momil/PrOMet param --repo-type model --local-dir ./param

# Or using Python
from huggingface_hub import snapshot_download
snapshot_download(repo_id="momil/PrOMet", repo_type="model", local_dir="./param")
```

The `param` folder contains:
- `dl.ckpt`: Pre-trained deep learning model checkpoint
- `randomforest_param`: Random Forest model parameters

## Quick Start

### Basic Usage with CLI

```bash
# Training with default config
python scripts/run.py

# Training with custom config
python scripts/run.py --config config.yaml

# Prediction mode
python scripts/run.py predict --ckpt_path ./checkpoints/model.ckpt
```

### Using the Pipeline in Python

```python
from ligmet.pipeline import LigMetPipeline
from ligmet.dataset import LigMetDataset

# Initialize dataset
dataset = LigMetDataset(data_path="./dataset")

# Create pipeline
pipeline = LigMetPipeline(model_type="dl")  # or "randomforest"

# Make predictions
predictions = pipeline.predict(dataset)
```

### Using Pre-trained Models

```python
from ligmet.pl import LigMetTestModule
import torch

# Load deep learning model
model = LigMetTestModule.load_from_checkpoint("./param/dl.ckpt")
model.eval()

# Load random forest model
import joblib
rf_model = joblib.load("./param/randomforest_param")
```

## Project Structure

```
LigMet/
├── code/
│   ├── src/ligmet/
│   │   ├── models.py           # Model architectures
│   │   ├── dataset.py          # Dataset classes
│   │   ├── featurizer.py       # Feature engineering
│   │   ├── pipeline.py         # Inference pipeline
│   │   ├── pl.py               # PyTorch Lightning modules
│   │   └── utils/              # Utility functions
│   ├── scripts/
│   │   ├── run.py              # Main training script
│   │   ├── analyze_aligned_pdb.py
│   │   └── benchmark/          # Benchmarking scripts
│   ├── config.yaml             # Configuration file
│   ├── environment.yml         # Conda environment
│   └── pyproject.toml          # Package setup
├── dataset/                    # Data directory
└── figure/                     # Results and figures
```

## Configuration

Configuration is managed via `config.yaml`. Key sections:

```yaml
seed_everything: 42

trainer:
  accelerator: auto              # GPU/CPU selection
  strategy: auto                 # Distributed training strategy
  devices: auto                  # Number of devices
  max_epochs: 100               # Training epochs
  precision: 32-true            # Floating point precision
```

Modify these settings before training:

```bash
python scripts/run.py --config custom_config.yaml
```

## Model Types

### Deep Learning Model
- Trained checkpoint: `param/dl.ckpt`
- Based on PyTorch Lightning
- Supports distributed training

### Random Forest Model
- Trained parameters: `param/randomforest_param`
- Scikit-learn implementation
- Good for baseline comparisons

## Requirements

Key dependencies:
- `pytorch==1.11`
- `dgl-cuda11.3==0.9.1` (Graph neural networks)
- `e3nn==0.5` (Equivariant neural networks)
- `rdkit==2023.9` (Molecular processing)
- `pytorch-lightning` (Training framework)
- `scikit-learn` (Traditional ML)
- `freesasa` (Solvent accessibility calculations)

See `environment.yml` for complete list.

## Usage Examples

### Example 1: Analyze PDB Files

```bash
python src/ligmet/analysis/analyze_aligned_pdb_example.py
```

### Example 2: Benchmark Performance

```bash
python scripts/benchmark/run_benchmark.py
```

### Example 3: Custom Dataset Training

```python
from ligmet.dataset import LigMetDataset
from ligmet.pl import LigMetTestModule, LigMetTestDataModule
from lightning.pytorch import Trainer

# Prepare data
dataset = LigMetDataset(split="train")
datamodule = LigMetTestDataModule(dataset)

# Create and train model
model = LigMetTestModule()
trainer = Trainer(max_epochs=100)
trainer.fit(model, datamodule)
```

## GPU Support

The environment is configured for CUDA 11.3. For different CUDA versions:

```bash
# CUDA 12.x
conda install pytorch::pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
conda install dgl-cuda12 -c dglteam
```

## Troubleshooting

**CUDA Out of Memory**: Reduce batch size in config or use `strategy: ddp`

**Import Errors**: Ensure package is installed with `pip install -e .`

**Missing `param` folder**: Download using Hugging Face Hub (see Installation Step 4)

## Citation

If you use LigMet, please cite:

```bibtex
@software{ligmet2024,
  author = {MOMIL},
  title = {LigMet: Ligand-Protein Binding Prediction},
  year = {2024},
  url = {https://github.com/momil/LigMet}
}
```

## Model Repository

Pre-trained models are available at: https://huggingface.co/momil/PrOMet

## License

[Add your license here]

## Support

For issues, questions, or contributions, please open an issue on the GitHub repository.
