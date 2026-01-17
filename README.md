# OperoNN: A Neural Network Approach to Bacterial Operon Prediction

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**OperoNN** is a deep learning tool for predicting bacterial operons using RNA-seq expression data and genomic features. It uses a simple but effective neural network architecture trained on well-characterized organisms to predict operon structure in new bacterial species.

## Features

- **Predict operons** for any bacterial genome with RNA-seq data
- **Cross-species transfer** — models trained on E. coli work well on other bacteria
- **Only two features** — intergenic distance + expression correlation
- **Fast inference** — predict thousands of gene pairs in seconds
- **High accuracy** — 81.6% accuracy, 0.855 F1 score on benchmark datasets

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/Abdullah0x0/operonn.git
cd operonn

# Install the package
pip install -e .

# Verify installation
operonn --help
```

### Available Commands

```
usage: operonn [-h] {predict,train,formats,models} ...

OperoNN: A Neural Network Approach to Bacterial Operon Prediction

Commands:
    predict     Predict operons using a pre-trained model
    train       Train a new model on annotated data
    formats     Show input file format examples
    models      List available pre-trained models

Examples:
  operonn predict --genes genes.csv --expression counts.csv --output results.csv
  operonn train --genes genes.csv --expression counts.csv --operons operons.csv --output my_model/
  operonn formats
```

### Try with Example Data

Test the installation using the included example dataset:

```bash
# Step 1: Run prediction on example data
operonn predict \
    --genes example/genes.csv \
    --expression example/expression.csv \
    --output example/predictions.csv

# Step 2: View results
cat example/predictions.csv
```

**Expected output:**
```
gene1_id,gene1_name,gene2_id,gene2_name,distance_bp,correlation,probability,prediction
gene-001,geneA,gene-002,geneB,50,0.9989,0.948,same_operon
gene-002,geneB,gene-003,geneC,50,0.9957,0.9469,same_operon
gene-003,geneC,gene-004,geneD,100,-0.8999,0.1324,different_operon
...
```

### Basic Usage

```bash
# Predict operons on your own data
operonn predict \
    --genes my_genes.csv \
    --expression my_counts.csv \
    --output predicted_operons.csv

# See available pre-trained models with their performance metrics
operonn models

# View input format examples
operonn formats
```

## Input File Formats

### 1. Gene Annotations (`--genes`)

CSV file with gene coordinates:

```csv
gene_id,name,start,end,strand
gene-b0001,thrL,190,255,+
gene-b0002,thrA,337,2799,+
gene-b0003,thrB,2801,3733,+
```

Also supports standard **GFF3 format**.

### 2. Expression Data (`--expression`)

CSV matrix with genes as rows, samples as columns:

```csv
,sample_1,sample_2,sample_3,sample_4
b0001,100.5,95.2,110.3,98.7
b0002,50.1,48.9,52.0,51.5
b0003,200.0,210.5,195.8,205.2
```

Also supports **featureCounts output** (tab-separated).

### 3. Operon Annotations (`--operons`) — For Training Only

CSV file with operon-to-gene mappings:

```csv
operon,gene,start,end
thrLABC,gene-b0001,190,255
thrLABC,gene-b0002,337,2799
thrLABC,gene-b0003,2801,3733
lacZYA,gene-b0344,361249,364285
```

## Commands

### Predict Operons

Use a pre-trained model to predict operons on new data:

```bash
operonn predict \
    --genes genes.csv \
    --expression counts.csv \
    --output predictions.csv \
    --model ecoli_PRECISE1K_278  # optional, this is the default
```

**Options:**
- `--genes`, `-g`: Gene annotations file - CSV or GFF3 format (required)
- `--expression`, `-e`: Expression matrix - CSV or featureCounts output (required)
- `--output`, `-o`: Output CSV path (required)
- `--model`, `-m`: Model to use - either a bundled model name (e.g., `ecoli_k12`) or path to custom model directory (default: `ecoli_PRECISE1K_278`)
- `--threshold`, `-t`: Prediction threshold (default: 0.5)

### Train Custom Model

Train a model on your own annotated data:

```bash
operonn train \
    --genes genes.csv \
    --expression counts.txt \
    --operons known_operons.csv \
    --output my_model/
```

**Options:**
- `--genes`, `-g`: Gene annotations file - CSV or GFF3 format (required)
- `--expression`, `-e`: Expression matrix - CSV or featureCounts output (required)
- `--operons`, `-p`: Known operon annotations - CSV format (required)
- `--output`, `-o`: Output directory for model (required)
- `--epochs`: Training epochs (default: 100)
- `--batch-size`: Batch size (default: 64)

### Use Custom Model

After training, use your custom model for prediction:

```bash
operonn predict \
    --genes new_organism_genes.csv \
    --expression new_organism_counts.csv \
    --model ./my_model \
    --output predictions.csv
```

### List Models

View all available pre-trained models:

```bash
operonn models
```

### Show Formats

Display input file format examples:

```bash
operonn formats
```

## Output Format

The prediction output is a CSV file with the following columns:

| Column | Description |
|--------|-------------|
| `gene1_id` | First gene identifier |
| `gene1_name` | First gene name |
| `gene2_id` | Second gene identifier |
| `gene2_name` | Second gene name |
| `distance_bp` | Intergenic distance in base pairs |
| `correlation` | Expression correlation (Pearson) |
| `probability` | Model confidence (0-1) |
| `prediction` | `same_operon` or `different_operon` |

## Pre-trained Models

Run `operonn models` to see all available models with their metrics:

| Model | Organism | Accuracy | F1 Score |
|-------|----------|----------|----------|
| `ecoli_PRECISE1K_278` * | E. coli K-12 | 0.868 | 0.890 |
| `ecoli_k12` | E. coli K-12 | 0.849 | 0.870 |
| `bsubtilis_GSE160346` | B. subtilis | 0.806 | 0.859 |
| `bsubtilis_GSE160347` | B. subtilis | 0.792 | 0.849 |
| `bsubtilis_GSE160348` | B. subtilis | 0.802 | 0.856 |

\* = Recommended default model (used when `--model` is not specified)

## Python API

You can also use OperoNN programmatically:

```python
from operonn import predict_operons, train_model

# Predict operons
results = predict_operons(
    genes_file='genes.csv',
    expression_file='counts.csv',
    output_file='predictions.csv'
)

# Access results as DataFrame
print(results.head())

# Train a custom model
metrics = train_model(
    genes_file='genes.csv',
    expression_file='counts.csv',
    operons_file='known_operons.csv',
    output_dir='my_model/'
)
```

## How It Works

OperoNN uses a simple feedforward neural network (64→32→16 architecture) trained on two features:

1. **Intergenic Distance**: The number of base pairs between adjacent genes
2. **Expression Correlation**: Pearson correlation of expression profiles across RNA-seq samples

Genes in the same operon typically have:
- Short intergenic distances (often overlapping or ≤50bp)
- Highly correlated expression patterns (co-transcribed)

The model learns the optimal decision boundary in this 2D feature space.

## Citation

If you use OperoNN in your research, please cite:

```bibtex
@software{operonn2026,
  title={OperoNN: A Neural Network Approach to Bacterial Operon Prediction},
  author={Choudhry, Abdullah Tariq and Orwin, Paul M. and Olivieri, Julia Eve},
  year={2026},
  url={https://github.com/Abdullah0x0/operonn}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- E. coli operon annotations from [RegulonDB](http://regulondb.ccg.unam.mx/)
- B. subtilis operon annotations from [DBTBS](https://dbtbs.hgc.jp/) and [BSGatlas](https://rth.dk/resources/bsgatlas/)
- Expression data from [NCBI GEO](https://www.ncbi.nlm.nih.gov/geo/)
- PRECISE1K dataset from [Palsson Lab](https://github.com/SBRG/precise1k)

