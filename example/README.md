# Example Data

This directory contains sample data to test OperoNN.

## Files

- `genes.csv` - 10 example genes with coordinates
- `expression.csv` - Expression matrix (10 genes × 10 samples)

## Expected Results

The sample data is designed to produce predictable results:

| Gene Pair | Expected Prediction | Why |
|-----------|---------------------|-----|
| geneA-geneB | same_operon | Close (50bp), high correlation |
| geneB-geneC | same_operon | Close (50bp), high correlation |
| geneC-geneD | different_operon | Far (100bp), low correlation |
| geneE-geneF | same_operon | Close (100bp), high correlation |
| geneF-geneG | different_operon | Close but low correlation |
| geneH-geneI | same_operon | Close (50bp), high correlation |
| geneI-geneJ | different_operon | Close but low correlation |

## Usage

```bash
cd operonn
operonn predict \
    --genes example/genes.csv \
    --expression example/expression.csv \
    --output example/predictions.csv

# View results
cat example/predictions.csv
```

