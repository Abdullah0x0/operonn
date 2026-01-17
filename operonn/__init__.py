"""
OperoNN: A Neural Network Approach to Bacterial Operon Prediction

A tool for predicting bacterial operons using RNA-seq expression data 
and genomic features.
"""

__version__ = "1.0.0"
__author__ = "Abdullah Tariq Choudhry"

from .core import predict_operons, train_model, load_pretrained_model
from .formats import validate_genes_file, validate_expression_file, validate_operons_file

__all__ = [
    "predict_operons",
    "train_model", 
    "load_pretrained_model",
    "validate_genes_file",
    "validate_expression_file",
    "validate_operons_file",
]

