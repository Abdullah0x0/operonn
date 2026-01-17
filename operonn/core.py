"""
Core functionality for OperoNN.

This module provides a clean API for operon prediction and model training.
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from .formats import validate_genes_file, validate_expression_file, validate_operons_file, FormatError


# =============================================================================
# NEURAL NETWORK MODEL
# =============================================================================

class OperonDataset(Dataset):
    """Dataset for operon prediction."""
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class OperonNet(nn.Module):
    """Neural network for operon prediction (64→32→16 architecture)."""
    def __init__(self, input_size=2):
        super(OperonNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.network(x)


# =============================================================================
# FEATURE EXTRACTION
# =============================================================================

def get_adjacent_gene_pairs(genes):
    """
    Get pairs of consecutive genes on the same strand.
    
    Args:
        genes: dict {gene_id: [name, type, start, end, strand, product]}
    
    Returns:
        list: [(gene1_id, gene2_id), ...]
    """
    # Convert to list and sort by position
    gene_list = []
    for gene_id, info in genes.items():
        gene_list.append({
            'id': gene_id,
            'name': info[0],
            'type': info[1],
            'start': int(info[2]),
            'end': int(info[3]),
            'strand': info[4]
        })
    
    gene_list.sort(key=lambda x: x['start'])
    
    # Find adjacent pairs on same strand
    pairs = []
    valid_types = {'CDS', 'gene', 'sRNA', 'SRP', 'ncRNA', 'rRNA', 'tRNA'}
    
    for i in range(len(gene_list) - 1):
        g1, g2 = gene_list[i], gene_list[i + 1]
        
        # Same strand check
        if g1['strand'] == g2['strand']:
            # Valid feature type check
            if g1['type'] in valid_types and g2['type'] in valid_types:
                pairs.append((g1['id'], g2['id']))
    
    return pairs


def extract_features(genes, transcripts, gene_pairs, verbose=True):
    """
    Extract features for gene pairs.
    
    Features:
        1. Intergenic distance (bp)
        2. Expression correlation (Pearson)
    
    Args:
        genes: dict of gene annotations
        transcripts: dict of expression data
        gene_pairs: list of (gene1_id, gene2_id) tuples
    
    Returns:
        tuple: (features_array, processed_pairs)
    """
    features = []
    processed_pairs = []
    
    skipped_missing_expr = 0
    skipped_nan_corr = 0
    
    for pair in gene_pairs:
        gene1_id, gene2_id = pair
        
        # Check if both genes have expression data
        if gene1_id not in transcripts or gene2_id not in transcripts:
            skipped_missing_expr += 1
            continue
        
        # Get gene positions
        g1_info = genes[gene1_id]
        g2_info = genes[gene2_id]
        
        stop1 = int(g1_info[3])  # end of gene 1
        start2 = int(g2_info[2])  # start of gene 2
        end2 = int(g2_info[3])
        
        # Calculate intergenic distance
        if end2 <= stop1:
            # Overlapping genes
            distance = 0
        else:
            distance = start2 - stop1
        
        # Get expression vectors
        expr1 = transcripts[gene1_id][3]
        expr2 = transcripts[gene2_id][3]
        
        # Calculate correlation
        correlation = np.corrcoef(expr1, expr2)[0, 1]
        
        if np.isnan(correlation):
            skipped_nan_corr += 1
            correlation = 0.0
        
        features.append([distance, correlation])
        processed_pairs.append(pair)
    
    if verbose:
        print(f"  Extracted features for {len(features)} gene pairs")
        if skipped_missing_expr > 0:
            print(f"    (skipped {skipped_missing_expr} pairs missing expression data)")
        if skipped_nan_corr > 0:
            print(f"    (set {skipped_nan_corr} NaN correlations to 0)")
    
    return np.array(features), processed_pairs


def create_labeled_pairs(gene_pairs, genes, operons):
    """
    Create labeled gene pairs for training.
    
    Args:
        gene_pairs: list of (gene1_id, gene2_id) tuples
        genes: dict of gene annotations
        operons: list of [[name, start, end, [genes]], ...]
    
    Returns:
        list: [(pair, label), ...] where label is 1 for same-operon, 0 otherwise
    """
    # Build operon membership map
    # gene_id or gene_name -> operon_index
    operon_map = {}
    genes_name_to_id = {info[0]: gid for gid, info in genes.items()}
    
    for operon_idx, operon in enumerate(operons):
        gene_list = operon[3]
        for gene_ref in gene_list:
            # gene_ref might be gene_id or gene_name
            if gene_ref in genes:
                operon_map[gene_ref] = operon_idx
            elif gene_ref in genes_name_to_id:
                operon_map[genes_name_to_id[gene_ref]] = operon_idx
            else:
                operon_map[gene_ref] = operon_idx
    
    # Label pairs
    labeled_pairs = []
    for pair in gene_pairs:
        g1, g2 = pair
        
        if g1 in operon_map and g2 in operon_map:
            if operon_map[g1] == operon_map[g2]:
                labeled_pairs.append((pair, 1))
            else:
                labeled_pairs.append((pair, 0))
        else:
            labeled_pairs.append((pair, 0))
    
    return labeled_pairs


# =============================================================================
# MODEL LOADING
# =============================================================================

def get_bundled_model_path(model_name='ecoli_PRECISE1K_278'):
    """
    Get path to a bundled pre-trained model.
    
    Available models:
        - ecoli_PRECISE1K_278 (default, best general-purpose model)
        - ecoli_k12
        - bsubtilis_GSE160346
        - bsubtilis_GSE160347
        - bsubtilis_GSE160348
    
    Returns:
        tuple: (model_path, scaler_path)
    """
    # Find the package installation directory
    package_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(package_dir, 'trained_models')
    
    model_dir = os.path.join(models_dir, f'{model_name}-model')
    
    if not os.path.exists(model_dir):
        available = [d.replace('-model', '') for d in os.listdir(models_dir) 
                     if d.endswith('-model') and not d.startswith('naive_bayes')]
        raise FileNotFoundError(
            f"Model '{model_name}' not found. Available models: {available}"
        )
    
    model_path = os.path.join(model_dir, 'model.pth')
    scaler_path = os.path.join(model_dir, 'scaler.pkl')
    
    return model_path, scaler_path


def load_pretrained_model(model_path=None, scaler_path=None, model_name=None):
    """
    Load a pre-trained OperoNN model.
    
    Args:
        model_path: Path to model.pth file
        scaler_path: Path to scaler.pkl file
        model_name: Name of bundled model OR path to custom model directory
    
    Returns:
        tuple: (model, scaler)
    """
    if model_path is None:
        if model_name is None:
            model_name = 'ecoli_PRECISE1K_278'
        
        # Check if model_name is actually a path to a custom model directory
        if os.path.isdir(model_name):
            # It's a directory path - use as custom model
            model_path = os.path.join(model_name, 'model.pth')
            scaler_path = os.path.join(model_name, 'scaler.pkl')
            if not os.path.exists(model_path):
                raise ValueError(f"Model file not found: {model_path}")
        elif os.path.isfile(model_name):
            # It's a direct path to model file
            model_path = model_name
        else:
            # Treat as bundled model name
            model_path, scaler_path = get_bundled_model_path(model_name)
    
    if scaler_path is None:
        # Derive from model_path
        model_dir = os.path.dirname(model_path)
        scaler_path = os.path.join(model_dir, 'scaler.pkl')
    
    print(f"Loading model from: {model_path}")
    
    # Register our OperonNet class for safe loading of old models
    # Old models were saved with __main__.OperonNet, we need to make it loadable
    torch.serialization.add_safe_globals([OperonNet])
    
    # Also need to add it to __main__ module so torch can find it
    import __main__
    __main__.OperonNet = OperonNet
    
    # Load checkpoint
    checkpoint = torch.load(model_path, weights_only=False, map_location='cpu')
    
    # Try loading the saved architecture first, fall back to creating new model
    try:
        model = checkpoint['model_architecture']
        model.load_state_dict(checkpoint['model_state_dict'])
    except (KeyError, AttributeError, RuntimeError):
        # Create a new model and load just the weights
        model = OperonNet(input_size=2)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load scaler
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    return model, scaler


# =============================================================================
# MAIN API FUNCTIONS
# =============================================================================

def predict_operons(genes_file, expression_file, model_path=None, scaler_path=None,
                    model_name=None, threshold=0.5, output_file=None):
    """
    Predict operons for a new organism.
    
    Args:
        genes_file: Path to gene annotations (CSV or GFF)
        expression_file: Path to expression matrix
        model_path: Path to custom model (optional)
        scaler_path: Path to custom scaler (optional)
        model_name: Name of bundled model to use (default: ecoli_PRECISE1K_278)
        threshold: Prediction threshold (default: 0.5)
        output_file: Path to save predictions (optional)
    
    Returns:
        pd.DataFrame: Predictions with columns [gene1, gene2, probability, prediction]
    """
    print("\n" + "="*60)
    print("OperoNN Operon Prediction")
    print("="*60)
    
    # Load model
    print("\nLoading model...")
    model, scaler = load_pretrained_model(model_path, scaler_path, model_name)
    
    # Load input data
    print("\nLoading input files...")
    genes, genes2 = validate_genes_file(genes_file)
    transcripts = validate_expression_file(expression_file, genes2)
    
    # Get gene pairs
    print("\nFinding adjacent gene pairs...")
    gene_pairs = get_adjacent_gene_pairs(genes)
    print(f"  Found {len(gene_pairs)} adjacent gene pairs")
    
    # Extract features
    print("\nExtracting features...")
    features, processed_pairs = extract_features(genes, transcripts, gene_pairs)
    
    if len(features) == 0:
        raise ValueError("No valid gene pairs found. Check that gene IDs match between files.")
    
    # Scale features
    X_scaled = scaler.transform(features)
    
    # Run predictions
    print("\nRunning predictions...")
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    model.to(device)
    model.eval()
    
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_scaled).to(device)
        probabilities = model(X_tensor).squeeze().cpu().numpy()
    
    # Create results DataFrame
    results = []
    for i, (pair, prob) in enumerate(zip(processed_pairs, probabilities)):
        gene1_id, gene2_id = pair
        gene1_name = genes[gene1_id][0]
        gene2_name = genes[gene2_id][0]
        
        results.append({
            'gene1_id': gene1_id,
            'gene1_name': gene1_name,
            'gene2_id': gene2_id,
            'gene2_name': gene2_name,
            'distance_bp': int(features[i][0]),
            'correlation': round(features[i][1], 4),
            'probability': round(float(prob), 4),
            'prediction': 'same_operon' if prob >= threshold else 'different_operon'
        })
    
    df = pd.DataFrame(results)
    
    # Summary statistics
    n_same = (df['prediction'] == 'same_operon').sum()
    n_diff = (df['prediction'] == 'different_operon').sum()
    
    print(f"\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(f"Total gene pairs analyzed: {len(df)}")
    print(f"Predicted same operon:     {n_same} ({100*n_same/len(df):.1f}%)")
    print(f"Predicted different:       {n_diff} ({100*n_diff/len(df):.1f}%)")
    
    # Save results
    if output_file:
        df.to_csv(output_file, index=False)
        print(f"\nResults saved to: {output_file}")
    
    return df


def train_model(genes_file, expression_file, operons_file, output_dir,
                epochs=100, batch_size=64, test_size=0.2, random_state=42):
    """
    Train a new OperoNN model.
    
    Args:
        genes_file: Path to gene annotations
        expression_file: Path to expression matrix
        operons_file: Path to known operon annotations
        output_dir: Directory to save trained model
        epochs: Number of training epochs (default: 100)
        batch_size: Training batch size (default: 64)
        test_size: Fraction for test set (default: 0.2)
        random_state: Random seed (default: 42)
    
    Returns:
        dict: Training metrics
    """
    print("\n" + "="*60)
    print("OperoNN Model Training")
    print("="*60)
    
    # Load input data
    print("\nLoading input files...")
    genes, genes2 = validate_genes_file(genes_file)
    transcripts = validate_expression_file(expression_file, genes2)
    operons = validate_operons_file(operons_file)
    
    # Get gene pairs
    print("\nCreating training examples...")
    gene_pairs = get_adjacent_gene_pairs(genes)
    labeled_pairs = create_labeled_pairs(gene_pairs, genes, operons)
    
    # Extract features
    print("\nExtracting features...")
    pair_list = [p[0] for p in labeled_pairs]
    labels_list = [p[1] for p in labeled_pairs]
    
    features, processed_pairs = extract_features(genes, transcripts, pair_list, verbose=True)
    
    # Align labels with processed pairs
    pair_to_label = {pair: label for pair, label in labeled_pairs}
    labels = np.array([pair_to_label[p] for p in processed_pairs])
    
    print(f"\n  Total examples: {len(features)}")
    print(f"  Positive (same operon): {sum(labels)}")
    print(f"  Negative (different): {len(labels) - sum(labels)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=test_size, random_state=random_state, stratify=labels
    )
    
    print(f"\n  Training set: {len(X_train)} examples")
    print(f"  Test set: {len(X_test)} examples")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create datasets
    train_dataset = OperonDataset(X_train_scaled, y_train)
    test_dataset = OperonDataset(X_test_scaled, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"\nUsing device: {device}")
    
    model = OperonNet(input_size=2).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    print("\nTraining...")
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_features, batch_labels in train_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_features).squeeze()
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 20 == 0:
            print(f'  Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}')
    
    # Evaluate
    print("\nEvaluating...")
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch_features, batch_labels in test_loader:
            batch_features = batch_features.to(device)
            outputs = model(batch_features).squeeze()
            predictions = (outputs > 0.5).float()
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(batch_labels.numpy())
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(all_labels, all_predictions),
        'f1_score': f1_score(all_labels, all_predictions),
        'precision': precision_score(all_labels, all_predictions),
        'recall': recall_score(all_labels, all_predictions)
    }
    
    print(f"\n" + "="*60)
    print("TRAINING RESULTS")
    print("="*60)
    print(f"Accuracy:  {metrics['accuracy']:.3f} ({metrics['accuracy']*100:.1f}%)")
    print(f"F1 Score:  {metrics['f1_score']:.3f}")
    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Recall:    {metrics['recall']:.3f}")
    
    # Save model
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, 'model.pth')
    scaler_path = os.path.join(output_dir, 'scaler.pkl')
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_architecture': model,
        'accuracy': metrics['accuracy'],
        'f1_score': metrics['f1_score'],
        'precision': metrics['precision'],
        'recall': metrics['recall'],
        'features_used': ['gene_distance', 'expression_correlation']
    }, model_path)
    
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    print(f"\nModel saved to: {model_path}")
    print(f"Scaler saved to: {scaler_path}")
    
    return metrics

