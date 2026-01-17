#!/usr/bin/env python3
"""
OperoNN Command Line Interface

Usage:
    operonn predict --genes GENES --expression EXPR --output OUT [--model MODEL]
    operonn train --genes GENES --expression EXPR --operons OPS --output DIR
    operonn formats  # Show input format examples
"""

import argparse
import sys
import os


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog='operonn',
        description='OperoNN: A Neural Network Approach to Bacterial Operon Prediction',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Predict operons on new data using pre-trained model
  operonn predict --genes genes.csv --expression counts.csv --output results.csv

  # Predict using a specific bundled model
  operonn predict --genes genes.csv --expression counts.csv --output results.csv --model ecoli_k12

  # Train a custom model
  operonn train --genes genes.csv --expression counts.csv --operons known_operons.csv --output my_model/

  # Show input file format examples
  operonn formats
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # =========================================================================
    # PREDICT command
    # =========================================================================
    predict_parser = subparsers.add_parser(
        'predict',
        help='Predict operons using a pre-trained model',
        description='Predict operons for a new organism using expression data and a pre-trained model.'
    )
    
    predict_parser.add_argument(
        '--genes', '-g',
        required=True,
        help='Path to gene annotations file (CSV or GFF3 format)'
    )
    
    predict_parser.add_argument(
        '--expression', '-e',
        required=True,
        help='Path to expression matrix file (CSV, genes × samples)'
    )
    
    predict_parser.add_argument(
        '--output', '-o',
        required=True,
        help='Path for output predictions CSV'
    )
    
    predict_parser.add_argument(
        '--model', '-m',
        default=None,
        help='Name of bundled model to use (default: ecoli_PRECISE1K_278). '
             'Options: ecoli_PRECISE1K_278, ecoli_k12, bsubtilis_GSE160346, etc.'
    )
    
    predict_parser.add_argument(
        '--model-path',
        default=None,
        help='Path to custom model.pth file (overrides --model)'
    )
    
    predict_parser.add_argument(
        '--scaler-path',
        default=None,
        help='Path to custom scaler.pkl file (required if --model-path is used)'
    )
    
    predict_parser.add_argument(
        '--threshold', '-t',
        type=float,
        default=0.5,
        help='Prediction threshold (default: 0.5). Lower = more operon predictions.'
    )
    
    # =========================================================================
    # TRAIN command
    # =========================================================================
    train_parser = subparsers.add_parser(
        'train',
        help='Train a new model on annotated data',
        description='Train a custom OperoNN model using annotated operon data.'
    )
    
    train_parser.add_argument(
        '--genes', '-g',
        required=True,
        help='Path to gene annotations file (CSV or GFF3 format)'
    )
    
    train_parser.add_argument(
        '--expression', '-e',
        required=True,
        help='Path to expression matrix file (CSV, genes × samples)'
    )
    
    train_parser.add_argument(
        '--operons', '-p',
        required=True,
        help='Path to known operon annotations (CSV format)'
    )
    
    train_parser.add_argument(
        '--output', '-o',
        required=True,
        help='Directory to save trained model'
    )
    
    train_parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of training epochs (default: 100)'
    )
    
    train_parser.add_argument(
        '--batch-size',
        type=int,
        default=64,
        help='Training batch size (default: 64)'
    )
    
    train_parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Fraction of data for testing (default: 0.2)'
    )
    
    train_parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    # =========================================================================
    # FORMATS command
    # =========================================================================
    formats_parser = subparsers.add_parser(
        'formats',
        help='Show input file format examples',
        description='Display expected input file formats with examples.'
    )
    
    # =========================================================================
    # MODELS command
    # =========================================================================
    models_parser = subparsers.add_parser(
        'models',
        help='List available pre-trained models',
        description='List all bundled pre-trained models.'
    )
    
    # =========================================================================
    # Parse and execute
    # =========================================================================
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        print("\nQuick start:")
        print("   operonn predict --genes genes.csv --expression counts.csv --output results.csv")
        print("   operonn formats  # to see input file format examples")
        sys.exit(0)
    
    if args.command == 'predict':
        run_predict(args)
    elif args.command == 'train':
        run_train(args)
    elif args.command == 'formats':
        run_formats()
    elif args.command == 'models':
        run_models()


def run_predict(args):
    """Run prediction command."""
    from .core import predict_operons
    
    print("""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                              OperoNN v1.0.0                                    ║
║         A Neural Network Approach to Bacterial Operon Prediction               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    try:
        results = predict_operons(
            genes_file=args.genes,
            expression_file=args.expression,
            model_path=args.model_path,
            scaler_path=args.scaler_path,
            model_name=args.model,
            threshold=args.threshold,
            output_file=args.output
        )
        
        print("\nPrediction complete!")
        
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


def run_train(args):
    """Run training command."""
    from .core import train_model
    
    print("""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                           OperoNN Model Training                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    try:
        metrics = train_model(
            genes_file=args.genes,
            expression_file=args.expression,
            operons_file=args.operons,
            output_dir=args.output,
            epochs=args.epochs,
            batch_size=args.batch_size,
            test_size=args.test_size,
            random_state=args.seed
        )
        
        print("\nTraining complete!")
        
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


def run_formats():
    """Show format examples."""
    from .formats import print_format_examples
    print_format_examples()


def run_models():
    """List available models."""
    package_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(package_dir, 'trained_models')
    
    print("""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                        Available Pre-trained Models                            ║
╚═══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    if not os.path.exists(models_dir):
        print("  No bundled models found.")
        print(f"  Expected location: {models_dir}")
        return
    
    models = []
    for name in sorted(os.listdir(models_dir)):
        if name.endswith('-model') and not name.startswith('naive_bayes'):
            model_name = name.replace('-model', '')
            model_dir = os.path.join(models_dir, name)
            model_pth = os.path.join(model_dir, 'model.pth')
            
            if os.path.exists(model_pth):
                # Try to load metrics
                try:
                    import torch
                    from .core import OperonNet
                    # Register class for loading old models
                    torch.serialization.add_safe_globals([OperonNet])
                    import __main__
                    __main__.OperonNet = OperonNet
                    checkpoint = torch.load(model_pth, weights_only=False, map_location='cpu')
                    acc = checkpoint.get('accuracy', 'N/A')
                    f1 = checkpoint.get('f1_score', 'N/A')
                    if isinstance(acc, float):
                        acc = f"{acc:.3f}"
                    if isinstance(f1, float):
                        f1 = f"{f1:.3f}"
                except Exception:
                    acc, f1 = 'N/A', 'N/A'
                
                models.append({
                    'name': model_name,
                    'accuracy': acc,
                    'f1': f1
                })
    
    if not models:
        print("  No valid models found.")
        return
    
    print(f"  {'Model Name':<30} {'Accuracy':<12} {'F1 Score':<12}")
    print(f"  {'-'*30} {'-'*12} {'-'*12}")
    
    for m in models:
        marker = " *" if m['name'] == 'ecoli_PRECISE1K_278' else ""
        print(f"  {m['name']:<30} {m['accuracy']:<12} {m['f1']:<12}{marker}")
    
    print("\n  * = Recommended default model")
    print("\n  Usage: operonn predict --model MODEL_NAME ...")


if __name__ == '__main__':
    main()

