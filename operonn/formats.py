"""
Input format validation and parsing for OperoNN.

Supported formats:
- genes.csv: gene annotations (gene_id, name, start, end, strand)
- expression.csv: expression matrix (genes × samples)
- operons.csv: operon annotations (operon_id, gene_id) - optional for training
"""

import pandas as pd
import os
import sys


class FormatError(Exception):
    """Custom exception for format validation errors."""
    pass


def validate_genes_file(filepath):
    """
    Validate and load a genes annotation file.
    
    Expected format (CSV):
        gene_id,name,start,end,strand
        gene-b0001,thrL,190,255,+
        gene-b0002,thrA,337,2799,+
    
    Also supports GFF3 format.
    
    Returns:
        tuple: (genes_dict, genes_name_to_id_dict)
    """
    if not os.path.exists(filepath):
        raise FormatError(f"Gene file not found: {filepath}")
    
    # Check file extension
    ext = os.path.splitext(filepath)[1].lower()
    
    if ext == '.gff' or ext == '.gff3':
        return _parse_gff(filepath)
    elif ext == '.csv':
        return _parse_genes_csv(filepath)
    else:
        # Try to auto-detect format
        with open(filepath, 'r') as f:
            first_line = f.readline().strip()
        
        if first_line.startswith('##gff-version') or '\t' in first_line and first_line.count('\t') >= 7:
            return _parse_gff(filepath)
        else:
            return _parse_genes_csv(filepath)


def _parse_genes_csv(filepath):
    """Parse genes from CSV format."""
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        raise FormatError(f"Failed to read genes CSV: {e}")
    
    # Check required columns
    required_cols = {'start', 'end', 'strand'}
    
    # Accept either 'gene' or 'gene_id' as the ID column
    if 'gene_id' in df.columns:
        id_col = 'gene_id'
    elif 'gene' in df.columns:
        id_col = 'gene'
    else:
        raise FormatError(f"Genes CSV must have 'gene_id' or 'gene' column. Found: {list(df.columns)}")
    
    missing = required_cols - set(df.columns)
    if missing:
        raise FormatError(f"Genes CSV missing required columns: {missing}. Found: {list(df.columns)}")
    
    genes = {}
    for _, row in df.iterrows():
        gene_id = str(row[id_col])
        name = str(row.get('name', gene_id)) if pd.notna(row.get('name')) else gene_id
        gene_type = str(row.get('type', 'gene'))
        start = str(int(row['start']))
        end = str(int(row['end']))
        strand = str(row['strand'])
        
        # Format: [name, feature_type, start, end, strand, product]
        genes[gene_id] = [name, gene_type, start, end, strand, 'unknown']
    
    # Create name-to-id mapping
    genes2 = {info[0]: gene_id for gene_id, info in genes.items()}
    
    print(f"  Loaded {len(genes)} genes from CSV")
    return genes, genes2


def _parse_gff(filepath):
    """Parse genes from GFF/GFF3 format."""
    genes = {}
    features = {'CDS': True, 'ncRNA': True, 'rRNA': True, 'tRNA': True, 
                'pseudogene': True, 'sequence_feature': True, 'gene': True}
    
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            
            parts = line.strip().split('\t')
            if len(parts) < 9:
                continue
            
            feature = parts[2]
            start = parts[3]
            stop = parts[4]
            strand = parts[6]
            attributes = parts[8]
            
            if feature not in features:
                continue
            
            # Parse attributes
            gene_id, name, product = None, None, 'unknown'
            for attr in attributes.split(';'):
                attr = attr.strip()
                if attr.startswith('ID='):
                    gene_id = attr[3:]
                elif attr.startswith('Parent='):
                    if gene_id is None:
                        gene_id = attr[7:]
                elif attr.startswith('Name=') or attr.startswith('gene='):
                    name = attr.split('=')[1]
                elif attr.startswith('product='):
                    product = attr[8:]
            
            if gene_id:
                if name is None:
                    name = gene_id
                genes[gene_id] = [name, feature, start, stop, strand, product]
    
    # Create name-to-id mapping
    genes2 = {info[0]: gene_id for gene_id, info in genes.items()}
    
    print(f"  Loaded {len(genes)} genes from GFF")
    return genes, genes2


def validate_expression_file(filepath, genes2=None):
    """
    Validate and load an expression data file.
    
    Expected format (CSV):
        gene,sample1,sample2,sample3,...
        b0001,100.5,95.2,110.3,...
        b0002,50.1,48.9,52.0,...
    
    Also supports featureCounts output format (tab-separated with metadata columns).
    
    Args:
        filepath: Path to expression file
        genes2: Optional gene name to ID mapping
    
    Returns:
        dict: {gene_id: [name, class_code, num_samples, [expression_values]]}
    """
    if not os.path.exists(filepath):
        raise FormatError(f"Expression file not found: {filepath}")
    
    # Try to detect format
    with open(filepath, 'r') as f:
        first_line = f.readline().strip()
        second_line = f.readline().strip()
    
    # featureCounts format has comment lines or specific header
    if first_line.startswith('#') or 'Geneid\tChr\tStart' in first_line:
        return _parse_featurecounts(filepath)
    elif '\t' in first_line and first_line.count('\t') > 5:
        # Tab-separated with many columns, likely featureCounts
        return _parse_featurecounts(filepath)
    else:
        return _parse_expression_csv(filepath, genes2)


def _parse_expression_csv(filepath, genes2=None):
    """Parse expression data from simple CSV matrix."""
    try:
        # Try comma first, then tab
        df = pd.read_csv(filepath, index_col=0)
        if df.shape[1] == 0:
            df = pd.read_csv(filepath, index_col=0, sep='\t')
    except Exception as e:
        raise FormatError(f"Failed to read expression CSV: {e}")
    
    if df.shape[1] < 2:
        raise FormatError(f"Expression file must have at least 2 samples. Found: {df.shape[1]}")
    
    transcripts = {}
    for gene_name_raw, row in df.iterrows():
        gene_name = str(gene_name_raw)
        
        # Determine gene ID
        if gene_name.startswith('b') and len(gene_name) >= 5 and gene_name[1:5].isdigit():
            gene_id = f"gene-{gene_name}"
        elif genes2 and gene_name in genes2:
            gene_id = genes2[gene_name]
        elif gene_name.startswith('gene-'):
            gene_id = gene_name
        else:
            gene_id = f"gene-{gene_name}"
        
        expression_vector = row.astype(float).tolist()
        transcripts[gene_id] = [gene_name, 'expression', len(df.columns), expression_vector]
    
    print(f"  Loaded expression data for {len(transcripts)} genes ({df.shape[1]} samples)")
    return transcripts


def _parse_featurecounts(filepath):
    """Parse expression data from featureCounts output format."""
    try:
        df = pd.read_csv(filepath, sep='\t', comment='#')
    except Exception as e:
        raise FormatError(f"Failed to read featureCounts file: {e}")
    
    # featureCounts has: Geneid, Chr, Start, End, Strand, Length, then sample columns
    if 'Geneid' not in df.columns:
        raise FormatError("featureCounts file must have 'Geneid' column")
    
    # Get sample columns (everything after Length)
    if 'Length' in df.columns:
        len_idx = df.columns.get_loc('Length')
        count_cols = df.columns[len_idx + 1:]
    else:
        # Assume first 6 columns are metadata
        count_cols = df.columns[6:]
    
    transcripts = {}
    for _, row in df.iterrows():
        # Use gene ID as-is from the file (don't add prefix)
        gene_id = str(row['Geneid'])
        
        expression_vector = row[count_cols].astype(float).tolist()
        transcripts[gene_id] = [gene_id, 'featurecounts', len(count_cols), expression_vector]
    
    print(f"  Loaded expression data for {len(transcripts)} genes ({len(count_cols)} samples)")
    return transcripts


def validate_operons_file(filepath):
    """
    Validate and load an operons annotation file (for training).
    
    Expected format (CSV):
        operon,gene
        lacZYA,gene-b0344
        lacZYA,gene-b0343
        lacZYA,gene-b0342
    
    Returns:
        list: [[operon_name, start, end, [gene_ids]], ...]
    """
    if not os.path.exists(filepath):
        raise FormatError(f"Operons file not found: {filepath}")
    
    ext = os.path.splitext(filepath)[1].lower()
    
    if ext == '.csv':
        return _parse_operons_csv(filepath)
    else:
        return _parse_operons_tsv(filepath)


def _parse_operons_csv(filepath):
    """Parse operons from CSV format."""
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        raise FormatError(f"Failed to read operons CSV: {e}")
    
    # Check for required columns
    if 'operon' not in df.columns:
        raise FormatError(f"Operons CSV must have 'operon' column. Found: {list(df.columns)}")
    
    if 'gene' not in df.columns:
        raise FormatError(f"Operons CSV must have 'gene' column. Found: {list(df.columns)}")
    
    # Group by operon
    operon_groups = df.groupby('operon').agg({
        'gene': list,
        'start': 'min' if 'start' in df.columns else lambda x: 0,
        'end': 'max' if 'end' in df.columns else lambda x: 0
    }).reset_index()
    
    operons = []
    for _, row in operon_groups.iterrows():
        gene_list = row['gene']
        if len(gene_list) > 1:  # Only multi-gene operons
            start = int(row['start']) if 'start' in row else 0
            end = int(row['end']) if 'end' in row else 0
            operons.append([row['operon'], start, end, gene_list])
    
    print(f"  Loaded {len(operons)} multi-gene operons from CSV")
    return operons


def _parse_operons_tsv(filepath):
    """Parse operons from TSV format (e.g., RegulonDB)."""
    operons = []
    
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            
            parts = line.strip().split('\t')
            if len(parts) < 4:
                continue
            
            operon_name = parts[0]
            
            # Try to find gene list (usually last column or column 5/6)
            gene_list = []
            for col in parts[3:]:
                if ',' in col:
                    gene_list = [g.strip() for g in col.split(',') if g.strip()]
                    break
            
            if not gene_list and len(parts) > 3:
                gene_list = [parts[3].strip()]
            
            if len(gene_list) > 1:
                start = int(parts[1]) if parts[1].isdigit() else 0
                end = int(parts[2]) if parts[2].isdigit() else 0
                operons.append([operon_name, start, end, gene_list])
    
    print(f"  Loaded {len(operons)} multi-gene operons from TSV")
    return operons


def print_format_examples():
    """Print example input file formats."""
    print("""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                          OperoNN Input File Formats                            ║
╚═══════════════════════════════════════════════════════════════════════════════╝

1. GENES FILE (--genes)
   ─────────────────────
   CSV format with columns: gene_id/gene, name, start, end, strand
   
   Example (genes.csv):
   ┌─────────────┬──────┬───────┬───────┬────────┐
   │ gene_id     │ name │ start │ end   │ strand │
   ├─────────────┼──────┼───────┼───────┼────────┤
   │ gene-b0001  │ thrL │ 190   │ 255   │ +      │
   │ gene-b0002  │ thrA │ 337   │ 2799  │ +      │
   │ gene-b0003  │ thrB │ 2801  │ 3733  │ +      │
   └─────────────┴──────┴───────┴───────┴────────┘
   
   Also supports: GFF3 format


2. EXPRESSION FILE (--expression)
   ────────────────────────────────
   CSV matrix with genes as rows, samples as columns
   
   Example (expression.csv):
   ┌─────────┬──────────┬──────────┬──────────┐
   │         │ sample_1 │ sample_2 │ sample_3 │
   ├─────────┼──────────┼──────────┼──────────┤
   │ b0001   │ 100.5    │ 95.2     │ 110.3    │
   │ b0002   │ 50.1     │ 48.9     │ 52.0     │
   │ b0003   │ 200.0    │ 210.5    │ 195.8    │
   └─────────┴──────────┴──────────┴──────────┘
   
   Also supports: featureCounts output (tab-separated)


3. OPERONS FILE (--operons) - Only needed for training
   ────────────────────────────────────────────────────
   CSV with operon-to-gene mappings
   
   Example (operons.csv):
   ┌─────────┬─────────────┬───────┬───────┐
   │ operon  │ gene        │ start │ end   │
   ├─────────┼─────────────┼───────┼───────┤
   │ thrLABC │ gene-b0001  │ 190   │ 255   │
   │ thrLABC │ gene-b0002  │ 337   │ 2799  │
   │ thrLABC │ gene-b0003  │ 2801  │ 3733  │
   │ lacZYA  │ gene-b0344  │ 36... │ 36... │
   └─────────┴─────────────┴───────┴───────┘

""")

