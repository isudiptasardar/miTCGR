from typing import Dict

CONFIG: Dict[str, any] = {
    'raw_data_path': 'data/miraw.csv',
    'm_rna_col_name': 'mRNA_Site_Transcript',
    'mi_rna_col_name': 'mature_miRNA_Transcript',
    'class_col_name': 'validation',
    'k_mer': 6,
    'batch_size': 32,
    'save_dir': 'output',
    'early_stopping_patience': 10,
    'total_epochs': 100,
    'early_stopping_delta': 0.0001,
    'useBCEWithLogitsLoss': False
}