from typing import Dict

CONFIG: Dict[str, any] = {
    'raw_data_path': 'data/deepmirtar.csv',
    'm_rna_col_name': 'Target Site',
    'mi_rna_col_name': 'miRNA_seq',
    'class_col_name': 'label',
    'k_mers': 6,
    'batch_sizes': 32,
    'save_dir': 'output',
    'early_stopping_patience': 10,
    'total_epochs': 100,
    'early_stopping_delta': 0.0001,
    'useBCEWithLogitsLoss': False
}