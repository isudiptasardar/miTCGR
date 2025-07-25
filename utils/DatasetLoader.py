import logging
import torch
from torch.utils.data import Dataset
import pandas as pd
from typing import Literal
from utils.FCGR import FCGR
import numpy as np

class DatasetLoader(Dataset):
    def __init__(self, dataset: pd.DataFrame, dataset_type: Literal['Train', 'Test', 'Validation'], k_mer: int, m_rna_col_name: str, mi_rna_col_name: str, class_col_name: str, useBCEWithLogitsLoss: bool = False):

        self.dataset = dataset
        self.dataset_type = dataset_type
        self.k_mer = k_mer
        self.m_rna_col_name = m_rna_col_name
        self.mi_rna_col_name = mi_rna_col_name
        self.class_col_name = class_col_name
        self.useBCEWithLogitsLoss = useBCEWithLogitsLoss

        logging.info(f'Loaded {self.dataset_type} dataset with {len(self.dataset)} samples. Processing for {self.k_mer}-mer...')

        if self.useBCEWithLogitsLoss:
            logging.info("DatasetLoader initialized for BCEWithLogitsLoss...")
        else:
            logging.info("DatasetLoader initialized for CrossEntropyLoss...")
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        try:
            row: pd.Series = self.dataset.iloc[index]
            m_rna_seq: str = str(row[self.m_rna_col_name]).replace('U', 'T')
            mi_rna_seq: str = str(row[self.mi_rna_col_name]).replace('U', 'T')
            label: int = int(row[self.class_col_name])

            # Now generate FCGR matrix also
            m_rna_fcgr: np.ndarray = FCGR(sequence=m_rna_seq, k=self.k_mer).generate_fcgr()
            mi_rna_fcgr: np.ndarray = FCGR(sequence=mi_rna_seq, k=self.k_mer).generate_fcgr()

            # Check if the shape is aligned with the k_mer size given
            match self.k_mer:
                case 3:
                    assert m_rna_fcgr.shape == (8, 8), f"m_rna_fcgr shape is {m_rna_fcgr.shape}, expected (8, 8)"
                    assert mi_rna_fcgr.shape == (8, 8), f"mi_rna_fcgr shape is {mi_rna_fcgr.shape}, expected (8, 8)"
                case 4:
                    assert m_rna_fcgr.shape == (16, 16), f"m_rna_fcgr shape is {m_rna_fcgr.shape}, expected (16, 16)"
                    assert mi_rna_fcgr.shape == (16, 16), f"mi_rna_fcgr shape is {mi_rna_fcgr.shape}, expected (16, 16)"
                case 5:
                    assert m_rna_fcgr.shape == (32, 32), f"m_rna_fcgr shape is {m_rna_fcgr.shape}, expected (32, 32)"
                    assert mi_rna_fcgr.shape == (32, 32), f"mi_rna_fcgr shape is {mi_rna_fcgr.shape}, expected (32, 32)"
                case 6:
                    assert m_rna_fcgr.shape == (64, 64), f"m_rna_fcgr shape is {m_rna_fcgr.shape}, expected (64, 64)"
                    assert mi_rna_fcgr.shape == (64, 64), f"mi_rna_fcgr shape is {mi_rna_fcgr.shape}, expected (64, 64)"
                case _:
                    raise ValueError(f"Invalid k_mer value in config: {self.k_mer}")
            
            if self.useBCEWithLogitsLoss:
                return torch.FloatTensor(m_rna_fcgr).unsqueeze(0), torch.FloatTensor(mi_rna_fcgr).unsqueeze(0), torch.tensor([label], dtype=torch.float)
            else:
                return torch.FloatTensor(m_rna_fcgr).unsqueeze(0), torch.FloatTensor(mi_rna_fcgr).unsqueeze(0), torch.tensor(label, dtype=torch.long)
        
        except Exception as e:
            logging.error("Error in __getitem__ of DatasetLoader:", e)
            raise e
        

def custom_collate_fn(batch):
    try:
        x_mrna = torch.stack([item[0] for item in batch], dim=0)
        x_mirna = torch.stack([item[1] for item in batch], dim=0)
        y = torch.stack([item[2] for item in batch], dim=0)
        return x_mrna, x_mirna, y
    except Exception as e:
        logging.error("Error in custom_collate_fn of DatasetLoader:", e)
        raise e