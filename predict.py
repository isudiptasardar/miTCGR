import torch
from core.model import InteractionModel
class Predict():
    def __init__(self, model_path: str, mi_rna_seq: str, m_rna_seq: str, k_mer: int):
        self.model_path = model_path
        self.mi_rna_seq = mi_rna_seq
        self.m_rna_seq = m_rna_seq
        self.k_mer = k_mer
    
    def predict(self):
        model = InteractionModel(dropout_rate=0.3, k=self.k_mer)

        # Load Model State
        model.load_state_dict(torch.load(self.model_path))