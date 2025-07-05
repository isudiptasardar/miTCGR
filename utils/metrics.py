from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, matthews_corrcoef
from typing import Dict

class DetailedMetrics():
    def __init__(self, y_true: list[int], y_pred: list[int], y_prob: list[float]):
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_prob = y_prob

        # validate if all inputs are of the same length
        if len(self.y_true) != len(self.y_pred) or len(self.y_true) != len(self.y_prob):
            raise ValueError("All inputs must be of the same length")
        
        # compute confusion matrix
        self.tn, self.fp, self.fn, self.tp = confusion_matrix(self.y_true, self.y_pred).ravel()
    
    def _calculate_all_metrices(self):

        metrics: Dict[str, float] = {
            'accuracy': float,
            'precision': float,
            'recall': float,
            'specificity': float,
            'npv': float,
            'f1_score': float,
            'mcc': float,
            'roc_auc': float,
            'balanced_accuracy': float,
            'fpr': float,
            'fnr': float,
            'lr_positive': float,
            'lr_negative': float,
            'y_true': int,
            'y_pred': int
        }

        metrics['accuracy'] = accuracy_score(self.y_true, self.y_pred)
        metrics['precision'] = precision_score(self.y_true, self.y_pred, zero_division=0) # Positive Predictive Value -> PPV
        metrics['recall'] = recall_score(self.y_true, self.y_pred, zero_division=0) # Sensitivity = TP / (TP + FN) -> TPR
        metrics['specificity'] = self.tn / (self.tn + self.fp) if (self.tn + self.fp) > 0 else 0 # True Negative Rate -> TNR
        metrics['npv'] = self.tn / (self.tn + self.fn) if (self.tn + self.fn) > 0 else 0 # Negative Predictive Value -> NPV
        metrics['f1_score'] = f1_score(self.y_true, self.y_pred, zero_division=0)
        metrics['mcc'] = matthews_corrcoef(self.y_true, self.y_pred)

        try:
            metrics['roc_auc'] = roc_auc_score(self.y_true, self.y_prob)
        except ValueError:
            metrics['roc_auc'] = 0.0
        
        # some additional metrics
        metrics['balanced_accuracy'] = (metrics['recall'] + metrics['specificity']) / 2
        metrics['fpr'] = self.fp / (self.fp + self.tn) if (self.fp + self.tn) > 0 else 0.0
        metrics['fnr'] = self.fn / (self.fn + self.tp) if (self.fn + self.tp) > 0 else 0.0
        metrics['lr_positive'] = metrics['recall'] / metrics['fpr'] if metrics['fpr'] > 0 else float('inf') # Positive Likelihood Ratio -> PLR
        metrics['lr_negative'] = metrics['specificity'] / metrics['fnr'] if metrics['fnr'] > 0 else float('inf') # Negative Likelihood Ratio -> NLR

        return metrics