import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log', mode='w', encoding='utf-8'),
        logging.StreamHandler()
    ],

)
logger = logging.getLogger(__name__)

import pandas as pd
from config.settings import CONFIG
from sklearn.model_selection import train_test_split
from utils.DatasetLoader import DatasetLoader, custom_collate_fn
from torch.utils.data import DataLoader
from core.train import Trainer
from core.model import InteractionModel
# from core.crossmodelattention import InteractionModel
from torch import optim
import torch.nn as nn
import torch
import multiprocessing as mp
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import random
import os
from utils.visuals import Plotter

def main():
    
    #Set seed for reproducibility
    seed = 123
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # read the dataset
    dataset: pd.DataFrame = pd.read_csv(CONFIG['raw_data_path'])
    
    # check if all the required columns are in the dataset dataframe
    required_cols = [CONFIG['m_rna_col_name'], CONFIG['mi_rna_col_name'], CONFIG['class_col_name']]
    for col in required_cols:
        if col not in dataset.columns:
            raise Exception(f"Column {col} not found in the dataset, required columns: {required_cols}")
    
    
    # training -> 70%, validation -> 15%, testing -> 15%
    train, val_test = train_test_split(dataset, test_size=0.3, random_state=seed, stratify=dataset[CONFIG['class_col_name']], shuffle=True)
    val, test = train_test_split(val_test, test_size=0.5, random_state=seed, stratify=val_test[CONFIG['class_col_name']], shuffle=True)
    
    # Load the Dataset
    train_dataset = DatasetLoader(dataset=train,
                                  dataset_type='Train',
                                  k_mer=CONFIG['k_mer'],
                                  m_rna_col_name=CONFIG['m_rna_col_name'],
                                  mi_rna_col_name=CONFIG['mi_rna_col_name'],
                                  class_col_name=CONFIG['class_col_name'])

    test_dataset = DatasetLoader(dataset=test,
                                 dataset_type='Test',
                                 k_mer=CONFIG['k_mer'],
                                 m_rna_col_name=CONFIG['m_rna_col_name'],
                                 mi_rna_col_name=CONFIG['mi_rna_col_name'],
                                 class_col_name=CONFIG['class_col_name'])
    
    val_dataset = DatasetLoader(dataset=val,
                                dataset_type='Validation',
                                k_mer=CONFIG['k_mer'],
                                m_rna_col_name=CONFIG['m_rna_col_name'],
                                mi_rna_col_name=CONFIG['mi_rna_col_name'],
                                class_col_name=CONFIG['class_col_name'])

    # create dataloaders
    num_workers = mp.cpu_count() // 2

    logging.info(f"Number of workers: {num_workers} for DataLoader")
    train_dataloader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, collate_fn=custom_collate_fn, num_workers=num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False, collate_fn=custom_collate_fn, num_workers=num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False, collate_fn=custom_collate_fn, num_workers=num_workers)

    logging.info(f"Length of:\n\tTrain DataLoader: {len(train_dataloader)}\n\tTest DataLoader: {len(test_dataloader)}\n\tVal DataLoader: {len(val_dataloader)}\n")

    # Train the model
    model = InteractionModel(dropout_rate=0.3, k=CONFIG['k_mer'])
    # criterion = nn.CrossEntropyLoss()

    # Try with BCEWithLogitsLoss
    criterion = nn.BCEWithLogitsLoss()

    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

    # set device considering the macbook pro m1 also
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() and torch.backends.mps.is_built() else 'cpu')

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=7, min_lr=1e-6)

    #where to save?
    save_dir = os.path.join(os.getcwd(), str(CONFIG['save_dir']), str(CONFIG['k_mer']), str(CONFIG['batch_size']))

    training_history = Trainer(model=model,
                               optimizer=optimizer,
                               criterion=criterion,
                               scheduler=scheduler,
                               device=device,
                               train_dataloader=train_dataloader,
                               val_dataloader=val_dataloader,
                               epochs=100,
                               save_dir=save_dir,
                               early_stopping_metric='Val_Accuracy',
                               early_stopping_patience=15).train()
    
    train_losses, val_losses, train_accuracies, val_accuracies, best_val_accuracy, best_val_loss, best_metrics = training_history

    # Plot the training history
    Plotter(
        train_losses=train_losses,
        val_losses=val_losses,
        train_accuracies=train_accuracies,
        val_accuracies=val_accuracies,
        save_dir=save_dir
    )

    logger.info(f"Best Validation Accuracy: {best_val_accuracy}")
    logger.info(f"Best Validation Loss: {best_val_loss}")
    logger.info(f"Best Metrics: {best_metrics}")
    logger.info(f"Train Losses: {train_losses}")
    logger.info(f"Val Losses: {val_losses}")
    logger.info(f"Train Accuracies: {train_accuracies}")
    logger.info(f"Val Accuracies: {val_accuracies}")

if __name__ == "__main__":
    main()