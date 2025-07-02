
import pandas as pd
from config.settings import CONFIG
from sklearn.model_selection import train_test_split
from utils.DatasetLoader import DatasetLoader, custom_collate_fn
from torch.utils.data import DataLoader
from core.train import Trainer
from core.model import InteractionModel
from torch import optim
import torch.nn as nn
import torch
import multiprocessing as mp

def main():

    # read the dataset
    dataset: pd.DataFrame = pd.read_csv(CONFIG['raw_data_path'])
    
    # check if all the required columns are in the dataset dataframe
    required_cols = [CONFIG['m_rna_col_name'], CONFIG['mi_rna_col_name'], CONFIG['class_col_name']]
    for col in required_cols:
        if col not in dataset.columns:
            raise Exception(f"Column {col} not found in the dataset, required columns: {required_cols}")
    
    # split the dataset into train, test and validation sets -> Train 64%, Test 16%, Validation 20%, if you want test_and val 20%, set second test_size = 0.25
    train_test, val = train_test_split(dataset, test_size=0.2, random_state=42, stratify=dataset[CONFIG['class_col_name']], shuffle=True)
    train, test = train_test_split(train_test, test_size=0.2, random_state=42, stratify=train_test[CONFIG['class_col_name']], shuffle=True)
    
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
    print(f"Number of workers: {num_workers} for DataLoader")
    train_dataloader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, collate_fn=custom_collate_fn, num_workers=num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False, collate_fn=custom_collate_fn, num_workers=num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False, collate_fn=custom_collate_fn, num_workers=num_workers)

    print(f"Length of:\n\tTrain DataLoader: {len(train_dataloader)}\n\tTest DataLoader: {len(test_dataloader)}\n\tVal DataLoader: {len(val_dataloader)}\n")

    # Train the model
    model = InteractionModel(dropout_rate=0.3)
    criterion = nn.CrossEntropyLoss() # Try with BCELogitLoss once
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=7, min_lr=1e-6)

    training_history = Trainer(model=model,
                               optimizer=optimizer,
                               criterion=criterion,
                               scheduler=scheduler,
                               device=device,
                               train_dataloader=train_dataloader,
                               val_dataloader=val_dataloader,
                               epochs=100,
                               save_dir=CONFIG['save_dir'],
                               early_stopping_metric='Val_Accuracy',
                               early_stopping_patience=15).train()
    print(training_history)
if __name__ == "__main__":
    main()