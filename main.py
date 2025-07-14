import logging
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
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import random
import os
from utils.visuals import Plotter

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def set_seed(seed: int = 123):
    #os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8" # uncomment if using CUDA
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    #torch.use_deterministic_algorithms(True, warn_only=True) # if error related to this then do remove this line


def main(seed: int = 123):
    
    #Set seed for reproducibility
    set_seed(seed)

    # read the dataset
    dataset: pd.DataFrame = pd.read_csv(CONFIG['raw_data_path'])
    
    # check if all the required columns are in the dataset dataframe
    required_cols = [CONFIG['m_rna_col_name'], CONFIG['mi_rna_col_name'], CONFIG['class_col_name']]
    for col in required_cols:
        if col not in dataset.columns:
            raise Exception(f"Column {col} not found in the dataset, required columns: {required_cols}")
    
    # Train -> 80%, Test -> 20%
    train, val = train_test_split(dataset, test_size=0.2, random_state=seed, stratify=dataset[CONFIG['class_col_name']], shuffle=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for k in CONFIG['k_mers']:
        
        for batch_size in CONFIG['batch_sizes']:

            for learning_rate in CONFIG['learning_rates']:

                for dropout_rate in CONFIG['dropouts']:

                    train_dataset = DatasetLoader(
                    dataset=train,
                    dataset_type='Train',
                    k_mer=k,
                    m_rna_col_name=CONFIG['m_rna_col_name'],
                    mi_rna_col_name=CONFIG['mi_rna_col_name'],
                    class_col_name=CONFIG['class_col_name'],
                    useBCEWithLogitsLoss=CONFIG['useBCEWithLogitsLoss']
                )

                val_dataset = DatasetLoader(
                    dataset=val,
                    dataset_type='Validation',
                    k_mer=k,
                    m_rna_col_name=CONFIG['m_rna_col_name'],
                    mi_rna_col_name=CONFIG['mi_rna_col_name'],
                    class_col_name=CONFIG['class_col_name'],
                    useBCEWithLogitsLoss=CONFIG['useBCEWithLogitsLoss']
                )

                num_workers = mp.cpu_count() // 2

                g = torch.Generator()
                g.manual_seed(seed)

                logger.info(f"Number of workers: {num_workers}")

                train_dataloader = DataLoader(
                    dataset=train_dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=num_workers,
                    worker_init_fn=seed_worker,
                    generator=g,
                    collate_fn=custom_collate_fn
                )

                val_dataloader = DataLoader(
                    dataset=val_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    worker_init_fn=seed_worker,
                    generator=g,
                    collate_fn=custom_collate_fn
                )

                model = InteractionModel(dropout_rate=dropout_rate, k=k, useBCEWithLogits=CONFIG['useBCEWithLogitsLoss'])

                criterion = nn.BCEWithLogitsLoss() if CONFIG['useBCEWithLogitsLoss'] else nn.CrossEntropyLoss()

                optimizer = optim.AdamW(params=model.parameters(), lr=learning_rate, weight_decay=1e-4)

                save_dir = os.path.join(CONFIG['save_dir'], f"k_{k}_batch_{batch_size}_lr_{learning_rate}_dropout_{dropout_rate}")

                training_history = Trainer(
                    model=model,
                    optimizer=optimizer,
                    criterion=criterion,
                    device=device,
                    train_dataloader=train_dataloader,
                    val_dataloader=val_dataloader,
                    epochs=CONFIG['total_epochs'],
                    save_dir=save_dir,
                    early_stopping_metric='Val_Accuracy',
                    early_stopping_patience=CONFIG['early_stopping_patience'],
                    early_stopping_delta=CONFIG['early_stopping_delta'],
                    useBCEWithLogitsLoss=CONFIG['useBCEWithLogitsLoss']
                ).train()

                train_losses, val_losses, train_accuracies, val_accuracies, best_val_accuracy, best_val_loss, best_metrics = training_history

                plotter = Plotter(
                    train_losses=train_losses,
                    val_losses=val_losses,
                    train_accuracies=train_accuracies,
                    val_accuracies=val_accuracies,
                    save_dir=save_dir
                )

                plotter.plot_training()
                plotter.plot_confusion_matrix(cm=best_metrics['confusion_matrix'])
                logger.log(f"Best Validation Accuracy: {best_val_accuracy}")

                logger.log(f"Best Validation Loss: {best_val_loss}")
                logger.log(f"Best Validation Loss: {best_val_loss}")
    

if __name__ == "__main__":
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
    main()