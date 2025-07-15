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
import json
import time
from datetime import datetime

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
    
    # Hyperparameter optimization tracking
    hyperparameter_results = []
    best_hyperparams = None
    global_best_val_accuracy = 0.0
    
    # Calculate total combinations for progress tracking
    total_combinations = len(CONFIG['k_mers']) * len(CONFIG['batch_sizes']) * len(CONFIG['learning_rates']) * len(CONFIG['dropouts'])
    current_combination = 0
    start_time = time.time()
    
    logger.info(f"Starting hyperparameter optimization with {total_combinations} combinations")
    logger.info(f"Device: {device}")

    for k in CONFIG['k_mers']:
        
        for batch_size in CONFIG['batch_sizes']:

            for learning_rate in CONFIG['learning_rates']:

                for dropout_rate in CONFIG['dropouts']:
                    
                    current_combination += 1
                    elapsed_time = time.time() - start_time
                    avg_time_per_combination = elapsed_time / current_combination if current_combination > 0 else 0
                    remaining_combinations = total_combinations - current_combination
                    estimated_remaining_time = avg_time_per_combination * remaining_combinations
                    
                    logger.info(f"=== Combination {current_combination}/{total_combinations} ===")
                    logger.info(f"k_mer: {k}, batch_size: {batch_size}, learning_rate: {learning_rate}, dropout: {dropout_rate}")
                    logger.info(f"Elapsed: {elapsed_time/60:.1f}min, Estimated remaining: {estimated_remaining_time/60:.1f}min")

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
                
                # Store hyperparameter results
                hyperparameter_result = {
                    'k_mer': k,
                    'batch_size': batch_size,
                    'learning_rate': learning_rate,
                    'dropout_rate': dropout_rate,
                    'val_accuracy': best_val_accuracy,
                    'val_loss': best_val_loss,
                    'save_dir': save_dir,
                    'timestamp': datetime.now().isoformat(),
                    'metrics': best_metrics
                }
                hyperparameter_results.append(hyperparameter_result)
                
                # Track best hyperparameters
                if best_val_accuracy > global_best_val_accuracy:
                    global_best_val_accuracy = best_val_accuracy
                    best_hyperparams = {
                        'k_mer': k,
                        'batch_size': batch_size,
                        'learning_rate': learning_rate,
                        'dropout_rate': dropout_rate,
                        'val_accuracy': best_val_accuracy,
                        'val_loss': best_val_loss,
                        'save_dir': save_dir
                    }
                
                logger.info(f"Current Run - Val Accuracy: {best_val_accuracy}, Val Loss: {best_val_loss}")
                logger.info(f"Global Best - Val Accuracy: {global_best_val_accuracy}")
                
                # Save intermediate results
                results_file = os.path.join(CONFIG['save_dir'], 'hyperparameter_results.json')
                os.makedirs(CONFIG['save_dir'], exist_ok=True)
                with open(results_file, 'w') as f:
                    json.dump(hyperparameter_results, f, indent=2, default=str)
    
    # Final hyperparameter optimization summary
    total_time = time.time() - start_time
    logger.info(f"\n=== HYPERPARAMETER OPTIMIZATION COMPLETE ===")
    logger.info(f"Total time: {total_time/60:.1f} minutes")
    logger.info(f"Total combinations tested: {total_combinations}")
    logger.info(f"Average time per combination: {total_time/total_combinations:.1f} seconds")
    
    if best_hyperparams:
        logger.info(f"\n=== BEST HYPERPARAMETERS ===")
        logger.info(f"k_mer: {best_hyperparams['k_mer']}")
        logger.info(f"batch_size: {best_hyperparams['batch_size']}")
        logger.info(f"learning_rate: {best_hyperparams['learning_rate']}")
        logger.info(f"dropout_rate: {best_hyperparams['dropout_rate']}")
        logger.info(f"Best validation accuracy: {best_hyperparams['val_accuracy']:.4f}")
        logger.info(f"Best validation loss: {best_hyperparams['val_loss']:.4f}")
        logger.info(f"Best model saved at: {best_hyperparams['save_dir']}")
    
    # Save final results and create CSV summary
    final_results_file = os.path.join(CONFIG['save_dir'], 'final_hyperparameter_results.json')
    with open(final_results_file, 'w') as f:
        json.dump({
            'best_hyperparams': best_hyperparams,
            'all_results': hyperparameter_results,
            'summary': {
                'total_time_minutes': total_time/60,
                'total_combinations': total_combinations,
                'best_accuracy': global_best_val_accuracy
            }
        }, f, indent=2, default=str)
    
    # Create CSV summary for easy analysis
    df_results = pd.DataFrame(hyperparameter_results)
    csv_file = os.path.join(CONFIG['save_dir'], 'hyperparameter_results.csv')
    df_results.to_csv(csv_file, index=False)
    
    logger.info(f"\nResults saved to:")
    logger.info(f"- JSON: {final_results_file}")
    logger.info(f"- CSV: {csv_file}")
    

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