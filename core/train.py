import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix
from utils.metrics import DetailedMetrics
from typing import Literal, Union
from utils.EarlyStopping import EarlyStopping
import os
class Trainer():
    def __init__(self, model: nn.Module, optimizer: Optimizer, criterion: nn.Module, device: torch.device, train_dataloader: DataLoader, val_dataloader: DataLoader, epochs: int, save_dir: str, early_stopping_metric: Literal['Val_Accuracy', 'Val_Loss'], early_stopping_patience: int, early_stopping_delta: float, useBCEWithLogitsLoss: bool = False):

        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.epochs = epochs
        self.save_dir = save_dir
        self.early_stopping_metric = early_stopping_metric
        self.early_stopping_patience = early_stopping_patience
        self.delta = early_stopping_delta
        self.useBCEWithLogitsLoss = useBCEWithLogitsLoss

        if self.useBCEWithLogitsLoss:
            logging.info("Trainer initialized for BCEWithLogitsLoss...")
        else:
            logging.info("Trainer initialized for CrossEntropyLoss...")

        # create save directory if it doesn't exist
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        
        # initialize early stopping variables
        if self.early_stopping_metric == 'Val_Accuracy':
            self.early_stopping = EarlyStopping(patience=self.early_stopping_patience, mode='max')
            logging.info(f"Using Early Stopping with Metric: {self.early_stopping_metric} and Patience: {self.early_stopping_patience}")
        elif self.early_stopping_metric == 'Val_Loss':
            self.early_stopping = EarlyStopping(patience=self.early_stopping_patience, mode='min')
            logging.info(f"Using Early Stopping with Metric: {self.early_stopping_metric} and Patience: {self.early_stopping_patience}")
        else:
            raise ValueError(f"Invalid early stopping metric: {self.early_stopping_metric}")
        

        # instantly move the model to the device
        self.model.to(self.device)
    
    def train_epoch(self, epoch: int):

        self.model.train()

        total_loss: float = 0.0
        correct_predictions: int = 0
        total_predictions: int = 0

        progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {epoch + 1} - Training", leave=False)

        for batch_idx, (x_m_rna, x_mi_rna, label) in enumerate(progress_bar):

            # move data to the device
            x_m_rna = x_m_rna.to(self.device)
            x_mi_rna = x_mi_rna.to(self.device)
            label = label.to(self.device)

            self.optimizer.zero_grad()

            # forward pass
            outputs = self.model(x_m_rna, x_mi_rna)
            loss = self.criterion(outputs, label)

            # backward pass
            loss.backward()

            # Gradient Clippint to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            # update weights
            self.optimizer.step()

            total_loss += loss.item()

            if self.useBCEWithLogitsLoss:
                probabilities = torch.sigmoid(outputs)
                predicted = (probabilities > 0.5).long()

            else:
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs.data, 1)
            
            total_predictions += label.size(0)
            correct_predictions += int(predicted.eq(label).sum().item())

            progress_bar.set_postfix({'Loss': f"{loss.item():.4f}", 'Accuracy': f"{(correct_predictions / total_predictions):.4f}"})
        
        avg_loss = total_loss / len(self.train_dataloader)
        avg_accuracy = correct_predictions / total_predictions

        logging.info(f"Epoch {epoch + 1} - Training Loss: {avg_loss:.4f}, Training Accuracy: {avg_accuracy:.4f}")

        return avg_loss, avg_accuracy
    
    def validate_epoch(self, epoch: int):

        self.model.eval()

        total_loss: float = 0.0
        all_predictions = []
        all_labels = []
        all_probabilities = []

        with torch.no_grad():
            progress_bar = tqdm(self.val_dataloader, desc=f"Epoch {epoch + 1} - Validation", leave=False)

            for batch_idx, (x_m_rna, x_mi_rna, label) in enumerate(progress_bar):
                # move data to the device
                x_m_rna = x_m_rna.to(self.device)
                x_mi_rna = x_mi_rna.to(self.device)
                label = label.to(self.device)

                # forward pass
                outputs = self.model(x_m_rna, x_mi_rna)
                loss = self.criterion(outputs, label)

                total_loss += loss.item()

                if self.useBCEWithLogitsLoss:
                    probabilities = torch.sigmoid(outputs)
                    predicted = (probabilities > 0.5).long().squeeze(1)

                    all_predictions.extend(predicted.cpu().numpy().flatten())
                    all_labels.extend(label.cpu().numpy().flatten())
                    all_probabilities.extend(probabilities.cpu().numpy().flatten())
                
                else:
                    probabilities = torch.softmax(outputs, dim=1)
                    _, predicted = torch.max(outputs.data, 1)

                    all_predictions.extend(predicted.cpu().numpy())
                    all_labels.extend(label.cpu().numpy())
                    all_probabilities.extend(probabilities.cpu().numpy())

                progress_bar.set_postfix({'Loss': f"{loss.item():.4f}"})

        avg_loss = total_loss/len(self.val_dataloader)
        accuracy = accuracy_score(all_labels, all_predictions)

        if self.useBCEWithLogitsLoss:
            logging.info("Trainer validate_epoch() using BCEWithLogitsLoss as Criterion to get Metrics ...")
            metrics = DetailedMetrics(
                y_true=all_labels,
                y_pred=all_predictions,
                y_prob=all_probabilities
            )._calculate_all_metrices()
        else:
            logging.info("Trainer validate_epoch() using CrossEntropyLoss as Criterion to get Metrics ...")
            class_1_probs = [prob[1] for prob in all_probabilities]
            metrics = DetailedMetrics(
                y_true=all_labels,
                y_pred=all_predictions,
                y_prob=class_1_probs
            )._calculate_all_metrices()

        
        cm = confusion_matrix(all_labels, all_predictions)
        
        metrics['confusion_matrix'] = cm
        logging.info(f"Epoch {epoch + 1} - Validation Loss: {avg_loss:.4f}, Validation Accuracy: {accuracy:.4f}, metrics: {metrics}")

        return avg_loss, accuracy, metrics
    
    def train(self):
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []


        best_val_accuracy = 0.0
        best_val_loss = float('inf')
        best_metrics = None
        best_epoch = 0

        logging.info(f"Starting training for {self.epochs} epochs on device: {self.device}")
        logging.info(f"Training on {len(self.train_dataloader)} batches and validating on {len(self.val_dataloader)} batches...")
        logging.info(f"Model Parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}")


        for epoch in range(self.epochs):
            
            train_loss, train_acc = self.train_epoch(epoch)
            val_loss, val_acc, val_metrics = self.validate_epoch(epoch)

            # Update learning rate scheduler
            #self.scheduler.step(val_loss)

            # Store history
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accuracies.append(train_acc)
            val_accuracies.append(val_acc)

            # current metric for the early stopping judgement
            if self.early_stopping_metric == 'Val_Accuracy':
                current_metric_value = val_acc
            elif self.early_stopping_metric == 'Val_Loss':
                current_metric_value = val_loss
            else:
                raise ValueError(f"Invalid early stopping metric: {self.early_stopping_metric}")

            # check if the model has improved
            is_improved = False

            if self.early_stopping_metric == 'Val_Accuracy' and val_acc > best_val_accuracy + self.delta:
                logging.info(f"Model improved - {self.early_stopping_metric}:  from {best_val_accuracy} to {val_acc} at epoch {epoch + 1}")

                is_improved = True
                best_val_accuracy = val_acc
                best_val_loss = val_loss
                best_metrics = val_metrics
                best_epoch = epoch + 1
            elif self.early_stopping_metric == 'Val_Loss' and val_loss < best_val_loss - self.delta:
                logging.info(f"Model improved - {self.early_stopping_metric}:  from {best_val_loss} to {val_loss} at epoch {epoch + 1}")
                is_improved = True
                best_val_accuracy = val_acc
                best_val_loss = val_loss
                best_metrics = val_metrics
                best_epoch = epoch + 1
            
            # if the model has improved then save the model
            if is_improved:
                #create best model directory if it doesn't exist
                if not os.path.exists(os.path.join(self.save_dir, 'models')):
                    os.makedirs(os.path.join(self.save_dir, 'models'))

                best_model_path = os.path.join(self.save_dir, 'models', 'best_model.pth')
                torch.save(self.model.state_dict(), best_model_path)
                logging.info(f"Best model saved to: {best_model_path}")
            
            # save model checkpoints
            if (epoch + 1) % 10 == 0:
                
                # create checkpoints directory if it doesn't exist
                if not os.path.exists(os.path.join(self.save_dir, 'models','checkpoints')):
                    os.makedirs(os.path.join(self.save_dir, 'models','checkpoints'))

                checkpoint_path  = os.path.join(self.save_dir, 'models','checkpoints', f'checkpoint_epoch_{epoch + 1}.pth')
                torch.save(self.model.state_dict(), checkpoint_path)
                logging.info(f"Checkpoint saved to: {checkpoint_path}")
            
            # early stopping
            if self.early_stopping(score=current_metric_value):
                logging.info(f"Early stopping triggered at epoch {epoch + 1}")
                logging.info(f"Best model is at epoch {best_epoch}")
                break

            # get the current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            logging.info(f"Current Learning Rate: {current_lr} at epoch {epoch + 1}")
        
        return train_losses, val_losses, train_accuracies, val_accuracies, best_val_accuracy, best_val_loss, best_metrics