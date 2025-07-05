import logging
import matplotlib.pyplot as plt
import seaborn as sns

class Plotter():
    def __init__(self, train_losses, val_losses, train_accuracies, val_accuracies, save_dir: str):

        self.train_losses = train_losses
        self.val_losses = val_losses
        self.train_accuracies = train_accuracies
        self.val_accuracies = val_accuracies
        self.save_dir = save_dir

        # check if all the lists have the same length
        if len(self.train_losses) != len(self.val_losses) != len(self.train_accuracies) != len(self.val_accuracies):
            logging.error(f"All the lists must have the same length, got train_losses: {len(self.train_losses)}, val_losses: {len(self.val_losses)}, train_accuracies: {len(self.train_accuracies)}, val_accuracies: {len(self.val_accuracies)}")

            raise ValueError(f"All the lists must have the same length, got {len(self.train_losses)}, {len(self.val_losses)}, {len(self.train_accuracies)}, {len(self.val_accuracies)}")
        
    def plot_training(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        ax1.plot(self.train_losses, label='Training Loss', color='blue')
        ax1.plot(self.val_losses, label='Validation Loss', color='red')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()

        ax2.plot(self.train_accuracies, label='Training Accuracy', color='blue')
        ax2.plot(self.val_accuracies, label='Validation Accuracy', color='red')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()

        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/training_history.png", dpi = 600, bbox_inches='tight')
        plt.close()
        return
    