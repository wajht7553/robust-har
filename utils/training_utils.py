##################################################
# Training and evaluation utilities with verbose logging
##################################################

import torch
from torch import nn
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)


class Trainer:
    """Generic trainer for PyTorch models with verbose logging"""

    def __init__(self, model, device, criterion=None, optimizer=None, scheduler=None, 
                 early_stopping_patience=10, checkpoint_path=None):
        """
        Args:
            model: PyTorch model
            device: torch device (cuda/cpu)
            criterion: loss function (default: CrossEntropyLoss)
            optimizer: optimizer (default: Adam)
            scheduler: learning rate scheduler (optional)
            early_stopping_patience: number of epochs to wait before early stopping (default: 10)
            checkpoint_path: path to save best model checkpoint (optional)
        """
        self.model = model.to(device)
        self.device = device
        self.criterion = criterion if criterion else nn.CrossEntropyLoss()
        self.optimizer = (
            optimizer if optimizer else torch.optim.Adam(model.parameters(), lr=1e-3)
        )
        self.scheduler = scheduler
        self.early_stopping_patience = early_stopping_patience
        self.checkpoint_path = checkpoint_path

        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        
        # Early stopping tracking
        self.best_val_acc = 0.0
        self.best_val_loss = float('inf')
        self.epochs_no_improve = 0
        self.early_stopped = False
        self.best_epoch = 0

    def train_epoch(self, train_loader, epoch):
        """
        Train for one epoch

        Returns:
            avg_loss, avg_acc
        """
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_targets = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]", leave=False)
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            preds = output.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(target.cpu().numpy())

            # Update progress bar
            pbar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "avg_loss": f"{total_loss/(batch_idx+1):.4f}",
                }
            )

        avg_loss = total_loss / len(train_loader)
        avg_acc = accuracy_score(all_targets, all_preds)

        return avg_loss, avg_acc

    def validate(self, val_loader, epoch=None):
        """
        Validate the model

        Returns:
            avg_loss, avg_acc, all_preds, all_targets
        """
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []

        desc = f"Epoch {epoch} [Val]" if epoch is not None else "Validation"
        pbar = tqdm(val_loader, desc=desc, leave=False)

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(pbar):
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                loss = self.criterion(output, target)

                total_loss += loss.item()
                preds = output.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_targets.extend(target.cpu().numpy())

                # Update progress bar
                pbar.set_postfix(
                    {
                        "loss": f"{loss.item():.4f}",
                        "avg_loss": f"{total_loss/(batch_idx+1):.4f}",
                    }
                )

        avg_loss = total_loss / len(val_loader)
        avg_acc = accuracy_score(all_targets, all_preds)

        return avg_loss, avg_acc, all_preds, all_targets

    def train(self, train_loader, val_loader, epochs, verbose=True):
        """
        Full training loop with early stopping and best model checkpointing

        Args:
            train_loader: training data loader
            val_loader: validation data loader
            epochs: number of epochs
            verbose: whether to print epoch summaries

        Returns:
            history dict with losses, accuracies, and early stopping info
        """
        for epoch in range(1, epochs + 1):
            # Training
            train_loss, train_acc = self.train_epoch(train_loader, epoch)
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)

            # Validation
            val_loss, val_acc, _, _ = self.validate(val_loader, epoch)
            self.val_losses.append(val_loss)
            self.val_accs.append(val_acc)

            # Learning rate scheduling
            if self.scheduler:
                self.scheduler.step()

            # Check for improvement
            improved = False
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                self.epochs_no_improve = 0
                improved = True
                
                # Save best model checkpoint
                if self.checkpoint_path:
                    save_model(self.model, self.checkpoint_path)
                    if verbose:
                        print(f"  → Best model saved to {self.checkpoint_path}")
            else:
                self.epochs_no_improve += 1

            # Print epoch summary
            if verbose:
                status = "✓ BEST" if improved else f"(no improve: {self.epochs_no_improve}/{self.early_stopping_patience})"
                print(
                    f"Epoch {epoch}/{epochs} - "
                    f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                    f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} {status}"
                )

            # Early stopping check
            if self.epochs_no_improve >= self.early_stopping_patience:
                self.early_stopped = True
                if verbose:
                    print(f"\n⚠ Early stopping triggered after {epoch} epochs")
                    print(f"  Best validation accuracy: {self.best_val_acc:.4f} at epoch {self.best_epoch}")
                break

        # Load best model if checkpoint exists
        if self.checkpoint_path and self.early_stopped:
            if verbose:
                print(f"  Loading best model from {self.checkpoint_path}")
            load_model(self.model, self.checkpoint_path, self.device)

        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "train_accs": self.train_accs,
            "val_accs": self.val_accs,
            "best_val_acc": self.best_val_acc,
            "best_val_loss": self.best_val_loss,
            "best_epoch": self.best_epoch,
            "early_stopped": self.early_stopped,
            "total_epochs": epoch
        }


def compute_metrics(y_true, y_pred, num_classes=None):
    """
    Compute comprehensive metrics

    Args:
        y_true: true labels
        y_pred: predicted labels
        num_classes: number of classes (optional)

    Returns:
        dict of metrics
    """
    acc = accuracy_score(y_true, y_pred)

    # Compute per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )

    # Macro averages
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    # Weighted averages
    precision_weighted, recall_weighted, f1_weighted, _ = (
        precision_recall_fscore_support(
            y_true, y_pred, average="weighted", zero_division=0
        )
    )

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    return {
        "accuracy": float(acc),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(precision_weighted),
        "recall_weighted": float(recall_weighted),
        "f1_weighted": float(f1_weighted),
        "precision_per_class": precision.tolist(),
        "recall_per_class": recall.tolist(),
        "f1_per_class": f1.tolist(),
        "support_per_class": support.tolist(),
        "confusion_matrix": cm.tolist(),
    }


def save_model(model, filepath):
    """Save model state dict"""
    torch.save(model.state_dict(), filepath)


def load_model(model, filepath, device):
    """Load model state dict"""
    model.load_state_dict(torch.load(filepath, map_location=device))
    return model
