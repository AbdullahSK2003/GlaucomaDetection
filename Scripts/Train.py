import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
import logging
from pathlib import Path
import json
from datetime import datetime
import random

# Import the model definition
from Model import ImprovedGlaucomaNet, get_model

# Assuming DataLoader is defined in a separate file
from DataLoader import get_data_loaders

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0, verbose=False):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.verbose = verbose

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            logging.info(f'Validation loss decreased ({self.best_loss:.6f} --> {val_loss:.6f}). Saving model ...')
        torch.save(model.state_dict(), path)
        self.best_loss = val_loss

class Trainer:
    def __init__(self, model, train_loader, val_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = next(model.parameters()).device
        
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.optimizer = self._get_optimizer()
        self.scheduler = self._get_scheduler()
        self.scaler = GradScaler()
        
        self.writer = SummaryWriter(log_dir=config['log_dir'])
        self.early_stopping = EarlyStopping(
            patience=config['early_stopping_patience'],
            verbose=True
        )
        
        self.best_val_acc = 0.0
        self._setup_logging()
        
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'lr': []
        }

    def _get_optimizer(self):
        return optim.AdamW(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )

    def _get_scheduler(self):
        return optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=self.config['T_0'],
            T_mult=self.config['T_mult'],
            eta_min=self.config['min_lr']
        )

    def _setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.config['log_dir'] / 'training.log'),
                logging.StreamHandler()
            ]
        )

    def train_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.config["num_epochs"]}')
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            self.optimizer.zero_grad()
            
            with autocast():
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
            
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            pbar.set_postfix({
                'loss': running_loss/(batch_idx+1),
                'acc': 100.*correct/total,
                'lr': self.optimizer.param_groups[0]['lr']
            })
        
        return running_loss/len(self.train_loader), 100.*correct/total

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, targets in self.val_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        return running_loss/len(self.val_loader), 100.*correct/total

    def train(self):
        for epoch in range(self.config['num_epochs']):
            train_loss, train_acc = self.train_epoch(epoch)
            val_loss, val_acc = self.validate()
            
            self.scheduler.step()
            
            # Logging
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            self.writer.add_scalar('Accuracy/train', train_acc, epoch)
            self.writer.add_scalar('Accuracy/val', val_acc, epoch)
            self.writer.add_scalar('LR', self.optimizer.param_groups[0]['lr'], epoch)
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['lr'].append(self.optimizer.param_groups[0]['lr'])
            
            logging.info(f'Epoch {epoch+1}: '
                         f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
                         f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_val_acc': self.best_val_acc,
                }, self.config['model_dir'] / 'best_model.pth')
            
            # Early stopping
            self.early_stopping(val_loss)
            if self.early_stopping.early_stop:
                logging.info(f"Early stopping triggered after {epoch + 1} epochs")
                break
        
        self.writer.close()
        return {
            'best_val_acc': self.best_val_acc,
            'history': self.history,
            'epochs_trained': epoch + 1
        }

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def save_model_state(model, trainer, config, save_dir='models'):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = save_dir / f"model_state.pth"
    
    state = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': trainer.optimizer.state_dict(),
        'scheduler_state_dict': trainer.scheduler.state_dict(),
        'best_val_acc': trainer.best_val_acc,
        'config': config
    }
    
    torch.save(state, save_path)
    logging.info(f"Model state saved to {save_path}")
    return save_path

def save_metadata(config, results, save_dir='models'):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metadata_path = save_dir / f"metadata.json"
    
    metadata = {
        # 'timestamp': timestamp,
        'config': {k: str(v) if isinstance(v, Path) else v for k, v in config.items()},
        'best_val_acc': results['best_val_acc'],
        'num_epochs_trained': results['epochs_trained'],
        'final_train_loss': results['history']['train_loss'][-1],
        'final_train_acc': results['history']['train_acc'][-1],
        'final_val_loss': results['history']['val_loss'][-1],
        'final_val_acc': results['history']['val_acc'][-1],
        'final_learning_rate': results['history']['lr'][-1],
        'early_stopping_triggered': results['epochs_trained'] < config['num_epochs']
    }
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    
    logging.info(f"Metadata saved to {metadata_path}")
    return metadata_path

def main():
    # Configuration
    config = {
        'data_dir': Path('data/raw'),
        'log_dir': Path('logs'),
        'model_dir': Path('models'),
        'num_epochs': 100,
        'batch_size': 32,
        'learning_rate': 1e-3,
        'weight_decay': 1e-4,
        'T_0': 10,
        'T_mult': 2,
        'min_lr': 1e-6,
        'early_stopping_patience': 10,
        'seed': 42
    }

    # Set up directories
    config['log_dir'].mkdir(parents=True, exist_ok=True)
    config['model_dir'].mkdir(parents=True, exist_ok=True)

    # Set seed for reproducibility
    set_seed(config['seed'])

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Get data loaders
    data_loaders = get_data_loaders(config['data_dir'], config['batch_size'])
    
    # Check the number of returned loaders
    if len(data_loaders) == 2:
        train_loader, val_loader = data_loaders
    elif len(data_loaders) == 3:
        train_loader, val_loader, test_loader = data_loaders
        logging.info("Test loader is available but won't be used in this training script.")
    else:
        raise ValueError(f"Unexpected number of data loaders: {len(data_loaders)}. Expected 2 or 3.")

    # Initialize model
    model = get_model(device)

    # Initialize trainer
    trainer = Trainer(model, train_loader, val_loader, config)

    # Train the model
    results = trainer.train()

    # Save final model state
    model_path = save_model_state(model, trainer, config)

    # Save training history
    history_path = config['model_dir'] / 'training_history.npy'
    np.save(history_path, results['history'])
    logging.info(f"Training history saved to {history_path}")

    # Save metadata
    metadata_path = save_metadata(config, results)

    logging.info(f"Training completed after {results['epochs_trained']} epochs.")
    logging.info(f"Best validation accuracy: {results['best_val_acc']:.2f}%")
    logging.info(f"Model saved to: {model_path}")
    logging.info(f"Training history saved to: {history_path}")
    logging.info(f"Metadata saved to: {metadata_path}")

    if results['epochs_trained'] < config['num_epochs']:
        logging.info("Note: Training stopped early due to early stopping criterion.")

if __name__ == '__main__':
    main()