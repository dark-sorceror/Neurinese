import torch
import torch.nn as nn
import torch.optim as optim

class Trainer:
    def __init__(self, 
                 model: nn.Module, 
                 lr = 0.01, 
                 scheduler = None, 
                 device = None
                ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.criterion = nn.CrossEntropyLoss(label_smoothing = 0.1)
        self.optimizer = optim.Adam(
            model.parameters(),
            lr = lr,
            weight_decay = 0.0001
        )
        self.scheduler = scheduler
        self.history = {
            "train_loss": [],
            "val_loss": []
        }
        
    def train(self, loader):
        self.model.train()
        
        total_loss = 0.0
        
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            
            self.optimizer.zero_grad()
            
            pred = self.model(x_batch)
            loss = self.criterion(pred, y_batch)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item() * x_batch.size(0)
            
        return total_loss / len(loader.dataset)
    
    def validate(self, loader):
        self.model.eval()
        
        total_loss = 0.0
        
        with torch.no_grad():
            for x_batch, y_batch in loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                pred = self.model(x_batch)
                loss = self.criterion(pred, y_batch)
                
                total_loss += loss.item() * x_batch.size(0)
            
        return total_loss / len(loader.dataset)
    
    def fit(self, train_loader, val_loader, epochs: int, checkpoint_path = None):
        best_val_loss = float("inf")
        
        for epoch in range(epochs):
            train_loss = self.train(train_loader)
            val_loss = self.validate(val_loader)
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0 
                
                torch.save(self.model.state_dict(), checkpoint_path)
                
                print(f"Epoch {epoch:3d}/{epochs}: Training Loss: {train_loss:.4f} Validation Loss: {val_loss:.4f} (Saved best model)")
            else:
                epochs_no_improve += 1
                
                print(f"Epoch {epoch:3d}/{epochs}: Training Loss: {train_loss:.4f} Validation Loss: {val_loss:.4f} (No improvement)")
                
            if epochs_no_improve >= 5:
                print(f"Early stop")
                
                self.model.load_state_dict(torch.load(checkpoint_path))
                
                break
                
            if self.scheduler is not None and isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_loss)
                
        print(f"Training finished. Best Validation Loss: {best_val_loss:.4f}")