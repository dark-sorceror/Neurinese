import torch
import torch.nn as nn
import torch.optim as optim

class CharacterRecognizingTrainer:
    def __init__(
        self, 
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

class HandwritingTrainer:
    def __init__(self, model, learning_rate = 0.0001):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def vae_loss_function(self, mdn_params, target_strokes, mean_dist, log_var):
        target_coords = target_strokes[:, :, :2]
        target_flags = target_strokes[:, :, 2:]

        pred_coords = mdn_params[:, :, :2]
        pred_flags = mdn_params[:, :, 2:]
        
        recon_loss_coords = torch.nn.functional.mse_loss(pred_coords, target_coords)
        recon_loss_flags = torch.nn.functional.binary_cross_entropy_with_logits(pred_flags, target_flags)
        
        drawing_loss = recon_loss_coords + recon_loss_flags

        # KL Divergence
        kl_loss = -0.5 * torch.sum(1 + log_var - mean_dist.pow(2) - log_var.exp())
        
        return drawing_loss + (0.01 * kl_loss)

    def train_step(self, batch_strokes, batch_char_ids):
        self.model.train()
        self.optimizer.zero_grad()
        
        batch_strokes = batch_strokes.to(self.device)
        batch_char_ids = batch_char_ids.to(self.device)

        mdn_params, mean_dist, log_var = self.model(batch_strokes, batch_char_ids)
        
        target = batch_strokes[:, 1:, :] 
        
        loss = self.vae_loss_function(mdn_params, target, mean_dist, log_var)
        
        loss.backward()
        self.optimizer.step()
        
        return loss.item()