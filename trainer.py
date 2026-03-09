import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F

from stroke_model import ReconstructionLoss

class CharacterRecognizingTrainer:
    def __init__(
        self, 
        model: nn.Module, 
        learning_rate: float, 
        device: str = None
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optimizer = optim.Adam(
            model.parameters(),
            lr = learning_rate,
            weight_decay = 0.0001
        )
        
        self.criterion = nn.CrossEntropyLoss(label_smoothing = 0.1)
        
        self.history = {
            "train_loss": [],
            "val_loss": []
        }
    
    def train(self, loader: DataLoader):
        self.model.train()
        
        total_loss = 0.0
        
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            
            pred = self.model(x_batch)
            loss = self.criterion(pred, y_batch)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item() * x_batch.size(0)
            
        return total_loss / len(loader.dataset)
    
    @torch.no_grad()
    def validate(self, loader: DataLoader):
        self.model.eval()
        
        total_loss = 0.0

        for x_batch, y_batch in loader:
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            
            pred = self.model(x_batch)
            loss = self.criterion(pred, y_batch)
            
            total_loss += loss.item() * x_batch.size(0)
            
        return total_loss / len(loader.dataset)
    
    def fit(
        self, 
        train_loader: DataLoader, 
        val_loader: DataLoader, 
        epochs: int,
        patience: int,
        checkpoint_path: str = None
    ):
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
                
                print(f"Epoch {epoch:3d}/{epochs}: Training Loss: {train_loss:.4f} Validation Loss: {val_loss:.4f} (No improvement) x{epochs_no_improve}")
                
            if epochs_no_improve >= patience:     
                self.model.load_state_dict(torch.load(checkpoint_path))
                
                break
                
        print(f"Training finished. Best Validation Loss: {best_val_loss:.4f}")

class HandwritingTrainer:
    def __init__(
        self, 
        model: nn.Module, 
        learning_rate: float, 
        device: str = None
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optimizer = optim.Adam(
            model.parameters(), 
            lr = learning_rate
        )
        
        self.criterion = ReconstructionLoss(num_mixtures = 20)
        
        self.history = {
            "train_loss": [],
            "val_loss": []
        }
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        
        return mu + eps * std
    
    def train(self, loader: DataLoader, kl_w: float):
        self.model.train()
        
        total_loss = 0.0

        sos = torch.tensor(
            [0, 0, 1, 0, 0], 
            dtype = torch.float32, 
            device = self.device
        )
        
        for i, (seq, char_id) in enumerate(loader):
            seq = seq.to(self.device)
            char_id = char_id.to(self.device)
            
            decoder_input_seq = seq[:, :-1, :].clone()
            mask = torch.rand_like(decoder_input_seq[:, :, 0]) < 0.20
            decoder_input_seq[mask] = 0.0
            
            batch_size = seq.size(0)
            sos_b = sos.view(1, 1, 5).repeat(batch_size, 1, 1)
            decoder_input = torch.cat([sos_b, decoder_input_seq], dim = 1)

            mean, log_var = self.model.encoder(seq, char_id)
            z = self.reparameterize(mean, log_var)

            # Normal reconstruction loss
            pred, _ = self.model.decoder(z, decoder_input, char_id)
            loss, kl = self.criterion(pred, seq, mean, log_var)

            # Cross-character style transfer loss
            shuffled_char_id = char_id[torch.randperm(char_id.size(0))]

            if not torch.all(shuffled_char_id == char_id):
                pred_cross, _ = self.model.decoder(z, decoder_input, shuffled_char_id)

                pen_logits  = pred_cross[..., :3]
                pen_target  = seq[..., 2:5]
                pen_target_idx = torch.argmax(pen_target, dim = -1)
                cross_pen_loss = F.cross_entropy(
                    pen_logits.view(-1, 3),
                    pen_target_idx.view(-1)
                ) * 0.1

                loss = loss + cross_pen_loss

            loss = loss + kl * kl_w
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm = 1.0)
            self.optimizer.step()
            
            total_loss += loss.item() * seq.size(0)
        
        return total_loss / len(loader.dataset)

    @torch.no_grad()
    def validate(self, loader: DataLoader, kl_w: float):
        self.model.eval()
        total_loss = 0.0
        
        sos = torch.tensor(
            [0, 0, 1, 0, 0], 
            dtype = torch.float32, 
            device = self.device
        )
        
        for i, (seq, char_id) in enumerate(loader):
            seq = seq.to(self.device)
            char_id = char_id.to(self.device)
            
            batch_size = seq.size(0)
            sos_b = sos.view(1, 1, 5).repeat(batch_size, 1, 1)
            decoder_input = torch.cat([sos_b, seq[:, :-1, :]], dim=1)
            
            mean, log_var = self.model.encoder(seq, char_id)
            z = mean 
            
            pred, _ = self.model.decoder(z, decoder_input, char_id)
            loss, kl = self.criterion(pred, seq, mean, log_var)

            shuffled_char_id = char_id[torch.randperm(char_id.size(0))]
            if not torch.all(shuffled_char_id == char_id):
                pred_cross, _ = self.model.decoder(z, decoder_input, shuffled_char_id)
                
                pen_logits = pred_cross[..., :3]
                pen_target_idx = torch.argmax(seq[..., 2:5], dim = -1)
                cross_pen_loss = F.cross_entropy(
                    pen_logits.view(-1, 3),
                    pen_target_idx.view(-1)
                ) * 0.1
                
                loss = loss + cross_pen_loss

            loss = loss + kl * kl_w 
            total_loss += loss.item() * seq.size(0)
        
        return total_loss / len(loader.dataset)

    def fit(
        self, 
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        patience: int,
        checkpoint_path: str = None
    ):
        best_val_loss = float("inf")
        epochs_no_improve = 0
        
        for epoch in range(epochs):
            kl_w = min(0.001, 0.001 * epoch / 20)
            train_loss = self.train(train_loader, kl_w)
            val_loss = self.validate(val_loader, kl_w)
            
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                torch.save(self.model.state_dict(), checkpoint_path)
                
                print(f"Epoch {epoch:3d}/{epochs}: Training Loss: {train_loss:.4f} Validation Loss: {val_loss:.4f} (Saved best model)")
            else:
                epochs_no_improve += 1
                
                print(f"Epoch {epoch:3d}/{epochs}: Training Loss: {train_loss:.4f} Validation Loss: {val_loss:.4f} (No improvement) x{epochs_no_improve}")
            
            if epochs_no_improve >= patience:
                self.model.load_state_dict(torch.load(checkpoint_path))
                
                break
                
        print(f"Training finished. Best Validation Loss: {best_val_loss:.4f}")