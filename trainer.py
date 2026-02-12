import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from mdn import MDN
from utils import plot_strokes
from preprocess import normalize, to_relative
from stroke_model import StrokeModel, StrokeDataset, ReconstructionLoss

DATA_PATH = "./data/strokes.npy"
MODEL_PATH = "./models/handwriting_model.pth"

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
        
        for i, seq in enumerate(loader):
            seq = seq.to(self.device)
            
            # Decoder Input: [SOS, A, B]
            batch_size = seq.size(0)
            sos = sos.view(1, 1, 5).repeat(batch_size, 1, 1)
            
            # Input to Decoder = [SOS] + [Seq excluding last step], then full seq, then history
            decoder_input = torch.cat([sos, seq[:, :-1, :]], dim=1)
            mean, log_var = self.model.encoder(seq)
            z = self.reparameterize(mean, log_var)
            pred, _ = self.model.decoder(z, decoder_input)
            loss, kl = self.criterion(pred, seq, mean, log_var)
            
            loss = loss + kl * kl_w
            
            self.optimizer.zero_grad()
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(parameters = self.model.parameters(), max_norm = 1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item() * seq.size(0)
        
        return total_loss / len(loader.dataset)
    
    @torch.no_grad()
    def validate(self, loader: DataLoader):
        self.model.eval()
        total_loss = 0.0
        
        sos = torch.tensor(
            [0, 0, 1, 0, 0], 
            dtype = torch.float32, 
            device = self.device
        )
        
        for i, seq in enumerate(loader):
            kl_w = min(0.05, 0.05 * (i / 2000))
            seq = seq.to(self.device)
            
            batch_size = seq.size(0)
            sos = sos.view(1, 1, 5).repeat(batch_size, 1, 1)
            decoder_input = torch.cat([sos, seq[:, :-1, :]], dim = 1)
            
            mean, log_var = self.model.encoder(seq)
            z = mean 
            
            pred, _ = self.model.decoder(z, decoder_input)
            
            loss, kl = self.criterion(pred, seq, mean, log_var)
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
        
        for epoch in range(epochs):
            kl_w = min(0.05, 0.05 * epoch / 20)
            train_loss = self.train(train_loader, kl_w)
            val_loss = self.validate(val_loader)
            
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

if __name__ == "__main__":
    collate_fn = lambda batch: pad_sequence(batch, batch_first = True)

    samples = np.load(DATA_PATH, allow_pickle = True)

    processed_samples = []

    for raw in samples:
        seq_rel = to_relative(raw) 
        seq_final = normalize(seq_rel)

        processed_samples.append(seq_final)

    # Overfit on a single sample - perfect memorization and learning
    single_sample = processed_samples[0]
    debug_samples = [single_sample for _ in range(300)]

    dataset = StrokeDataset(debug_samples)

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

    # Train with batch size of one to prevent any padding
    train_loader = DataLoader(
        dataset = train_ds, 
        batch_size = 1, 
        shuffle = True, 
        collate_fn = collate_fn
    )
    val_loader = DataLoader(
        dataset = val_ds, 
        batch_size = 1, 
        shuffle = False, 
        collate_fn = collate_fn
    )

    model = StrokeModel(
        input_size = len(single_sample[0]),
        hidden_size = 256,
        latent_size = 64,
        num_layers = 1
    )
    trainer = HandwritingTrainer(
        model = model, 
        learning_rate = 0.001
    )
    trainer.fit(
        train_loader = train_loader, 
        val_loader = val_loader, 
        epochs = 100,
        patience = 10,
        checkpoint_path = MODEL_PATH
    )