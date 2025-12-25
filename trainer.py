import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from utils import plot_strokes
from preprocess import normalize, to_relative
from stroke_model import StrokeModel, StrokeDataset, ReconstructionLoss

class CharacterRecognizingTrainer:
    def __init__(
        self, 
        model: nn.Module, 
        learning_rate: float, 
        device = None
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.criterion = nn.CrossEntropyLoss(label_smoothing = 0.1)
        self.optimizer = optim.Adam(
            model.parameters(),
            lr = learning_rate,
            weight_decay = 0.0001
        )
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
                
                print(f"Epoch {epoch:3d}/{epochs}: Training Loss: {train_loss:.4f} Validation Loss: {val_loss:.4f} (No improvement)")
                
            if epochs_no_improve >= 5:     
                self.model.load_state_dict(torch.load(checkpoint_path))
                
                break
                
        print(f"Training finished. Best Validation Loss: {best_val_loss:.4f}")

class HandwritingTrainer:
    def __init__(
        self, 
        model: nn.Module, 
        learning_rate: float, 
        device = None
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optimizer = optim.Adam(
            model.parameters(), 
            lr = learning_rate
        )
        
        self.criterion = ReconstructionLoss()
        
        self.history = {
            "train_loss": [],
            "val_loss": []
        }
    
    def train(self, loader: DataLoader):
        self.model.train()
        
        total_loss = 0.0
        
        for seq in loader:
            seq = seq.to(self.device)
            
            pred, mean_dist, log_var = self.model(seq)
            loss = self.criterion(pred, seq, mean_dist, log_var)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item() * seq.size(0)
        
        return total_loss / len(loader.dataset)
    
    @torch.no_grad()
    def validate(self, loader: DataLoader):
        self.model.eval()
        
        total_loss = 0.0
        
        for seq in loader:
            seq = seq.to(self.device)
            
            pred, mean_dist, log_var = self.model(seq)
            loss = self.criterion(pred, seq, mean_dist, log_var)
            
            total_loss += loss.item() * seq.size(0)
        
        return total_loss / len(loader.dataset)
    
    @torch.no_grad()
    def reconstruct(self, seq: StrokeDataset):
        self.model.eval()
        
        seq = seq.unsqueeze(0).to(self.device)
        mean_dist, log_var = self.model.encoder(seq)
        z = mean_dist
        
        out, _ = self.model.decoder(z, seq)
        
        return out.squeeze(0).cpu().numpy()
    
    @torch.no_grad()
    def generate(self, seq: StrokeDataset, max_steps = 150):
        self.model.eval()
        
        seq = seq.unsqueeze(0).to(self.device)
        mean_dist, _ = self.model.encoder(seq)
        z = mean_dist
        
        x = torch.zeros(1, 1, 3).to(self.device)
        
        hidden = None
        out_seq = []
        pen_up_count = 0
        
        for _ in range(max_steps):
            out, hidden = self.model.decoder(z, x, hidden)
            
            step = out[:, -1]
            dx, dy = step[:, 0], step[:, 1]
            pen_logit = step[:, 2]
            pen = torch.sigmoid(pen_logit)
            
            out_seq.append([dx.item(), dy.item(), pen.item()])
            
            x = torch.tensor([[[dx, dy, pen]]]).to(self.device)
            
            if pen < 0.5:
                pen_up_count += 1
            else:
                pen_up_count = 0
                
            if pen_up_count > 8:
                break
        
        return out_seq
        
    def fit(
        self, 
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int, 
        checkpoint_path: str = None
    ):
        best_val_loss = float("inf")
        
        for epoch in range(epochs):
            kl_w = min(0.05, epoch / 200)
            
            train_loss = self.train(train_loader)
            val_loss = self.validate(val_loader)
            
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                
                torch.save(self.model.state_dict(), checkpoint_path)
                
                print(f"Epoch {epoch:3d}/{epochs}: Training Loss: {train_loss:.4f} KL: {kl_w:.4f} Validation Loss: {val_loss:.4f} (Saved best model)")
            else:
                epochs_no_improve += 1
                
                print(f"Epoch {epoch:3d}/{epochs}: Training Loss: {train_loss:.4f} KL: {kl_w:.4f} Validation Loss: {val_loss:.4f} (No improvement)")
                
            if epochs_no_improve >= 5:
                self.model.load_state_dict(torch.load(checkpoint_path))
                
                break
                
        print(f"Training finished. Best Validation Loss: {best_val_loss:.4f}")

DATA_PATH = "./data/strokes.npy"
MODEL_PATH = "./model/handwriting_model.pth"

if __name__ == "__main__":
    collate_fn = lambda batch: pad_sequence(batch, batch_first = True)

    samples = np.load(DATA_PATH, allow_pickle = True)
    raw_data = [seq.astype(np.float32) for seq in samples]

    processed_samples = []

    for raw in raw_data:
        seq_abs = raw.copy()
        seq_abs = normalize(seq_abs)
        seq = to_relative(seq_abs)

        processed_samples.append(seq)

    # Overfit on a single sample - perfect memorization and learning
    single_sample = processed_samples[0]
    debug_samples = [single_sample for _ in range(100)]

    dataset = StrokeDataset(debug_samples)

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

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
        checkpoint_path = MODEL_PATH
    )

    sample = train_ds[0]
    recon = trainer.reconstruct(sample)
    gen = trainer.generate(sample)

    # Plotting the dataset itself
    plot_strokes(sample.numpy(), multiple = False)

    # Reconstruction inference method (using ground truth)
    plot_strokes(recon, multiple = False)

    # Generative (free hand) is buggy.
    # TODO: Experiment with masking to pad samples
    plot_strokes(gen, multiple = False)