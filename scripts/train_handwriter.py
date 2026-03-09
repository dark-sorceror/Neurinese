import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from trainer import HandwritingTrainer
from preprocess import normalize, to_relative
from stroke_model import StrokeModel, StrokeDataset

DATA_PATH = "./data/strokes.npy"
MODEL_PATH = "./models/handwriting_model.pth"

def collate_fn(batch):
    seqs = [item[0] for item in batch]
    char_ids = torch.stack([item[1] for item in batch])
    
    padded_seqs = pad_sequence(seqs, batch_first = True)
    
    return padded_seqs, char_ids

def load_sample(raw):
    arr = np.array(raw)
    
    if arr.ndim == 2 and arr.shape[1] == 5:
        return normalize(arr)
    else:
        return normalize(to_relative(raw))

if __name__ == "__main__":
    samples = np.load(DATA_PATH, allow_pickle = True)

    processed_samples = []

    for raw, char_id in samples:
        seq = load_sample(raw)  

        processed_samples.append((seq, char_id))

    dataset = StrokeDataset(processed_samples)

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
        input_size = len(dataset[0][0][0]),
        hidden_size = 256,
        latent_size = 64,
        num_layers = 1,
        num_classes = 10,
        class_emb_dim = 32
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
    
    import matplotlib.pyplot as plt
    
    def plot_loss(history, save_path="./media/loss_curve.png"):
        fig, ax = plt.subplots(figsize=(10, 5))
        
        ax.plot(history["train_loss"], label="Train Loss", linewidth=2, color="#4f8ef7")
        ax.plot(history["val_loss"],   label="Val Loss",   linewidth=2, color="#f7704f")
        
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss (ELBO)")
        ax.set_title("Neurinese CVAE — Training Curve")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.show()

    # After trainer.fit():
    plot_loss(trainer.history)
    
    import json
    with open("./media/mdn_history.json", "w") as f:
        json.dump(trainer.history, f)