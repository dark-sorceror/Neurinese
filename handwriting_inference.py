import torch
import numpy as np
import torch.nn as nn

from mdn import MDN
from utils import plot_strokes
from preprocess import normalize, to_relative
from stroke_model import StrokeDataset, StrokeModel

class Handwrite:
    def __init__(
        self, 
        model: nn.Module,
        device: str = None
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        
        self.mdn = MDN(num_mixtures = 20)

    @torch.no_grad()
    def reconstruct(self, seq: StrokeDataset):
        self.model.eval()
        
        seq = seq.unsqueeze(0).to(self.device)
        
        mean, _ = self.model.encoder(seq)
        z = mean 
        
        sos = torch.tensor(
            [0, 0, 1, 0, 0], 
            dtype = torch.float32, 
            device = self.device
        )
        batch_size = seq.size(0)
        sos = sos.view(1, 1, 5).repeat(batch_size, 1, 1)
        decoder_input = torch.cat([sos, seq[:, :-1, :]], dim = 1)
        
        # out Shape: [1, seq_len, 121]
        out, _ = self.model.decoder(z, decoder_input)
        
        recon_seq = []
        
        for i in range(out.size(1)):
            step_pred = out[:, i : i + 1, :] 
            
            dx, dy, pen = self.mdn.sample_step(step_pred, temperature = 0.1)
            p1, p2, p3 = pen
            recon_seq.append([dx, dy, p1, p2, p3])
            
        return np.array(recon_seq)
    
    @torch.no_grad()
    def generate(self, seq: StrokeDataset, max_steps=150):
        self.model.eval()
        seq = seq.unsqueeze(0).to(self.device)
        mean, _ = self.model.encoder(seq)
        z = mean.to(self.device)
        
        x = torch.tensor(
            [[[0, 0, 1, 0, 0]]], 
            dtype = torch.float32, 
            device = self.device
        )
        hidden = None
        out_seq = []
        
        for _ in range(max_steps):
            out, hidden = self.model.decoder(z, x, hidden)
            dx, dy, pen = self.mdn.sample_step(out, temperature = 0.6)
            
            p1, p2, p3 = pen
            out_seq.append([dx, dy, p1, p2, p3])
            
            # SketchRNN third pen state
            if p3 == 1:
                break
                
            x = torch.tensor(
                [[[dx, dy, p1, p2, p3]]], 
                dtype = torch.float32,
                device = self.device
            )
            
        return np.array(out_seq)

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = StrokeModel(
        input_size = 5,
        hidden_size = 256,
        latent_size = 64,
        num_layers = 1
    ).to(device)

    model.load_state_dict(torch.load(
            "models/handwriting_model.pth", 
            map_location = device
        )
    )

    generator = Handwrite(model = model, device = device)

    samples = np.load("./data/strokes.npy", allow_pickle = True)

    raw_sample = samples[0]
    relative_sample = to_relative(raw_sample)
    single_sample = [normalize(relative_sample)]
    
    dataset_obj = StrokeDataset(single_sample) 
    sample_tensor = dataset_obj[0]
    
    recon = generator.reconstruct(sample_tensor)
    gen = generator.generate(sample_tensor)

    # Plotting the dataset itself
    plot_strokes(sample_tensor.numpy(), multiple = False)

    # Reconstruction inference method (using ground truth)
    plot_strokes(recon, multiple = False)

    # Generative (free hand) is buggy.
    # TODO: Experiment with masking to pad samples
    plot_strokes(gen, multiple = False)