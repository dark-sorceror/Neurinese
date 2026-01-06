import torch
import numpy as np
import torch.nn as nn

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

    @torch.no_grad()
    def reconstruct(self, seq: StrokeDataset):
        self.model.eval()
        
        seq = seq.unsqueeze(0).to(self.device)
        mean_dist, log_var = self.model.encoder(seq)
        z = mean_dist
        
        out, _ = self.model.decoder(z, seq)
        
        return out.squeeze(0).cpu().numpy()
    
    @torch.no_grad()
    def generate(self, z: StrokeDataset, max_steps = 150):
        self.model.eval()
        
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
            
            dx_val = dx.item()
            dy_val = dy.item()
            pen_val = pen.item()
            
            out_seq.append([dx_val, dy_val, pen_val])
            
            x = torch.tensor([[[dx_val, dy_val, pen_val]]], dtype=torch.float32).to(self.device)
            
            if pen < 0.5:
                pen_up_count += 1
            else:
                pen_up_count = 0
                
            if pen_up_count > 8:
                break
        
        return np.array(out_seq)

device = "cuda" if torch.cuda.is_available() else "cpu"

model = StrokeModel(
    input_size = 3,
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
single_raw = samples[0].astype(np.float32)
single_rel = to_relative(normalize(single_raw))
sample_tensor = torch.tensor(single_rel, dtype = torch.float32).unsqueeze(0).to(device)

model.eval()
with torch.no_grad():
    mean_dist, log_var = model.encoder(sample_tensor)
    z = mean_dist

gen_strokes = generator.generate(z = z)
plot_strokes(gen_strokes, multiple = False)