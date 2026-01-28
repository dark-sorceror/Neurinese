import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

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
        
    def sample(self, pred: torch.Tensor):
        # pred shape: (1, 1, output_size)
        
        mdn_params = pred[0, 0, 1:]
        num_mixtures = 20
        
        pi_l = mdn_params[:num_mixtures]
        
        gaussian_params = mdn_params[num_mixtures:].view(num_mixtures, 5)
        
        mu_x = gaussian_params[:, 0]
        mu_y = gaussian_params[:, 1]
        sigma_x = torch.exp(gaussian_params[:, 2])
        sigma_y = torch.exp(gaussian_params[:, 3])
        rho = torch.tanh(gaussian_params[:, 4])
        
        temperature = 0.8 
        pi = F.softmax(pi_l / temperature, dim=0)

        k = torch.multinomial(pi, 1).item()

        z_x = torch.randn(1).to(self.device).item()
        z_y = torch.randn(1).to(self.device).item()
        
        chosen_mu_x = mu_x[k].item()
        chosen_mu_y = mu_y[k].item()
        chosen_sigma_x = sigma_x[k].item() * np.sqrt(temperature)
        chosen_sigma_y = sigma_y[k].item() * np.sqrt(temperature)
        chosen_rho = rho[k].item()

        dx = chosen_mu_x + chosen_sigma_x * z_x
        dy = chosen_mu_y + chosen_sigma_y * (chosen_rho * z_x + np.sqrt(1 - chosen_rho**2) * z_y)
        
        pen_logit = pred[0, 0, 0]
        
        pen_prob = torch.sigmoid(pen_logit)
        pen_state = 1 if torch.rand(1).item() < pen_prob else 0
        
        return dx, dy, pen_state
    
        # https://arxiv.org/pdf/1704.03477 Goat research paper

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

            dx, dy, pen = self.sample(out)
            
            out_seq.append([dx, dy, pen])
            
            x = torch.tensor([[[dx, dy, pen]]], dtype=torch.float32).to(self.device)
            
            if pen == 1:
                pen_up_count += 1
            else:
                pen_up_count = 0
                
            if pen_up_count > 8:
                break
        
        return np.array(out_seq)

if __name__ == "__main__":
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
    single_sample = [to_relative(normalize(samples[0].astype(np.float32)))]

    dataset_obj = StrokeDataset(single_sample) 

    sample_tensor = dataset_obj[0] 
    sample_batch = sample_tensor.unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        mean_dist, log_var = model.encoder(sample_batch)
        z = mean_dist

    gen_strokes = generator.generate(z = z)
    plot_strokes(gen_strokes, multiple = False)

    # recon = generator.reconstruct(sample_tensor)
    # print(recon)
    
    # plot_strokes(sample_tensor.cpu().numpy(), multiple = False)
    # plot_strokes(recon, multiple = False)