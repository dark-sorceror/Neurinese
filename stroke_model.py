import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch.nn.functional as F

class StrokeDataset(Dataset):
    def __init__(self, data):
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        seq = self.data[index]
        
        seq_tensor = torch.tensor(seq, dtype = torch.float32)
        
        return seq_tensor

class StrokeEncoder(nn.Module):
    def __init__(
        self, 
        input_size = 3,
        hidden_size = 256, 
        latent_size = 64, 
        num_layers = 1
    ):
        super().__init__()

        self.model = nn.LSTM(
            input_size = input_size,
            hidden_size = hidden_size,
            num_layers = num_layers,
            batch_first = True
        )
        
        self.m_h = nn.Linear(
            in_features = hidden_size, 
            out_features = latent_size
        )
        self.lv_h = nn.Linear(
            in_features = hidden_size, 
            out_features = latent_size
        )

    def forward(self, stroke_seq: torch.Tensor):
        # stroke_seq shape: (batch_size, seq_len, input_size = 3)

        # Final hidden state
        _, (h_n, _) = self.model(stroke_seq)
        
        h_n = h_n[-1]

        mean_dist = self.m_h(h_n)
        log_var = self.lv_h(h_n)

        return mean_dist, log_var

    def reparameterize(self, mean_dist: torch.Tensor, log_var: torch.Tensor):
        # https://medium.com/data-science/reparameterization-trick-126062cfd3c3
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        
        return mean_dist + eps * std

class StrokeDecoder(nn.Module):
    def __init__(
        self,
        input_size: int = 3,
        hidden_size: int = 256, 
        latent_size: int = 64,
        num_layers: int = 1,
    ):
        super().__init__()

        self.model = nn.LSTM(
            input_size = 64 + latent_size,
            hidden_size = hidden_size,
            num_layers = num_layers,
            batch_first = True
        )
        
        self.embedding = nn.Linear(
            in_features = input_size,
            out_features = latent_size
        )
        
        self.z_to_hidden = nn.Linear(
            in_features = latent_size, 
            out_features = hidden_size
        )
        
        self.output = nn.Linear(
            in_features = hidden_size, 
            out_features = input_size
        )
        
    def forward(
        self, 
        z: torch.Tensor,
        stroke_seq: torch.Tensor, 
        hidden_state: bool = None
    ):
        # stroke_seq shape: (batch_size, seq_len, input_size = 3)
        # z shape: (batch_size, latent_size = 64)
        
        seq_len = stroke_seq.size(1)
        z_ext = z.unsqueeze(1).repeat(1, seq_len, 1)
        stroke_seq_emb = F.relu(self.embedding(stroke_seq))
        
        dec_in = torch.cat([stroke_seq_emb, z_ext], dim = -1)
        
        if not hidden_state:
            h_0 = torch.tanh(self.z_to_hidden(z))
            c_0 = torch.zeros_like(h_0)
            
            hidden_state = (h_0.unsqueeze(0), c_0.unsqueeze(0))
        
        out, hidden_state = self.model(dec_in, hidden_state)
        
        output = self.output(out)
        
        return output, hidden_state

class KLDivergenceLoss(nn.Module):
    def forward(
        self, 
        log_var: torch.Tensor, 
        mean_dist: torch.Tensor
    ):
        return -0.5 * torch.sum(1 + log_var - mean_dist ** 2 - torch.exp(log_var))

class ReconstructionLoss(nn.Module):
    def forward(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor, 
        mean_dist: torch.Tensor, 
        log_var: torch.Tensor
    ):
        coor_loss = F.mse_loss(
            input = pred[..., :2], 
            target = target[..., :2]
        ) * 20.0
        
        pen_loss = F.binary_cross_entropy_with_logits(
            input = pred[..., 2],
            target = target[..., 2],
            pos_weight = torch.tensor(
                [5.0], 
                device = pred.device
            )
        )
        
        kl = KLDivergenceLoss()
        
        kl_loss = kl(log_var, mean_dist)
        kl_loss = kl_loss / pred.size(0)
        
        return coor_loss + pen_loss + (kl_loss * 0.05)
    
class StrokeModel(nn.Module):
    def __init__(
        self, 
        input_size: int = 3, 
        hidden_size: int = 256, 
        latent_size: int = 64, 
        num_layers: int = 1
    ):
        super().__init__()
        
        self.encoder = StrokeEncoder(
            input_size = input_size,
            hidden_size = hidden_size,
            latent_size = latent_size,
            num_layers = num_layers
        )
        
        self.decoder = StrokeDecoder(
            input_size = input_size,
            hidden_size = hidden_size,
            latent_size = latent_size,
            num_layers = num_layers
        )
    
    def forward(self, stroke_seq: torch.Tensor):
        mean_dist, log_var = self.encoder(stroke_seq)
        z = self.encoder.reparameterize(mean_dist, log_var)
        
        # Teacher forcing by shifting inputs (i + 1)
        batch_size = stroke_seq.size(0)
        sos = torch.zeros(batch_size, 1, 3, device = stroke_seq.device)
        
        dec_in = torch.cat([sos, stroke_seq[:, :-1, :]], dim = 1)
        out, _ = self.decoder(z, dec_in)
        
        return out, mean_dist, log_var