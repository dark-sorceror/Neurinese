import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch.nn.functional as F

from mdn import MDN

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
        input_size: int = 3,
        hidden_size: int = 256, 
        latent_size: int = 64, 
        num_layers: int = 1
    ):
        super().__init__()

        self.model = nn.LSTM(
            input_size = input_size,
            hidden_size = hidden_size,
            num_layers = num_layers,
            batch_first = True,
            bidirectional = True
        )
        
        self.m_h = nn.Linear(
            in_features = hidden_size * 2, 
            out_features = latent_size
        )
        self.lv_h = nn.Linear(
            in_features = hidden_size * 2, 
            out_features = latent_size
        )

    def forward(self, stroke_seq: torch.Tensor):
        # stroke_seq shape: (batch_size, seq_len, input_size = 3)

        # Final hidden state
        _, (h_n, _) = self.model(stroke_seq)
        
        h_forward = h_n[-2, :, :]
        h_backward = h_n[-1, :, :]
        
        h = torch.cat([h_forward, h_backward], dim = 1)

        mean_dist = self.m_h(h)
        log_var = self.lv_h(h)

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
        num_mixtures: int = 20
    ):
        super().__init__()

        self.model = nn.LSTM(
            input_size = latent_size * 2,
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
        
        self.z_to_cell = nn.Linear(
            in_features = latent_size,
            out_features = hidden_size
        )
        
        self.output = nn.Linear(
            in_features = hidden_size, 
            out_features = 3 + num_mixtures * 6
        )
        
    def forward(
        self, 
        z: torch.Tensor,
        stroke_seq: torch.Tensor, 
        hidden_state: torch.Tensor = None
    ):
        # stroke_seq shape: (batch_size, seq_len, input_size = 3)
        # z shape: (batch_size, latent_size = 64)
        
        seq_len = stroke_seq.size(1)
        z_ext = z.unsqueeze(1).repeat(1, seq_len, 1)
        
        # Teacher Forcing: Taking directly from the stroke dataset
        stroke_seq_emb = F.relu(self.embedding(stroke_seq))
        
        dec_in = torch.cat([stroke_seq_emb, z_ext], dim = -1)
        
        if hidden_state is None:
            # [h_0 ; c_0] = tanh(W_z * z + b_z)
            h_0 = torch.tanh(self.z_to_hidden(z))
            c_0 = torch.tanh(self.z_to_cell(z))
            
            hidden_state = (h_0.unsqueeze(0), c_0.unsqueeze(0))
        
        out, hidden_state = self.model(dec_in, hidden_state)
        
        params = self.output(out)
        
        return params, hidden_state

class KLDivergence(nn.Module):
    def forward(
        self, 
        log_var: torch.Tensor, 
        mean_dist: torch.Tensor
    ):
        return -0.5 * torch.sum(1 + log_var - mean_dist ** 2 - torch.exp(log_var))

class ReconstructionLoss(nn.Module):
    def __init__(self, num_mixtures: int):
        super().__init__()
        
        self.mdn_loss = MDN(num_mixtures = num_mixtures)
        
        self.kl_loss = KLDivergence()
        
    def forward(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor, 
        mean_dist: torch.Tensor, 
        log_var: torch.Tensor
    ):
        pen_logits = pred[..., :3]
        mdn_params = pred[..., 3:]
        
        coor_loss = self.mdn_loss(mdn_params, target)
        
        pen_target = target[..., 2:5]
        pen_target_idx = torch.argmax(pen_target, dim =- 1)
        pen_loss = F.cross_entropy(
            input = pen_logits.view(-1, 3), 
            target = pen_target_idx.view(-1)
        )
        
        kl_loss = self.kl_loss(log_var, mean_dist) / pred.size(0)
        
        return coor_loss + pen_loss, kl_loss
    
class StrokeModel(nn.Module):
    def __init__(
        self, 
        input_size: int = 5, 
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
        
        # Teacher forcing by shifting inputs (0, ..., i - 1)
        batch_size = stroke_seq.size(0)
        sos = torch.tensor(
            [0, 0, 1, 0, 0], 
            dtype = torch.float32, 
            device = stroke_seq.device
        )
        sos = sos.view(1, 1, 5).repeat(batch_size, 1, 1)
        
        dec_in = torch.cat([sos, stroke_seq[:, :-1, :]], dim = 1)
        out, _ = self.decoder(z, dec_in)
        
        return out, mean_dist, log_var