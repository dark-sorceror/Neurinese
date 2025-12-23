import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch.functional as F

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
        input_dim = 4,
        hidden_size = 512, 
        latent_dim = 256, 
        num_layers = 2
    ):
        super().__init__()
        
        # Bi-directional LSTM to process the sequence from both directions
        self.model = nn.LSTM(
            input_size = input_dim,
            hidden_size = hidden_size,
            num_layers = num_layers,
            bidirectional = True,
            batch_first = True
        )
        
        output_size = 2 * hidden_size
        
        self.m_h = nn.Linear(
            in_features = output_size, 
            out_features = latent_dim
        )
        self.lv_h = nn.Linear(
            in_features = output_size, 
            out_features = latent_dim
        )

    def forward(self, stroke_seq):
        # stroke_seq shape: (batch_size, seq_len, 4)

        # Final hidden state
        h_s = self.model(stroke_seq)[1][0]

        final_forward = h_s[-2]
        final_backward = h_s[-1]
        final_state = torch.cat([final_forward, final_backward], dim = 1)

        mean_dist = self.m_h(final_state)
        log_var = self.lv_h(final_state)
        
        z = self.reparameterize(mean_dist, log_var)
        
        return z, mean_dist, log_var

    def reparameterize(self, mean_dist, log_var):
        # https://medium.com/data-science/reparameterization-trick-126062cfd3c3
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        
        return mean_dist + eps * std
    
class StrokeDecoder(nn.Module):
    def __init__(
        self,
        num_layers: int = 2,
        hidden_size: int = 512, 
        latent_dim: int = 256,
        num_mixtures: int = 5
    ):
        super().__init__()
        
        # Unidirectional LSTM for sequence generation
        self.model = nn.LSTM(
            input_size = latent_dim + 4,
            hidden_size = hidden_size,
            num_layers = 2,
            batch_first = True
        )
        
        self.initial = nn.Linear(
            in_features = latent_dim, 
            out_features = 2 * hidden_size * num_layers
        )

        # MDN params: (pi, mean_x, mean_y, sigma_x, sigma_y, rho)
        self.m_h = nn.Linear(
            in_features = hidden_size, 
            out_features = 6 * num_mixtures
        )
        
        self.p_h = nn.Linear(
            in_features = hidden_size,
            out_features = 2
        )
        
        self.num_mixtures = num_mixtures
        self.hidden_size = hidden_size
        
    def forward(
        self, 
        stroke_seq_input: torch.Tensor, 
        z: torch.Tensor
    ):
        # stroke_seq_input shape: (batch_size, seq_len, 4)
        # z shape: (batch_size, 256)
        
        states = self.initial(z).view(2, 2, stroke_seq_input.size()[0], self.hidden_size)
        h, c = states[0], states[1]
        
        z_ext = z.unsqueeze(1).expand(-1, stroke_seq_input.size()[1], -1)
        stroke_seq_input = torch.cat([stroke_seq_input, z_ext], dim = -1)
        
        output, (h_n, c_n) = self.model(stroke_seq_input, (h, c))
        
        mdn_params = self.mdn_linear(output)
        p_l = self.p_h(output)
        
        return mdn_params, p_l, (h_n, c_n)

class ReconstructionLoss(nn.Module):
    def split(self, mdn_params):
        p_l = mdn_params[:, :, -2:]
        mixture_params = mdn_params[:, :, :-2]
        mixture_params = mixture_params.view(mixture_params.size(0), mixture_params.size(1))
        
        pi, mean_x, mean_y, sigma_x, sigma_y, rho = torch.split(mixture_params, 1, dim = 3)
        
        pi = F.softmax(pi, dim = 2)
        sigma_x = torch.exp(sigma_x)
        sigma_y = torch.exp(sigma_y)
        rho = torch.tanh(rho)
        
        return pi, mean_x, mean_y, sigma_x, sigma_y, rho, p_l
        
    def forward(
        self, 
        mdn_params: torch.Tensor, 
        target: torch.Tensor, 
        num_mixtures: int = 5
    ):
        pi, mean_x, mean_y, sigma_x, sigma_y, rho, p_l = self.split(mdn_params, num_mixtures)
        
        # target shape: (batch, seq_len, 4)
        
        dx = target[:, :, 0:1].unsqueeze(2)
        dy = target[:, :, 1:2].unsqueeze(2)
        
        z_x = ((dx - mean_x) / sigma_x) ** 2
        z_y = ((dy -mean_y) / sigma_y) ** 2
        z_xy = (2 * rho * (dx - mean_x) * (dy - mean_y)) / (sigma_x * sigma_y)
        
        exp = z_x + z_y - z_xy
        norm = 2 * torch.pi * sigma_x * sigma_y * torch.sqrt(1 - (rho ** 2))
        
        prob = torch.exp(-exp / (2 * (1 - (rho ** 2)))) / norm
        prob_sum = torch.sum(pi * prob, dim=2)
        nll_stroke = -torch.log(prob_sum + 1e-6)
        
        pen_targets = target[:, :, 2:]
        nll_pen = F.binary_cross_entropy_with_logits(p_l, pen_targets, reduction = "none")
        
        total = nll_stroke.sum() + nll_pen.sum()
        
        return total

class HandwritingModel(nn.Module):
    def __init__(
        self, 
        num_classes = 2, 
        latent_dim = 256, 
        hidden_size = 512, 
        num_mixtures = 5
    ):
        super().__init__()
        
        self.encoder = StrokeEncoder(
            latent_dim = latent_dim, 
            hidden_size = hidden_size
        )
        self.decoder = StrokeDecoder(
            hidden_size = hidden_size, 
            num_mixtures = num_mixtures
        )

    def forward(self, stroke_seq, char_ids):
        # stroke_seq shape: (batch_size, seq_len, 4)
        
        z, mean_dist, log_var = self.encoder(stroke_seq)
        
        # wait
        
        mdn_params = self.decoder(
            char_ids = char_ids, 
            z = z, 
            stroke_seq_input = stroke_seq_input
        )[0]
        
        return mdn_params, mean_dist, log_var