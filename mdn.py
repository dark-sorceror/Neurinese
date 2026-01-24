import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class MDN(nn.Module):
    def __init__(self, num_mixtures: int):
        super().__init__()
        
        self.num_mixtures = num_mixtures
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        # 'target' shape: (batch_size, seq_len, input_size = 3)
        # 'do' shape: (batch_size, seq_len, 1 + 6 * num_mixtures)
            
        pi_logits = pred[..., 1:1 + self.num_mixtures] 
        
        start_idx = 1 + self.num_mixtures
        gaussian_params = pred[..., start_idx:].view(pred.size(0), pred.size(1), self.num_mixtures, 5)
        
        mu_x = gaussian_params[..., 0]
        mu_y = gaussian_params[..., 1]
        sigma_x_logits = gaussian_params[..., 2]
        sigma_y_logits = gaussian_params[..., 3]
        rho_logits = gaussian_params[..., 4]

        pi = F.softmax(pi_logits, dim =- 1)
        
        sigma_x = torch.exp(sigma_x_logits) + 1e-6
        sigma_y = torch.exp(sigma_y_logits) + 1e-6
        
        rho = torch.tanh(rho_logits)

        dx_target = target[..., 0].unsqueeze(2)
        dy_target = target[..., 1].unsqueeze(2)

        z_x = (dx_target - mu_x) / sigma_x
        z_y = (dy_target - mu_y) / sigma_y
        rho_term = 1 - rho ** 2
        
        z_pow = z_x ** 2 + z_y ** 2 - 2 * rho * z_x * z_y
        exp_term = -z_pow / (2 * rho_term)
        
        norm_const = 2 * np.pi * sigma_x * sigma_y * torch.sqrt(rho_term)
        
        probs = (torch.exp(exp_term) / norm_const) + 1e-6
        
        final_prob = torch.sum(pi * probs, dim = -1)

        return -torch.log(final_prob + 1e-6).mean()

# https://arxiv.org/pdf/1704.03477 Goat research paper

# https://medium.com/the-ml-intuition/youve-been-using-negative-log-loss-here-s-what-it-actually-means-7753e476d346

# https://github.com/dusenberrymw/mixture-density-networks/blob/master/mixture_density_networks.ipynb

# https://deep-and-shallow.com/2021/03/20/mixture-density-networks-probabilistic-regression-for-uncertainty-estimation/