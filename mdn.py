import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class MDN(nn.Module):
    def __init__(self, num_mixtures: int):
        super().__init__()
        
        self.num_mixtures = num_mixtures
        
    def forward(self, mdn_params: torch.Tensor, target: torch.Tensor):
        pi_logits = mdn_params[..., :self.num_mixtures] 
        gaussian_params = mdn_params[..., self.num_mixtures:].view(
            mdn_params.size(0), mdn_params.size(1), self.num_mixtures, 5
        )
        
        mu_x = gaussian_params[..., 0]
        mu_y = gaussian_params[..., 1]
        sigma_x_logits = gaussian_params[..., 2]
        sigma_y_logits = gaussian_params[..., 3]
        rho_logits = gaussian_params[..., 4]

        pi = F.softmax(pi_logits, dim = -1)
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
        
        # Moved pen state logits to CE loss

        return -torch.log(final_prob + 1e-6).mean()
    
    def sample_step(self, pred: torch.Tensor, temperature: int = 0.1):
        # 'pred' shape: (batch_size, seq_len, 1 + 6 * num_mixtures)
        
        pred = pred.squeeze()
        
        pen_logits = pred[:3] / temperature
        pen_probs = torch.nn.functional.softmax(pen_logits, dim = 0)
        pen_idx = torch.distributions.Categorical(pen_probs).sample().item()
        
        pen_state = [1.0 if i == pen_idx else 0.0 for i in range(3)]
        
        mdn_params = pred[3:]
        pi_logits = mdn_params[: self.num_mixtures] / temperature
        pi = torch.nn.functional.softmax(pi_logits, dim = 0)
        k = torch.distributions.Categorical(pi).sample().item()
        
        gaussian_params = mdn_params[self.num_mixtures:].view(self.num_mixtures, 5)
        params = gaussian_params[k]
        mu_x, mu_y, sigma_x_log, sigma_y_log, rho_log = params
        
        sigma_x = torch.exp(sigma_x_log) * np.sqrt(temperature)
        sigma_y = torch.exp(sigma_y_log) * np.sqrt(temperature)
        rho = torch.tanh(rho_log)
        
        mean = torch.tensor([mu_x, mu_y])
        cov_xy = rho * sigma_x * sigma_y
        covariance = torch.tensor([[sigma_x ** 2, cov_xy], [cov_xy, sigma_y ** 2]])
        
        mvn = torch.distributions.MultivariateNormal(mean, covariance)
        sample_coords = mvn.sample()
        
        return sample_coords[0].item(), sample_coords[1].item(), pen_state

# https://arxiv.org/pdf/1704.03477 Goat research paper

# https://medium.com/the-ml-intuition/youve-been-using-negative-log-loss-here-s-what-it-actually-means-7753e476d346

# https://github.com/dusenberrymw/mixture-density-networks/blob/master/mixture_density_networks.ipynb

# https://deep-and-shallow.com/2021/03/20/mixture-density-networks-probabilistic-regression-for-uncertainty-estimation/