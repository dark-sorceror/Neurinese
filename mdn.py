import torch
import numpy as np
import torch.nn.functional as F

target = torch.zeros(2, 5, 3)
target[:, :, 0] = 1.0
target[:, :, 1] = 0.0
target[:, :, 2] = 1.0

print(target)

# 'target' shape: (batch_size, seq_len, input_size = 3)
# 'do' shape: (batch_size, seq_len, 1 + 6 * num_mixtures)
    
dx_target = target[..., 0].unsqueeze(2)
dy_target = target[..., 1].unsqueeze(2)
pen_target = target[..., 2].unsqueeze(2)

print(dx_target)
print(dy_target)
print(pen_target)

do = torch.randn(2, 5, 1 + 6 * 20, requires_grad = True)

print(do)

eos = do[..., 0:1]
params = do[..., 1:].view(do.size(0), do.size(1), 20, 6)

print(do[..., 1:].view(2, 5, 20, 6)[:, :, :, 1].shape, do[..., 1:].view(2, 5, 20, 6)[:, :, :, 1])
print(do[..., 1:].view(2, 5, 20, 6)[:, :, :, 0])

log_pi = F.log_softmax(params[..., 0], dim = -1)
mu_x = params[..., 1]
mu_y = params[..., 2]

sigma_x = torch.exp(params[..., 3])
sigma_y = torch.exp(params[..., 4])

rho = torch.tanh(params[..., 5])

z_x = (dx_target - mu_x) / sigma_x
z_y = (dy_target - mu_y) / sigma_y

# https://www.probabilitycourse.com/chapter5/5_3_2_bivariate_normal_dist.php
z_term = z_x**2 + z_y**2 - 2 * rho * z_x * z_y

log_norm_const = -torch.log(2 * np.pi * sigma_x * sigma_y * torch.sqrt(1 - rho ** 2))
log_exp_term = -z_term / (2 * (1 - rho ** 2))
log_prob_components = log_norm_const + log_exp_term
log_prob = torch.logsumexp(log_pi + log_prob_components, dim = -1)

print(-log_prob.mean())

# https://deep-and-shallow.com/2021/03/20/mixture-density-networks-probabilistic-regression-for-uncertainty-estimation/