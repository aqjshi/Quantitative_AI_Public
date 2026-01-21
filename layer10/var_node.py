import torch
import torch.nn as nn
import torch.optim as optim

# 1. Macro Inputs and Initial State
dem, frf, itl, usd_1998 = .15, .08, .05, .6
euro, usd_1999 = .28, .6 

# Dimensionality: 4 inputs (DEM, FRF, ITL, USD) -> 3 hidden features
d_in, d_out = 4, 3
W_1998 = torch.nn.Parameter(torch.randn(d_in, d_out))

# 2. Historical Covariance Matrix (Simulated)
# In production, this should be the covariance of the returns for these currencies
# It ensures we don't 'double-count' the signal from highly correlated parents.
cov_matrix = torch.eye(d_in) + 0.1  # Assuming slight positive correlation
# Explicitly high correlation between DEM (0) and FRF (1)
cov_matrix[0, 1] = cov_matrix[1, 0] = 0.9 

# 3. Define Structural Projection Matrix (P)
# Maps [DEM, FRF, ITL, USD] -> [EURO, USD]
# Row 1: Merges first three (Indices 0,1,2). Row 2: Retains USD (Index 3).
P = torch.tensor([
    [1.0, 1.0, 1.0, 0.0], # Euro merge
    [0.0, 0.0, 0.0, 1.0]  # USD pass-through
])

def mahalanobis_fusion(W_legacy, sigma, proj_matrix):
    """
    Solves the redundancy problem using GLS Projection:
    W_new = (P * Sigma * P.T)^-1 * P * Sigma * W_legacy
    """
    # System Variance in New Subspace
    sigma_p = torch.matmul(proj_matrix, torch.matmul(sigma, proj_matrix.t()))
    
    # Information Whitening
    inv_sigma_p = torch.inverse(sigma_p)
    
    # Covariance-Weighted Mapping
    mapping = torch.matmul(inv_sigma_p, torch.matmul(proj_matrix, sigma))
    W_spliced = torch.matmul(mapping, W_legacy)
    
    return W_spliced

# Perform Surgery on Weights
W_1999_raw = mahalanobis_fusion(W_1998.detach(), cov_matrix, P)
W_1999 = torch.nn.Parameter(W_1999_raw)

# 4. Optimizer State Surgery (Adam Momentum Warping)
optimizer = optim.Adam([W_1998], lr=0.001)

# Simulate training to populate optimizer buffers
loss = (W_1998.sum() - 0).pow(2)
loss.backward()
optimizer.step()

# Extract legacy buffers (Momentum m and Velocity v)
old_state = optimizer.state[W_1998]
m_old = old_state['exp_avg']
v_old = old_state['exp_avg_sq']

# Warp Optimizer Memory into the new 2-node shape
m_new = mahalanobis_fusion(m_old, cov_matrix, P)
v_new = mahalanobis_fusion(v_old, cov_matrix, P)

# 5. Re-initialize and Inject Memory
new_optimizer = optim.Adam([W_1999], lr=0.001)
new_optimizer.state[W_1999]['exp_avg'] = m_new
new_optimizer.state[W_1999]['exp_avg_sq'] = v_new
new_optimizer.state[W_1999]['step'] = old_state['step']

# 6. Verification
x_1999 = torch.tensor([euro, usd_1999])
output_1999 = torch.matmul(x_1999, W_1999)

print(f"Surgery Complete.")
print(f"Weight Shape: {W_1998.shape} -> {W_1999.shape}")
print(f"Output 1999: {output_1999.detach().numpy()}")