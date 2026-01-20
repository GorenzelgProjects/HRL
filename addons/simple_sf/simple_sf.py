"""
Simple Successor Features implementation.

Adapted from the setup document and Simple-SF repository.
Provides basis features phi(z) and successor features psi(z,a) for continual learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class SimpleSF(nn.Module):
    """
    Simple Successor Features module.
    
    Shares an encoder and adds:
    - phi_head(z) -> phi (L2-normalized basis features)
    - psi_head(z) -> psi(z,a) (successor features, shaped (B, A, d_sf))
    - w_task vector (learned parameter per level/task) updated via reward prediction
    """
    
    def __init__(self, encoder: nn.Module, num_actions: int, d_sf: int = 256):
        """
        Initialize Simple-SF module.
        
        Args:
            encoder: Shared pixel encoder (PixelEncoder instance)
            num_actions: Number of actions in the action space
            d_sf: Dimension of successor features
        """
        super().__init__()
        self.encoder = encoder
        self.num_actions = num_actions
        self.d_sf = d_sf
        
        # Get encoder output dimension
        # Access the LayerNorm to get normalized_shape
        if hasattr(encoder, 'ln') and hasattr(encoder.ln, 'normalized_shape'):
            encoder_dim = encoder.ln.normalized_shape[0]
        else:
            # Fallback: assume z_dim was passed or use default
            encoder_dim = getattr(encoder, 'z_dim', 256)
        
        # Basis features head: z -> phi (L2-normalized)
        self.phi = nn.Sequential(
            nn.Linear(encoder_dim, d_sf),
            nn.ReLU(),
            nn.Linear(d_sf, d_sf)
        )
        
        # Successor features head: z -> psi(z,a) for all actions
        self.psi = nn.Sequential(
            nn.Linear(encoder_dim, d_sf),
            nn.ReLU(),
            nn.Linear(d_sf, num_actions * d_sf)
        )
    
    def phi_feat(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute basis features phi(z) with L2 normalization.
        
        Args:
            z: Encoded features (B, z_dim)
        
        Returns:
            Normalized basis features (B, d_sf)
        """
        phi = self.phi(z)
        phi = F.normalize(phi, p=2, dim=-1)
        return phi
    
    def psi_feat(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute successor features psi(z,a) for all actions.
        
        Args:
            z: Encoded features (B, z_dim)
        
        Returns:
            Successor features (B, num_actions, d_sf)
        """
        out = self.psi(z)
        out = out.view(z.size(0), self.num_actions, self.d_sf)
        return out


def simple_sf_losses(
    sf: SimpleSF,
    w_task: nn.Parameter,
    obs: torch.Tensor,
    act: torch.Tensor,
    rew: torch.Tensor,
    next_obs: torch.Tensor,
    done: torch.Tensor,
    gamma: float = 0.99,
    encoder_target: Optional[nn.Module] = None,
    sf_target: Optional[SimpleSF] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Simple-SF losses: Q-SF TD loss and reward prediction loss.
    
    Args:
        sf: SimpleSF module
        w_task: Task vector parameter (learned)
        obs: Current observations (B, ...)
        act: Actions taken (B,) as long tensor
        rew: Rewards (B,)
        next_obs: Next observations (B, ...)
        done: Done flags (B,)
        gamma: Discount factor
        encoder_target: Optional target encoder (for stability)
        sf_target: Optional target SF module (for stability)
    
    Returns:
        Tuple of (L_psi, L_w):
        - L_psi: Q-SF TD loss
        - L_w: Reward prediction loss (with stop-gradient on phi)
    """
    # Current features
    z = sf.encoder(obs)
    psi = sf.psi_feat(z)  # (B, A, D)
    
    # Q-SF for taken actions: (B,)
    # Use einsum for consistency with original implementation
    # Equivalent to: (psi[torch.arange(z.size(0)), act] * w_task).sum(dim=-1)
    psi_selected = psi[torch.arange(z.size(0)), act]  # (B, D)
    q_sf = torch.einsum("bi,i->b", psi_selected, w_task)  # (B,)
    
    # Next features (using target networks if provided)
    with torch.no_grad():
        z2 = (encoder_target or sf.encoder)(next_obs)
        if sf_target is None:
            psi2 = sf.psi_feat(z2)
            phi2 = sf.phi_feat(z2)
        else:
            psi2 = sf_target.psi_feat(z2)
            phi2 = sf_target.phi_feat(z2)
        
        # Q-SF for next state: (B, A)
        # Compute Q-values for all actions: einsum("bai,i->ba", psi2, w_task)
        q2 = torch.einsum("bai,i->ba", psi2, w_task)  # (B, A)
        max_q2 = q2.max(dim=1).values  # (B,) - for discrete actions, use max
        
        # TD target
        y = rew + gamma * (1.0 - done.float()) * max_q2
    
    # Q-SF TD loss (matches original: 0.5 * MSE)
    L_psi = 0.5 * F.mse_loss(q_sf, y)
    
    # Reward prediction loss (updates w_task only; stop-grad on phi2)
    # Uses next state's basis features (matches original implementation)
    r_pred = torch.einsum("bi,i->b", phi2.detach(), w_task)  # (B,)
    L_w = 0.5 * F.mse_loss(r_pred, rew)
    
    return L_psi, L_w
