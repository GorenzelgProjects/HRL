"""
Integration utilities for Simple-SF with Option-Critic training.

Provides helper functions to wire Simple-SF into the Option-Critic training loop.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Tuple
from pathlib import Path

from addons.encoder.pixel_encoder import make_pixel_encoder
from addons.simple_sf.simple_sf import SimpleSF, simple_sf_losses


def create_simple_sf_module(
    num_actions: int,
    z_dim: int = 256,
    d_sf: int = 256,
    grid_height: int = 17,
    grid_width: int = 19,
    device: Optional[torch.device] = None
) -> Tuple[SimpleSF, nn.Parameter]:
    """
    Create Simple-SF module and task vector for a level.
    
    Args:
        num_actions: Number of actions
        z_dim: Encoder output dimension
        d_sf: Successor features dimension
        grid_height: Grid height
        grid_width: Grid width
        device: Device to place modules on
    
    Returns:
        Tuple of (SimpleSF module, w_task parameter)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create shared encoder
    encoder = make_pixel_encoder(
        z_dim=z_dim,
        grid_height=grid_height,
        grid_width=grid_width,
        use_embedding=True
    ).to(device)
    
    # Create Simple-SF module (shares encoder)
    sf = SimpleSF(encoder, num_actions, d_sf).to(device)
    
    # Create task vector (per level)
    # Initialize with small random values (matching original implementation)
    # The original uses torch.randn() and optionally normalizes it
    w_task = nn.Parameter(torch.randn(d_sf, device=device) * 0.01)
    
    return sf, w_task


def compute_sf_losses(
    sf: SimpleSF,
    w_task: nn.Parameter,
    batch: Dict[str, torch.Tensor],
    gamma: float = 0.99,
    encoder_target: Optional[nn.Module] = None,
    sf_target: Optional[SimpleSF] = None
) -> Dict[str, torch.Tensor]:
    """
    Compute Simple-SF losses from a batch of transitions.
    
    Args:
        sf: SimpleSF module
        w_task: Task vector parameter
        batch: Dictionary with keys: 'obs', 'action', 'reward', 'next_obs', 'done'
        gamma: Discount factor
        encoder_target: Optional target encoder
        sf_target: Optional target SF module
    
    Returns:
        Dictionary with 'L_psi' and 'L_w' losses
    """
    L_psi, L_w = simple_sf_losses(
        sf=sf,
        w_task=w_task,
        obs=batch['obs'],
        act=batch['action'],
        rew=batch['reward'],
        next_obs=batch['next_obs'],
        done=batch['done'],
        gamma=gamma,
        encoder_target=encoder_target,
        sf_target=sf_target
    )
    
    return {'L_psi': L_psi, 'L_w': L_w}


def create_sf_optimizers(
    sf: SimpleSF,
    w_task: nn.Parameter,
    lr_main: float = 1e-3,
    lr_w: float = 1e-2
) -> Tuple[torch.optim.Optimizer, torch.optim.Optimizer]:
    """
    Create separate optimizers for main network and task vector.
    
    Args:
        sf: SimpleSF module
        w_task: Task vector parameter
        lr_main: Learning rate for main network (encoder + SF heads)
        lr_w: Learning rate for task vector
    
    Returns:
        Tuple of (main_optimizer, w_optimizer)
    """
    # Main optimizer: encoder + SF heads
    main_params = list(sf.encoder.parameters()) + list(sf.phi.parameters()) + list(sf.psi.parameters())
    opt_main = torch.optim.Adam(main_params, lr=lr_main)
    
    # Task vector optimizer
    opt_w = torch.optim.Adam([w_task], lr=lr_w)
    
    return opt_main, opt_w
