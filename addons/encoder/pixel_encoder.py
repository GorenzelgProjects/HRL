"""
Shared pixel encoder module.

Provides a CNN-style encoder that converts pixel observations to feature vectors.
This is a mockup implementation that can later be swapped for a real CNN.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


class PixelEncoder(nn.Module):
    """
    CNN-style encoder that converts pixel observations to feature vectors.
    
    This encoder follows the structure from the setup doc:
    - Conv stack + FC + LayerNorm
    - Handles inputs as either [0,255] or [0,1] and normalizes if necessary
    - Outputs feature vectors z_t with shape (B, z_dim)
    
    For Thin Ice, since observations are flattened grid integers, we convert
    them to a pseudo-image format using an embedding layer.
    """
    
    def __init__(
        self, 
        in_channels: int = 1,
        z_dim: int = 256,
        grid_height: int = 17,
        grid_width: int = 19,
        use_embedding: bool = True,
        num_tile_types: int = 13,
        feature_vector_size: Optional[int] = None
    ):
        """
        Initialize the pixel encoder.
        
        Args:
            in_channels: Number of input channels (default 1 for grayscale)
            z_dim: Dimension of output feature vector
            grid_height: Height of the grid (for Thin Ice, typically 17)
            grid_width: Width of the grid (for Thin Ice, typically 19)
            use_embedding: If True, use embedding layer for tile types (for grid-based obs)
            num_tile_types: Number of distinct tile types (for embedding)
            feature_vector_size: If provided, use MLP encoder for feature vectors instead of CNN
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.z_dim = z_dim
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.use_embedding = use_embedding
        self.feature_vector_size = feature_vector_size
        self.expected_grid_size = grid_height * grid_width
        
        # If feature_vector_size is provided, use MLP encoder instead of CNN
        if feature_vector_size is not None:
            self.use_mlp = True
            self.mlp = nn.Sequential(
                nn.Linear(feature_vector_size, 256),
                nn.ReLU(),
                nn.Linear(256, 512),
                nn.ReLU(),
                nn.Linear(512, z_dim)
            )
            self.ln = nn.LayerNorm(z_dim)
            # CNN components not needed
            self.embedding = None
            self.conv = None
            self.adaptive_pool = None
            self.fc = None
        else:
            self.use_mlp = False
            if use_embedding:
                # For grid-based observations: embed tile types to channels
                self.embedding = nn.Embedding(num_tile_types, in_channels)
                # After embedding, we have (B, H, W, C) -> (B, C, H, W)
            else:
                self.embedding = None
            
            # CNN layers (assuming input will be resized/processed to ~84x84 or similar)
            # For now, we'll adapt to the actual grid size
            # Calculate conv output size
            # After conv layers: assuming 84x84 -> 7x7 (with the given stride/padding)
            # For grid-based: we'll use adaptive pooling to handle variable sizes
            
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, 32, kernel_size=8, stride=4, padding=0),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
                nn.ReLU(),
            )
            
            # Use adaptive pooling to handle variable input sizes
            self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))
            
            # Calculate FC input size: 64 * 7 * 7 = 3136
            fc_input_size = 64 * 7 * 7
            self.fc = nn.Linear(fc_input_size, z_dim)
            self.ln = nn.LayerNorm(z_dim)
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: convert observation to feature vector.
        
        Args:
            obs: Observation tensor. Can be:
                - (B, H*W) flattened grid integers -> converted to (B, 1, H, W)
                - (B, feature_size) feature vectors -> processed by MLP
                - (B, C, H, W) pixel images
                - (H*W,) or (H, W) single observation -> converted to (1, 1, H, W)
        
        Returns:
            Feature vector z with shape (B, z_dim)
        """
        # Handle different input formats
        if obs.dim() == 1:
            # Single observation
            obs = obs.unsqueeze(0)  # (1, ...)
        
        batch_size = obs.shape[0]
        obs_size = obs.shape[-1] if obs.dim() == 2 else obs.numel() // batch_size
        
        # Check if we should use MLP encoder (feature vector)
        if self.use_mlp or (obs_size != self.expected_grid_size and obs.dim() <= 2):
            # Feature vector mode: use MLP
            if obs.dim() > 2:
                obs = obs.view(batch_size, -1)
            else:
                obs = obs.float()
            
            # Normalize if needed
            if obs.max() > 1.5:
                obs = obs / 255.0
            
            # Apply MLP
            z = self.mlp(obs)  # (B, z_dim)
            z = self.ln(z)
            return z
        
        # Grid-based mode: use CNN
        if obs.dim() == 2:
            # Flattened grid: (B, H*W) -> (B, H, W)
            if obs_size == self.expected_grid_size:
                obs = obs.view(batch_size, self.grid_height, self.grid_width)
            else:
                # Try to infer dimensions or use adaptive approach
                # For now, assume it's a flattened grid that doesn't match expected size
                # Use a square root approximation
                side_len = int(np.sqrt(obs_size))
                if side_len * side_len == obs_size:
                    obs = obs.view(batch_size, side_len, side_len)
                else:
                    # Fallback: treat as feature vector
                    obs = obs.float()
                    if obs.max() > 1.5:
                        obs = obs / 255.0
                    # Use a simple MLP fallback
                    if not hasattr(self, '_fallback_mlp'):
                        self._fallback_mlp = nn.Sequential(
                            nn.Linear(obs_size, 256),
                            nn.ReLU(),
                            nn.Linear(256, self.z_dim)
                        ).to(obs.device)
                    z = self._fallback_mlp(obs)
                    return self.ln(z)
        elif obs.dim() == 2 and obs.shape[0] == self.grid_height and obs.shape[1] == self.grid_width:
            # Already (H, W) -> add batch dimension
            obs = obs.unsqueeze(0)
        
        # Now obs is (B, H, W) with integer tile types
        if self.use_embedding and self.embedding is not None:
            # Convert tile types to embeddings: (B, H, W) -> (B, H, W, C)
            obs = self.embedding(obs.long())
            # Permute to (B, C, H, W)
            obs = obs.permute(0, 3, 1, 2)
        else:
            # Treat as grayscale image: (B, H, W) -> (B, 1, H, W)
            if obs.dim() == 3:
                obs = obs.unsqueeze(1)
            # Normalize if needed
            if obs.max() > 1.5:
                obs = obs.float() / 255.0
            else:
                obs = obs.float()
        
        # Now obs is (B, C, H, W) with C = in_channels
        # Normalize pixel values if they're in [0, 255]
        if obs.max() > 1.5:
            obs = obs / 255.0
        
        # Apply CNN
        h = self.conv(obs)  # (B, 64, H', W')
        
        # Adaptive pooling to fixed size
        h = self.adaptive_pool(h)  # (B, 64, 7, 7)
        
        # Flatten
        h = h.view(h.size(0), -1)  # (B, 64*7*7)
        
        # FC + LayerNorm
        z = self.ln(self.fc(h))  # (B, z_dim)
        
        return z


def make_pixel_encoder(
    z_dim: int = 256,
    grid_height: int = 17,
    grid_width: int = 19,
    use_embedding: bool = True,
    feature_vector_size: Optional[int] = None,
    **kwargs
) -> PixelEncoder:
    """
    Factory function to create a pixel encoder.
    
    Args:
        z_dim: Dimension of output feature vector
        grid_height: Height of the grid
        grid_width: Width of the grid
        use_embedding: Whether to use embedding layer for grid-based observations
        feature_vector_size: If provided, use MLP encoder for feature vectors instead of CNN
        **kwargs: Additional arguments passed to PixelEncoder
    
    Returns:
        PixelEncoder instance
    """
    return PixelEncoder(
        z_dim=z_dim,
        grid_height=grid_height,
        grid_width=grid_width,
        use_embedding=use_embedding,
        feature_vector_size=feature_vector_size,
        **kwargs
    )
