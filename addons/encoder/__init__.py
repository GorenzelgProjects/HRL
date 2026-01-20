"""
Shared pixel encoder module.

Provides a CNN-style encoder that converts pixel observations to feature vectors.
"""

from addons.encoder.pixel_encoder import PixelEncoder, make_pixel_encoder

__all__ = ['PixelEncoder', 'make_pixel_encoder']
