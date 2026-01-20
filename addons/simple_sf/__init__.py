"""
Simple Successor Features addon.

Provides Simple-SF module and loss functions for continual learning.
"""

from addons.simple_sf.simple_sf import SimpleSF, simple_sf_losses

__all__ = ['SimpleSF', 'simple_sf_losses']
