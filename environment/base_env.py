from typing import Optional, Any
from abc import ABC, abstractmethod

import gymnasium as gym
import numpy as np

# NOTE: Not used for inheritance yet due to time constraints, but
#  this is the basic blueprint for what should be in an environment

class BaseDiscreteEnv(ABC, gym.Env):
    def __init__(self, *args, **kwargs):
        super().__init__()
        
    @abstractmethod
    def step(self, action: Any):
        pass
    
    @abstractmethod
    def reset(self):
        pass
    
    @abstractmethod
    def render(self):
        pass
    
    @abstractmethod
    def get_player_loc_from_state(self, state: Any) -> tuple[np.ndarray, Optional[str]]:
        """Gets the player location from the state and optionally returns
        extra info about the state (like has key or not) depending on the 
        environment.

        Args:
            state (Any): State of the environment

        Returns:
            tuple[np.ndarray, Optional[str]]: 
                - (2,) shape ndarray with (y,x) player coords (to index into a grid)
                - String with extra information about the state
        """
        pass
    
    @abstractmethod
    def get_wall_mask(self) -> np.ndarray:
        """Gets a mask with the walls of the level

        Returns:
            np.ndarray: Grid with 1s as mask and 0s as anything else
        """
        pass