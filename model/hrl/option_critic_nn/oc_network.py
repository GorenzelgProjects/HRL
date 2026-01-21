# Heavily inspired by https://github.com/lweitkamp/option-critic-pytorch/blob/master/main.py

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, 
                 image_size: list[int, int],
                 n_filters: list[int], 
                 conv_sizes: list[int], 
                 strides: list[int],
                 n_neurons: int,
                 n_options: int,
                 n_actions: int,
                 temperature: float,
                 device: str) -> None:
        super(Encoder, self).__init__()
        
        self.n_options = n_options
        self.n_actions = n_actions
        self.temperature = temperature
        
        # Pre-computed number of in_features of the final dense layer
        self.lin_features = self._precompute_linear_input_size(torch.tensor(image_size), conv_sizes, strides) * n_filters[-1]
        
        # Encoding state features
        self.conv1 = nn.Conv2d(in_channels=1, # Assumes gray-scaled images
                               out_channels=n_filters[0], 
                               kernel_size=(conv_sizes[0], conv_sizes[0]), 
                               stride=strides[0])
        self.conv2 = nn.Conv2d(in_channels=n_filters[0],
                               out_channels=n_filters[1],
                               kernel_size=(conv_sizes[1], conv_sizes[1]),
                               stride=strides[1])
        self.conv3 = nn.Conv2d(in_channels=n_filters[1],
                               out_channels=n_filters[2],
                               kernel_size=(conv_sizes[2], conv_sizes[2]),
                               stride=strides[2])
        self.linear = nn.Linear(in_features=self.lin_features,
                                out_features=n_neurons)
        self.relu = nn.ReLU()
        
        # Option transformation
        self.pi_options_nn = nn.Linear(in_features=n_neurons,
                                       out_features=n_options)
        self.beta_nn = nn.Linear(in_features=n_neurons,
                                 out_features=n_options)
        self.pi_actions_nn = nn.Linear(in_features=n_neurons,
                                       out_features=n_options * n_actions)
        
        self.temperature = temperature
        self.device = device
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        x = self.relu(x)
        
        return x

    def encode_state(self, state: np.ndarray) -> torch.Tensor:       
        # Convert env state to a 4D tensor
        state_tensor = torch.from_numpy(state).to(self.device)
        if state_tensor.ndim < 3 or state_tensor.ndim > 4:
            raise ValueError(f"The state should be of shape (batch_size, 4 (rgba), img_size, img_size). ndim = {state_tensor.ndim}")
        elif state_tensor.ndim == 3:
            state_tensor = state_tensor.unsqueeze(0)
        
        # Encode the state
        features = self.forward(state_tensor)
        
        return features
    
    def pi_options(self, state: torch.Tensor | np.ndarray) -> torch.Tensor:
        if isinstance(state, np.ndarray):
            state = self.encode_state(state) # Shape: (self.n_neurons, )
        
        Q_Omega = self.pi_options_nn(state) # Shape: (n_options, )
        
        return Q_Omega
    
    def beta(self, state: torch.Tensor | np.ndarray) -> torch.Tensor:
        if isinstance(state, np.ndarray):
            state = self.encode_state(state) # Shape: (self.n_neurons, )
        
        logits = self.beta_nn(state) # Shape: (n_options, )
        
        return F.sigmoid(logits)
    
    def intra_options(self, state: torch.Tensor | np.ndarray) -> torch.Tensor:
        
        if isinstance(state, np.ndarray):
            state = self.encode_state(state) # Shape: (self.n_neurons, )
        
        logits = self.pi_actions_nn(state) # Shape: (n_options * n_actions, )
        logits = logits.view((self.n_options, self.n_actions)) # Shape: (n_options, n_actions)
        
        return F.softmax(logits / self.temperature, dim=-1)
    
    def _precompute_linear_input_size(self,
                                      img_sizes: torch.Tensor,
                                      conv_sizes: list[int],
                                      strides: list[int]) -> int:
        # NOTE: Assumes that the conv_size and stride 
        # as applied the same in both height and width dimension
        out_size = img_sizes
        for conv_size, stride in zip(conv_sizes, strides):
            out_size = torch.ceil((out_size - conv_size + 1) / stride)
        
        return int(out_size[0].item() * out_size[1].item())
    