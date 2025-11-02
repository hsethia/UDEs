"""
Neural network architectures for UDEs.
Provides flexible network construction with various configurations.
"""

import torch
import torch.nn as nn
from typing import List, Optional, Callable


class FlexibleNN(nn.Module):
    """
    Flexible feedforward neural network with configurable architecture.
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int],
        activation: str = 'tanh',
        final_activation: Optional[str] = None,
        use_batch_norm: bool = False,
        dropout: float = 0.0
    ):
        """
        Args:
            input_dim: Input dimension
            output_dim: Output dimension
            hidden_dims: List of hidden layer dimensions (e.g., [64, 64, 32])
            activation: Activation function ('tanh', 'relu', 'elu', 'silu', 'gelu')
            final_activation: Optional final activation ('softplus', 'sigmoid', None)
            use_batch_norm: Whether to use batch normalization
            dropout: Dropout rate (0 = no dropout)
        """
        super(FlexibleNN, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        
        # Build network
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
                
            layers.append(self._get_activation(activation))
            
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
                
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        if final_activation is not None:
            layers.append(self._get_activation(final_activation))
        
        self.net = nn.Sequential(*layers)
        
    def _get_activation(self, name: str) -> nn.Module:
        """Get activation function by name"""
        activations = {
            'tanh': nn.Tanh(),
            'relu': nn.ReLU(),
            'elu': nn.ELU(),
            'silu': nn.SiLU(),
            'gelu': nn.GELU(),
            'sigmoid': nn.Sigmoid(),
            'softplus': nn.Softplus(),
            'leaky_relu': nn.LeakyReLU()
        }
        if name.lower() not in activations:
            raise ValueError(f"Unknown activation: {name}. Choose from {list(activations.keys())}")
        return activations[name.lower()]
    
    def forward(self, t, x):
        """
        Forward pass. Accepts time t for compatibility with torchdiffeq.
        
        Args:
            t: Time (can be ignored, but needed for torchdiffeq)
            x: Input tensor
        """
        # Ensure input has shape (batch_size, input_dim)
        if x.dim() == 1:
            x = x.unsqueeze(-1) if self.input_dim == 1 else x.unsqueeze(0)
        elif x.shape[-1] != self.input_dim and x.shape[0] == self.input_dim:
            x = x.unsqueeze(0)
            
        return self.net(x)
    
    def count_parameters(self) -> int:
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ResidualBlock(nn.Module):
    """Residual block for deeper networks"""
    def __init__(self, dim: int, activation: str = 'tanh'):
        super(ResidualBlock, self).__init__()
        self.linear1 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(dim, dim)
        self.activation = self._get_activation(activation)
        
    def _get_activation(self, name: str) -> nn.Module:
        activations = {
            'tanh': nn.Tanh(),
            'relu': nn.ReLU(),
            'elu': nn.ELU(),
            'silu': nn.SiLU(),
            'gelu': nn.GELU(),
        }
        return activations.get(name.lower(), nn.Tanh())
    
    def forward(self, x):
        residual = x
        out = self.activation(self.linear1(x))
        out = self.linear2(out)
        out = out + residual
        out = self.activation(out)
        return out


class ResidualNN(nn.Module):
    """
    Residual neural network for deeper architectures.
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        num_blocks: int = 3,
        activation: str = 'tanh',
        final_activation: Optional[str] = None
    ):
        """
        Args:
            input_dim: Input dimension
            output_dim: Output dimension
            hidden_dim: Hidden dimension (constant across blocks)
            num_blocks: Number of residual blocks
            activation: Activation function
            final_activation: Optional final activation
        """
        super(ResidualNN, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Input projection
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.activation = self._get_activation(activation)
        
        # Residual blocks
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, activation) for _ in range(num_blocks)
        ])
        
        # Output projection
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        
        self.final_activation = None
        if final_activation is not None:
            self.final_activation = self._get_activation(final_activation)
    
    def _get_activation(self, name: str) -> nn.Module:
        activations = {
            'tanh': nn.Tanh(),
            'relu': nn.ReLU(),
            'elu': nn.ELU(),
            'silu': nn.SiLU(),
            'gelu': nn.GELU(),
            'sigmoid': nn.Sigmoid(),
            'softplus': nn.Softplus(),
        }
        return activations.get(name.lower(), nn.Tanh())
    
    def forward(self, t, x):
        if x.dim() == 1:
            x = x.unsqueeze(-1) if self.input_dim == 1 else x.unsqueeze(0)
        elif x.shape[-1] != self.input_dim and x.shape[0] == self.input_dim:
            x = x.unsqueeze(0)
            
        x = self.activation(self.input_layer(x))
        
        for block in self.blocks:
            x = block(x)
        
        x = self.output_layer(x)
        
        if self.final_activation is not None:
            x = self.final_activation(x)
            
        return x
    
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_neural_network(
    input_dim: int,
    output_dim: int,
    architecture: str = 'flexible',
    **kwargs
) -> nn.Module:
    """
    Factory function to create neural networks.
    
    Args:
        input_dim: Input dimension
        output_dim: Output dimension
        architecture: 'flexible' or 'residual'
        **kwargs: Additional arguments for the network
        
    For 'flexible':
        - hidden_dims: List[int] (default: [32, 32])
        - activation: str (default: 'tanh')
        - final_activation: Optional[str] (default: None)
        - use_batch_norm: bool (default: False)
        - dropout: float (default: 0.0)
        
    For 'residual':
        - hidden_dim: int (default: 32)
        - num_blocks: int (default: 3)
        - activation: str (default: 'tanh')
        - final_activation: Optional[str] (default: None)
    """
    if architecture == 'flexible':
        hidden_dims = kwargs.get('hidden_dims', [32, 32])
        activation = kwargs.get('activation', 'tanh')
        final_activation = kwargs.get('final_activation', None)
        use_batch_norm = kwargs.get('use_batch_norm', False)
        dropout = kwargs.get('dropout', 0.0)
        
        return FlexibleNN(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            final_activation=final_activation,
            use_batch_norm=use_batch_norm,
            dropout=dropout
        )
    
    elif architecture == 'residual':
        hidden_dim = kwargs.get('hidden_dim', 32)
        num_blocks = kwargs.get('num_blocks', 3)
        activation = kwargs.get('activation', 'tanh')
        final_activation = kwargs.get('final_activation', None)
        
        return ResidualNN(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            num_blocks=num_blocks,
            activation=activation,
            final_activation=final_activation
        )
    
    else:
        raise ValueError(f"Unknown architecture: {architecture}")

