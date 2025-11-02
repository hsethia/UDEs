"""
UDE model definitions.
Allows flexible specification of known and unknown terms in ODEs.
"""

import torch
import torch.nn as nn
from typing import Callable, List, Optional, Dict
import numpy as np


class GenericUDE(nn.Module):
    """
    Generic Universal Differential Equation model.
    Allows flexible combination of known physics and neural networks.
    """
    def __init__(
        self,
        n_states: int,
        ode_func: Callable,
        neural_networks: Dict[str, nn.Module],
        known_params: Optional[Dict] = None
    ):
        """
        Args:
            n_states: Number of state variables
            ode_func: Function that computes dy/dt given (t, y, nn_outputs, known_params)
                     Signature: ode_func(t, y, nn_outputs, known_params) -> dy/dt
            neural_networks: Dictionary of neural networks {name: nn.Module}
            known_params: Dictionary of known parameters
        """
        super(GenericUDE, self).__init__()
        
        self.n_states = n_states
        self.ode_func = ode_func
        self.known_params = known_params or {}
        
        # Register neural networks as submodules
        self.nn_dict = nn.ModuleDict(neural_networks)
        
    def forward(self, t, y):
        """
        Compute dy/dt for the UDE.
        
        Args:
            t: Time (scalar or tensor)
            y: State tensor of shape (..., n_states)
            
        Returns:
            dy/dt tensor of same shape as y
        """
        # Evaluate all neural networks
        nn_outputs = {}
        for name, net in self.nn_dict.items():
            nn_outputs[name] = net(t, y)
        
        # Compute dy/dt using the ODE function
        dydt = self.ode_func(t, y, nn_outputs, self.known_params)
        
        return dydt
    
    def count_parameters(self) -> Dict[str, int]:
        """Count parameters in each neural network"""
        counts = {}
        total = 0
        for name, net in self.nn_dict.items():
            count = sum(p.numel() for p in net.parameters() if p.requires_grad)
            counts[name] = count
            total += count
        counts['total'] = total
        return counts


# ============================================================================
# Helper function to create UDEs
# ============================================================================

def create_ude(
    n_states: int,
    ode_func: Callable,
    neural_networks: Dict[str, nn.Module],
    known_params: Optional[Dict] = None
) -> GenericUDE:
    """
    Create a UDE with user-defined ODE function.
    
    Args:
        n_states: Number of state variables
        ode_func: Function (t, y, nn_outputs, known_params) -> dy/dt
        neural_networks: Dictionary of neural networks {name: nn.Module}
        known_params: Dictionary of known parameters
        
    Example:
        def my_ude_ode(t, y, nn_outputs, known_params):
            x = y[..., 0:1]
            y_var = y[..., 1:2]
            
            # Get NN output
            nn_term = nn_outputs['my_nn']
            if nn_term.dim() > 2:
                nn_term = nn_term.squeeze(-1)
            
            # Known parameters
            d = known_params['degradation']
            
            # Define ODEs
            dx_dt = nn_term - d * x
            dy_dt = x - d * y_var
            
            return torch.cat([dx_dt, dy_dt], dim=-1)
        
        ude = create_ude(
            n_states=2,
            ode_func=my_ude_ode,
            neural_networks={'my_nn': nn_model},
            known_params={'degradation': 1.0}
        )
    """
    return GenericUDE(
        n_states=n_states,
        ode_func=ode_func,
        neural_networks=neural_networks,
        known_params=known_params
    )


class TrueFunction:
    """
    Wrapper for true functions to compare against learned NNs.
    """
    def __init__(self, func: Callable, name: str = "True Function"):
        """
        Args:
            func: Function that takes numpy array and returns numpy array
            name: Name for plotting
        """
        self.func = func
        self.name = name
        
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the true function"""
        return self.func(x)
    
    @staticmethod
    def hill_function(Y: np.ndarray, v: float, K: float, n: float) -> np.ndarray:
        """Hill function: v * Y^n / (K^n + Y^n)"""
        return v * (Y**n) / (K**n + Y**n)
    
    @staticmethod
    def interaction_term(X: np.ndarray, Y: np.ndarray, 
                        beta: float, delta: float) -> np.ndarray:
        """
        Interaction term for Lotka-Volterra
        Returns [prey_loss, predator_gain]
        """
        prey_loss = beta * X * Y
        predator_gain = delta * X * Y
        return np.stack([prey_loss, predator_gain], axis=-1)

