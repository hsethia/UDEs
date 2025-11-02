"""
Data generation module for UDE experiments.
Handles ODE system definition, simulation, and noise injection.
"""

import numpy as np
from scipy.integrate import solve_ivp
from typing import Callable, Dict, Tuple, Optional, List


class ODESystem:
    """
    Defines an ODE system with its parameters and equations.
    """
    def __init__(
        self,
        name: str,
        equations: Callable,
        true_params: Dict,
        state_names: List[str],
        param_descriptions: Optional[Dict] = None
    ):
        """
        Args:
            name: Name of the ODE system
            equations: Function (t, y, params) -> dydt that defines the ODE
            true_params: Dictionary of true parameter values
            state_names: List of names for state variables (e.g., ['X', 'Y'])
            param_descriptions: Optional descriptions of parameters
        """
        self.name = name
        self.equations = equations
        self.true_params = true_params
        self.state_names = state_names
        self.param_descriptions = param_descriptions or {}
        
    def __call__(self, t, y):
        """Make the system callable for scipy's solve_ivp"""
        return self.equations(t, y, self.true_params)


class DataGenerator:
    """
    Generates synthetic data from ODE systems with optional noise.
    """
    def __init__(self, ode_system: ODESystem):
        """
        Args:
            ode_system: ODESystem instance defining the equations
        """
        self.ode_system = ode_system
        
    def generate(
        self,
        initial_conditions: np.ndarray,
        t_span: Tuple[float, float],
        n_points: int = 1000,
        noise_level: float = 0.0,
        noise_type: str = 'relative',
        method: str = 'RK45',
        random_seed: Optional[int] = None
    ) -> Dict:
        """
        Generate data from the ODE system.
        
        Args:
            initial_conditions: Initial values for state variables
            t_span: Tuple of (t_start, t_end)
            n_points: Number of time points
            noise_level: Noise level (0.05 = 5% noise)
            noise_type: 'relative' (proportional to signal) or 'absolute'
            method: Integration method for solve_ivp
            random_seed: Random seed for reproducibility
            
        Returns:
            Dictionary containing:
                - t: time points
                - y_true: true (noiseless) solution
                - y_noisy: noisy solution
                - y0: initial conditions
                - params: true parameters used
        """
        if random_seed is not None:
            np.random.seed(random_seed)
            
        t_eval = np.linspace(t_span[0], t_span[1], n_points)
        
        # Solve the ODE
        sol = solve_ivp(
            self.ode_system,
            t_span,
            initial_conditions,
            t_eval=t_eval,
            method=method
        )
        
        y_true = sol.y.T  # Shape: (n_points, n_states)
        
        # Add noise
        if noise_level > 0:
            if noise_type == 'relative':
                noise = noise_level * np.abs(y_true) * np.random.randn(*y_true.shape)
            elif noise_type == 'absolute':
                noise = noise_level * np.random.randn(*y_true.shape)
            else:
                raise ValueError(f"Unknown noise_type: {noise_type}")
            y_noisy = y_true + noise
        else:
            y_noisy = y_true.copy()
            
        return {
            't': t_eval,
            'y_true': y_true,
            'y_noisy': y_noisy,
            'y0': initial_conditions,
            'params': self.ode_system.true_params,
            'state_names': self.ode_system.state_names
        }


# ============================================================================
# Helper function to create custom systems
# ============================================================================

def create_ode_system(name: str, equations: Callable, params: Dict,
                      state_names: List[str]) -> ODESystem:
    """
    Create a custom ODE system.
    
    Args:
        name: Name of the system
        equations: Function (t, y, params) -> dydt
        params: Dictionary of parameter values
        state_names: List of state variable names
        
    Example:
        def my_ode(t, y, params):
            x, y = y
            a, b = params['a'], params['b']
            dx_dt = a * x - b * x * y
            dy_dt = b * x * y - a * y
            return [dx_dt, dy_dt]
        
        system = create_ode_system(
            name="My System",
            equations=my_ode,
            params={'a': 1.0, 'b': 0.1},
            state_names=['x', 'y']
        )
    """
    return ODESystem(
        name=name,
        equations=equations,
        true_params=params,
        state_names=state_names
    )

