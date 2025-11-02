"""
UDE Framework - A flexible framework for Universal Differential Equations.
"""

from .data_generator import (
    ODESystem,
    DataGenerator,
    create_ode_system
)

from .neural_networks import (
    FlexibleNN,
    ResidualNN,
    create_neural_network
)

from .ude_models import (
    GenericUDE,
    create_ude,
    TrueFunction
)

from .training import UDETrainer

from .evaluation import UDEEvaluator

__version__ = '0.1.0'

__all__ = [
    # Data generation
    'ODESystem',
    'DataGenerator',
    'create_ode_system',
    
    # Neural networks
    'FlexibleNN',
    'ResidualNN',
    'create_neural_network',
    
    # UDE models
    'GenericUDE',
    'create_ude',
    'TrueFunction',
    
    # Training
    'UDETrainer',
    
    # Evaluation
    'UDEEvaluator',
]

