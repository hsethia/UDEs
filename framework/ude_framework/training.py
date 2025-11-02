"""
Training utilities for UDE models.
Handles optimization, learning rate scheduling, and training loops.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchdiffeq import odeint
from typing import Optional, Dict, List, Callable
import numpy as np


class UDETrainer:
    """
    Trainer class for UDE models.
    """
    def __init__(
        self,
        ude_model: nn.Module,
        optimizer_name: str = 'adam',
        learning_rate: float = 1e-3,
        scheduler_type: Optional[str] = 'plateau',
        scheduler_params: Optional[Dict] = None,
        weight_decay: float = 0.0,
        grad_clip: Optional[float] = None,
        ode_solver: str = 'dopri5',
        ode_rtol: float = 1e-6,
        ode_atol: float = 1e-8
    ):
        """
        Args:
            ude_model: The UDE model to train
            optimizer_name: 'adam', 'sgd', 'adamw', 'rmsprop'
            learning_rate: Initial learning rate
            scheduler_type: 'plateau', 'cosine', 'step', 'exponential', or None
            scheduler_params: Parameters for scheduler
            weight_decay: L2 regularization
            grad_clip: Gradient clipping max norm (None = no clipping)
            ode_solver: ODE solver method ('dopri5', 'dopri8', 'rk4', 'euler')
            ode_rtol: Relative tolerance for ODE solver
            ode_atol: Absolute tolerance for ODE solver
        """
        self.ude_model = ude_model
        self.device = next(ude_model.parameters()).device
        
        # Optimizer
        self.optimizer = self._create_optimizer(
            optimizer_name, learning_rate, weight_decay
        )
        
        # Scheduler
        self.scheduler = self._create_scheduler(scheduler_type, scheduler_params)
        
        # Training settings
        self.grad_clip = grad_clip
        self.ode_solver = ode_solver
        self.ode_rtol = ode_rtol
        self.ode_atol = ode_atol
        
        # History
        self.history = {
            'loss': [],
            'loss_per_state': [],
            'learning_rate': [],
            'grad_norm': []
        }
        
    def _create_optimizer(self, name: str, lr: float, weight_decay: float):
        """Create optimizer"""
        optimizers = {
            'adam': optim.Adam,
            'adamw': optim.AdamW,
            'sgd': optim.SGD,
            'rmsprop': optim.RMSprop
        }
        
        if name.lower() not in optimizers:
            raise ValueError(f"Unknown optimizer: {name}")
        
        optimizer_class = optimizers[name.lower()]
        
        if name.lower() == 'sgd':
            return optimizer_class(
                self.ude_model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                momentum=0.9
            )
        else:
            return optimizer_class(
                self.ude_model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
    
    def _create_scheduler(self, scheduler_type: Optional[str], 
                         params: Optional[Dict]):
        """Create learning rate scheduler"""
        if scheduler_type is None:
            return None
        
        params = params or {}
        
        if scheduler_type == 'plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=params.get('factor', 0.5),
                patience=params.get('patience', 100),
                verbose=params.get('verbose', False),
                min_lr=params.get('min_lr', 1e-6)
            )
        elif scheduler_type == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=params.get('T_max', 1000),
                eta_min=params.get('eta_min', 1e-6)
            )
        elif scheduler_type == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=params.get('step_size', 100),
                gamma=params.get('gamma', 0.5)
            )
        elif scheduler_type == 'exponential':
            return optim.lr_scheduler.ExponentialLR(
                self.optimizer,
                gamma=params.get('gamma', 0.95)
            )
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_type}")
    
    def forward_solve(self, y0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Solve ODE forward in time.
        
        Args:
            y0: Initial conditions (batch_size, n_states) or (n_states,)
            t: Time points (n_points,)
            
        Returns:
            Solution tensor (n_points, batch_size, n_states) or (n_points, n_states)
        """
        if y0.dim() == 1:
            y0 = y0.unsqueeze(0)
        
        sol = odeint(
            self.ude_model,
            y0,
            t,
            method=self.ode_solver,
            rtol=self.ode_rtol,
            atol=self.ode_atol
        )
        
        # Remove batch dimension if it was added
        if sol.shape[1] == 1:
            sol = sol.squeeze(1)
        
        return sol
    
    def compute_loss(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        loss_type: str = 'mse'
    ) -> Dict[str, torch.Tensor]:
        """
        Compute loss between predicted and true trajectories.
        
        Args:
            y_pred: Predicted trajectories (n_points, n_states)
            y_true: True trajectories (n_points, n_states)
            weights: Optional weights for each state variable
            loss_type: 'mse', 'mae', or 'huber'
            
        Returns:
            Dictionary with 'total' loss and per-state losses
        """
        if loss_type == 'mse':
            loss_per_state = torch.mean((y_pred - y_true)**2, dim=0)
        elif loss_type == 'mae':
            loss_per_state = torch.mean(torch.abs(y_pred - y_true), dim=0)
        elif loss_type == 'huber':
            loss_fn = nn.HuberLoss(reduction='none')
            loss_per_state = torch.mean(loss_fn(y_pred, y_true), dim=0)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
        
        # Apply weights if provided
        if weights is not None:
            loss_per_state = loss_per_state * weights
        
        total_loss = torch.sum(loss_per_state)
        
        return {
            'total': total_loss,
            'per_state': loss_per_state
        }
    
    def train_step(
        self,
        y0: torch.Tensor,
        t: torch.Tensor,
        y_true: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        loss_type: str = 'mse'
    ) -> Dict[str, float]:
        """
        Perform one training step.
        
        Args:
            y0: Initial conditions
            t: Time points
            y_true: True trajectories
            weights: Optional state weights
            loss_type: Loss function type
            
        Returns:
            Dictionary with loss values
        """
        self.optimizer.zero_grad()
        
        # Forward solve
        y_pred = self.forward_solve(y0, t)
        
        # Compute loss
        loss_dict = self.compute_loss(y_pred, y_true, weights, loss_type)
        
        # Backward pass
        loss_dict['total'].backward()
        
        # Gradient clipping
        grad_norm = 0.0
        if self.grad_clip is not None:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.ude_model.parameters(),
                max_norm=self.grad_clip
            ).item()
        else:
            grad_norm = sum(
                p.grad.norm().item()**2 
                for p in self.ude_model.parameters() 
                if p.grad is not None
            )**0.5
        
        # Optimizer step
        self.optimizer.step()
        
        return {
            'loss': loss_dict['total'].item(),
            'loss_per_state': loss_dict['per_state'].detach().cpu().numpy(),
            'grad_norm': grad_norm
        }
    
    def train(
        self,
        y0: torch.Tensor,
        t: torch.Tensor,
        y_true: torch.Tensor,
        n_epochs: int,
        weights: Optional[torch.Tensor] = None,
        loss_type: str = 'mse',
        print_every: int = 10,
        callback: Optional[Callable] = None
    ):
        """
        Train the UDE model.
        
        Args:
            y0: Initial conditions
            t: Time points
            y_true: True trajectories
            n_epochs: Number of training epochs
            weights: Optional state weights
            loss_type: Loss function type
            print_every: Print progress every N epochs
            callback: Optional callback function(epoch, metrics)
        """
        print(f"Starting training for {n_epochs} epochs...")
        print(f"Optimizer: {self.optimizer.__class__.__name__}")
        print(f"Learning rate: {self.optimizer.param_groups[0]['lr']}")
        print(f"ODE solver: {self.ode_solver}")
        print()
        
        for epoch in range(n_epochs):
            # Training step
            metrics = self.train_step(y0, t, y_true, weights, loss_type)
            
            # Update scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(metrics['loss'])
                else:
                    self.scheduler.step()
            
            # Record history
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['loss'].append(metrics['loss'])
            self.history['loss_per_state'].append(metrics['loss_per_state'])
            self.history['learning_rate'].append(current_lr)
            self.history['grad_norm'].append(metrics['grad_norm'])
            
            # Print progress
            if (epoch + 1) % print_every == 0 or epoch == 0:
                print(f"Epoch {epoch+1}/{n_epochs} | "
                      f"Loss: {metrics['loss']:.6f} | "
                      f"LR: {current_lr:.6f} | "
                      f"Grad: {metrics['grad_norm']:.4f}")
            
            # Callback
            if callback is not None:
                callback(epoch, metrics)
        
        print("\nTraining complete!")
        print(f"Final loss: {self.history['loss'][-1]:.6f}")
    
    def get_history(self) -> Dict:
        """Get training history"""
        return self.history
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint"""
        torch.save({
            'model_state_dict': self.ude_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'history': self.history
        }, path)
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path)
        self.ude_model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.history = checkpoint['history']

