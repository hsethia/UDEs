"""
Evaluation and visualization utilities for UDE models.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Callable, Tuple
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


class UDEEvaluator:
    """
    Evaluator for UDE models with comprehensive visualization and metrics.
    """
    def __init__(self, ude_model, state_names: List[str]):
        """
        Args:
            ude_model: Trained UDE model
            state_names: Names of state variables
        """
        self.ude_model = ude_model
        self.state_names = state_names
        self.n_states = len(state_names)
        
    def compute_metrics(
        self,
        y_pred: np.ndarray,
        y_true: np.ndarray
    ) -> Dict:
        """
        Compute evaluation metrics.
        
        Args:
            y_pred: Predicted trajectories (n_points, n_states)
            y_true: True trajectories (n_points, n_states)
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        for i, name in enumerate(self.state_names):
            y_p = y_pred[:, i]
            y_t = y_true[:, i]
            
            metrics[f'{name}_r2'] = r2_score(y_t, y_p)
            metrics[f'{name}_mae'] = mean_absolute_error(y_t, y_p)
            metrics[f'{name}_rmse'] = np.sqrt(mean_squared_error(y_t, y_p))
            metrics[f'{name}_mape'] = np.mean(np.abs((y_t - y_p) / (np.abs(y_t) + 1e-8))) * 100
        
        # Overall metrics
        metrics['overall_r2'] = r2_score(y_true.flatten(), y_pred.flatten())
        metrics['overall_mae'] = mean_absolute_error(y_true.flatten(), y_pred.flatten())
        metrics['overall_rmse'] = np.sqrt(mean_squared_error(y_true.flatten(), y_pred.flatten()))
        
        return metrics
    
    def plot_trajectories(
        self,
        t: np.ndarray,
        y_pred: np.ndarray,
        y_true: np.ndarray,
        y_noisy: Optional[np.ndarray] = None,
        figsize: Tuple[int, int] = (12, 10),
        save_path: Optional[str] = None
    ):
        """
        Plot predicted vs true trajectories.
        
        Args:
            t: Time points
            y_pred: Predicted trajectories
            y_true: True trajectories
            y_noisy: Optional noisy observations
            figsize: Figure size
            save_path: Optional path to save figure
        """
        fig, axes = plt.subplots(self.n_states, 1, figsize=figsize)
        
        if self.n_states == 1:
            axes = [axes]
        
        for i, (ax, name) in enumerate(zip(axes, self.state_names)):
            # Plot true trajectory
            ax.plot(t, y_true[:, i], 'b--', label='True', linewidth=2, alpha=0.7)
            
            # Plot noisy data if provided
            if y_noisy is not None:
                ax.plot(t, y_noisy[:, i], 'bo', label='Noisy data', 
                       markersize=3, alpha=0.3)
            
            # Plot prediction
            ax.plot(t, y_pred[:, i], 'r-', label='UDE Predicted', linewidth=2)
            
            # Metrics
            r2 = r2_score(y_true[:, i], y_pred[:, i])
            ax.set_ylabel(f'{name}', fontsize=12)
            ax.set_title(f'{name} (R² = {r2:.4f})', fontsize=11)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
        
        axes[-1].set_xlabel('Time', fontsize=12)
        plt.suptitle('Trajectories: True vs Predicted', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_phase_portrait(
        self,
        y_pred: np.ndarray,
        y_true: np.ndarray,
        y0: np.ndarray,
        y_noisy: Optional[np.ndarray] = None,
        state_indices: Tuple[int, int] = (0, 1),
        figsize: Tuple[int, int] = (10, 8),
        save_path: Optional[str] = None
    ):
        """
        Plot phase portrait (2D only).
        
        Args:
            y_pred: Predicted trajectories
            y_true: True trajectories
            y0: Initial conditions
            y_noisy: Optional noisy observations
            state_indices: Which two states to plot
            figsize: Figure size
            save_path: Optional path to save figure
        """
        if self.n_states < 2:
            print("Need at least 2 state variables for phase portrait")
            return
        
        idx1, idx2 = state_indices
        name1 = self.state_names[idx1]
        name2 = self.state_names[idx2]
        
        plt.figure(figsize=figsize)
        
        # True trajectory
        plt.plot(y_true[:, idx1], y_true[:, idx2], 'b--', 
                label='True trajectory', linewidth=3, alpha=0.7)
        
        # Predicted trajectory
        plt.plot(y_pred[:, idx1], y_pred[:, idx2], 'r-', 
                label='UDE predicted', linewidth=2)
        
        # Noisy data
        if y_noisy is not None:
            plt.plot(y_noisy[:, idx1], y_noisy[:, idx2], 'g.', 
                    label='Noisy data', alpha=0.2, markersize=4)
        
        # Initial condition
        plt.plot(y0[idx1], y0[idx2], 'ko', markersize=12, 
                label='Initial condition', zorder=5)
        
        plt.xlabel(f'{name1}', fontsize=12)
        plt.ylabel(f'{name2}', fontsize=12)
        plt.title(f'Phase Portrait: {name1} vs {name2}', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_training_history(
        self,
        history: Dict,
        figsize: Tuple[int, int] = (12, 8),
        save_path: Optional[str] = None
    ):
        """
        Plot training history.
        
        Args:
            history: Training history dictionary
            figsize: Figure size
            save_path: Optional path to save figure
        """
        fig, axes = plt.subplots(3, 1, figsize=figsize)
        
        epochs = range(1, len(history['loss']) + 1)
        
        # Loss
        axes[0].plot(epochs, history['loss'], linewidth=2, color='blue')
        axes[0].set_ylabel('Loss', fontsize=11)
        axes[0].set_title('Training Loss', fontsize=12)
        axes[0].grid(True, alpha=0.3)
        axes[0].set_yscale('log')
        
        # Learning rate
        axes[1].plot(epochs, history['learning_rate'], linewidth=2, color='orange')
        axes[1].set_ylabel('Learning Rate', fontsize=11)
        axes[1].set_title('Learning Rate Schedule', fontsize=12)
        axes[1].grid(True, alpha=0.3)
        axes[1].set_yscale('log')
        
        # Gradient norm
        axes[2].plot(epochs, history['grad_norm'], linewidth=2, color='green')
        axes[2].set_ylabel('Gradient Norm', fontsize=11)
        axes[2].set_xlabel('Epoch', fontsize=11)
        axes[2].set_title('Gradient Norm', fontsize=12)
        axes[2].grid(True, alpha=0.3)
        axes[2].set_yscale('log')
        
        plt.suptitle('Training History', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def compare_learned_function(
        self,
        nn_model,
        true_function: Callable,
        input_range: np.ndarray,
        input_name: str = 'Input',
        output_name: str = 'Output',
        figsize: Tuple[int, int] = (12, 6),
        save_path: Optional[str] = None
    ):
        """
        Compare learned neural network to true function.
        
        Args:
            nn_model: Trained neural network
            true_function: True function (numpy callable)
            input_range: Input values to evaluate
            input_name: Name for input axis
            output_name: Name for output axis
            figsize: Figure size
            save_path: Optional path to save figure
        """
        # Evaluate true function
        y_true = true_function(input_range)
        
        # Evaluate learned function
        with torch.no_grad():
            x_torch = torch.tensor(input_range, dtype=torch.float32)
            if x_torch.dim() == 1:
                x_torch = x_torch.unsqueeze(-1)
            y_learned = nn_model(None, x_torch).squeeze().numpy()
        
        # Compute R²
        r2 = r2_score(y_true, y_learned)
        
        plt.figure(figsize=figsize)
        plt.plot(input_range, y_true, 'b--', label='True Function', 
                linewidth=3, alpha=0.7)
        plt.plot(input_range, y_learned, 'r-', label='Learned (NN)', linewidth=2)
        plt.xlabel(input_name, fontsize=12)
        plt.ylabel(output_name, fontsize=12)
        plt.title(f'Function Comparison (R² = {r2:.4f})', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        # Print statistics
        mae = mean_absolute_error(y_true, y_learned)
        rmse = np.sqrt(mean_squared_error(y_true, y_learned))
        mape = np.mean(np.abs((y_true - y_learned) / (np.abs(y_true) + 1e-8))) * 100
        
        print(f"\nFunction approximation metrics:")
        print(f"  R² score: {r2:.4f}")
        print(f"  MAE: {mae:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAPE: {mape:.2f}%")
        
        return r2, mae, rmse
    
    def extrapolation_test(
        self,
        trainer,
        y0: torch.Tensor,
        t_train: np.ndarray,
        t_extended: np.ndarray,
        true_generator: Callable,
        figsize: Tuple[int, int] = (12, 10),
        save_path: Optional[str] = None
    ):
        """
        Test model extrapolation beyond training data.
        
        Args:
            trainer: UDETrainer instance with forward_solve method
            y0: Initial conditions
            t_train: Training time points
            t_extended: Extended time points
            true_generator: Function to generate true solution
            figsize: Figure size
            save_path: Optional path to save figure
        """
        # Generate true solution for extended time
        y_true_ext = true_generator(t_extended)
        
        # UDE prediction for extended time
        t_ext_torch = torch.tensor(t_extended, dtype=torch.float32)
        with torch.no_grad():
            y_pred_ext = trainer.forward_solve(y0, t_ext_torch).numpy()
        
        # Plot
        fig, axes = plt.subplots(self.n_states, 1, figsize=figsize)
        
        if self.n_states == 1:
            axes = [axes]
        
        t_max_train = t_train[-1]
        
        for i, (ax, name) in enumerate(zip(axes, self.state_names)):
            # Mark training region
            ax.axvspan(t_extended[0], t_max_train, alpha=0.1, 
                      color='green', label='Training region')
            
            # Plot trajectories
            ax.plot(t_extended, y_true_ext[:, i], 'b--', 
                   label='True', linewidth=2, alpha=0.7)
            ax.plot(t_extended, y_pred_ext[:, i], 'r-', 
                   label='UDE Predicted', linewidth=2)
            
            ax.set_ylabel(f'{name}', fontsize=12)
            ax.set_title(f'Extrapolation: {name}', fontsize=11)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
        
        axes[-1].set_xlabel('Time', fontsize=12)
        plt.suptitle('Model Extrapolation Test', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        # Compute extrapolation error
        extrap_mask = t_extended > t_max_train
        if np.any(extrap_mask):
            y_true_extrap = y_true_ext[extrap_mask]
            y_pred_extrap = y_pred_ext[extrap_mask]
            
            print(f"\nExtrapolation metrics (beyond t={t_max_train:.1f}):")
            for i, name in enumerate(self.state_names):
                mae = mean_absolute_error(y_true_extrap[:, i], y_pred_extrap[:, i])
                print(f"  {name} MAE: {mae:.4f}")
    
    def print_metrics(self, metrics: Dict):
        """Pretty print metrics"""
        print("\n" + "="*60)
        print("EVALUATION METRICS")
        print("="*60)
        
        # Per-state metrics
        for name in self.state_names:
            print(f"\n{name}:")
            print(f"  R² score:  {metrics[f'{name}_r2']:.4f}")
            print(f"  MAE:       {metrics[f'{name}_mae']:.4f}")
            print(f"  RMSE:      {metrics[f'{name}_rmse']:.4f}")
            print(f"  MAPE:      {metrics[f'{name}_mape']:.2f}%")
        
        # Overall metrics
        print(f"\nOverall:")
        print(f"  R² score:  {metrics['overall_r2']:.4f}")
        print(f"  MAE:       {metrics['overall_mae']:.4f}")
        print(f"  RMSE:      {metrics['overall_rmse']:.4f}")
        print("="*60 + "\n")

