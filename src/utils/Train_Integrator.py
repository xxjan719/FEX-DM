# a decorator takes a function, extends it and returns.
# a function can return a function
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from dataclasses import dataclass
from typing import Callable, Tuple, Union
import torch
from torch import Tensor

try:
    from .FEX import FEX
except:
    from FEX import FEX

@dataclass
class Body4TrainIntegrationParams:
    dt: float

@dataclass
class Body4TrainIntegrationArgs:
    integration_func: Callable
    model_name: str = "equipart"  # Add parameter name to determine which model to use


class Body4TrainIntegrator:
    def __init__(self, integratorParams: Body4TrainIntegrationParams, method: str = "integration-based"):
        """
        Initialize the integrator
        Args:
            integratorParams: Integration parameters
            method: Integration method to use. Options:
                   - "derivative-based": Uses derivative-based method (ui_next - ui)/dt
                   - "integration-based": Uses Euler integration
        """
        self._integratorparams = integratorParams
        self.method = method.lower()
        if self.method not in ["derivative-based", "integration-based"]:
            raise ValueError("Method must be either 'derivative-based' or 'integration-based'")
    
    def integrate(self, 
                  current_state_train: Tensor, 
                  next_state_train: Tensor, 
                  integration_func: Callable,
                  dimension: int = 1) -> Tuple[Tensor, Tensor]:
        """
        Integrate using current_state_train and next_state_train.
        
        Args:
            current_state_train: Current state tensor, shape (N, dim) where N is number of samples
            next_state_train: Next state tensor, shape (N, dim)
            integration_func: The FEX model function to use for integration
            dimension: Dimension of the system (default: 1 for OU1d)
        
        Returns:
            expression_pred: Predicted next state or derivative (depending on method)
            label: Target label (next state or derivative)
        """
        dt = self._integratorparams.dt
        
        # Ensure tensors are on the same device
        device = current_state_train.device
        current_state = current_state_train.to(device)
        next_state = next_state_train.to(device)
        
        if self.method == "derivative-based":
            # Derivative-based method: predict du/dt
            # Label: (u_{i+1} - u_i) / dt
            label = (next_state - current_state) / dt
            # Prediction: f(u_i) where f is the FEX model
            expression_pred = integration_func(current_state)
            # Ensure output shape matches label shape (N, dim)
            # Handle various output shapes from FEX model
            if expression_pred.dim() == 0:
                # Scalar - expand to match
                expression_pred = expression_pred.expand_as(label)
            elif expression_pred.dim() == 1:
                # (N,) -> (N, dim)
                if dimension == 1:
                    expression_pred = expression_pred.unsqueeze(-1)
                else:
                    expression_pred = expression_pred.unsqueeze(-1).expand(-1, dimension)
            elif expression_pred.dim() == 2:
                # (N, M) -> (N, dim)
                if expression_pred.shape[1] == dimension:
                    expression_pred = expression_pred
                elif expression_pred.shape[1] == 1:
                    expression_pred = expression_pred.expand(-1, dimension) if dimension > 1 else expression_pred
                else:
                    # Take diagonal or first column
                    expression_pred = expression_pred[:, 0:dimension] if expression_pred.shape[1] >= dimension else expression_pred.mean(dim=1, keepdim=True).expand(-1, dimension)
            
        elif self.method == "integration-based":
            # Integration-based method (Euler): predict u_{i+1}
            # Euler method: u_{i+1} = u_i + dt * f(u_i)
            # where f(u_i) is the derivative function (FEX model)
            derivative_pred = integration_func(current_state)
            
            # Ensure derivative_pred has correct shape to match current_state (N, dim)
            # Handle various output shapes from FEX model
            if derivative_pred.dim() == 0:
                # Scalar - expand to match
                derivative_pred = derivative_pred.expand_as(current_state)
            elif derivative_pred.dim() == 1:
                # (N,) -> (N, dim)
                if dimension == 1:
                    derivative_pred = derivative_pred.unsqueeze(-1)
                else:
                    derivative_pred = derivative_pred.unsqueeze(-1).expand(-1, dimension)
            elif derivative_pred.dim() == 2:
                # (N, M) -> (N, dim)
                if derivative_pred.shape[1] == dimension:
                    derivative_pred = derivative_pred
                elif derivative_pred.shape[1] == 1:
                    derivative_pred = derivative_pred.expand(-1, dimension) if dimension > 1 else derivative_pred
                else:
                    # Take appropriate slice
                    if dimension == 1:
                        derivative_pred = derivative_pred[:, 0:1]
                    else:
                        derivative_pred = derivative_pred[:, 0:dimension] if derivative_pred.shape[1] >= dimension else derivative_pred.mean(dim=1, keepdim=True).expand(-1, dimension)
            
            expression_pred = current_state + dt * derivative_pred
            # Label: u_{i+1}
            label = next_state
        else:
            raise ValueError("Method must be either 'derivative-based' or 'integration-based'")
        
        return expression_pred, label


if __name__ == "__main__":
    """
    Example usage of Body4TrainIntegrator with FEX models.
    """
    # Test with regular FEX for 1D case (OU1d)
    print("="*60)
    print("Example 1: 1D FEX (OU1d) - Integration-based method")
    print("="*60)
    op_seq_1d = torch.tensor([2, 0, 3, 2])  # 4 operators for 1D
    model_1d = FEX(op_seq_1d, dim=1)
    
    integratorParams = Body4TrainIntegrationParams(dt=1e-2)
    
    # Create test data: (N, dim) where N is number of samples
    current_state_train = torch.randn(100, 1)  # 100 samples, 1 dimension
    next_state_train = torch.randn(100, 1)
    
    integrator = Body4TrainIntegrator(integratorParams, method="integration-based")
    expression_pred, label = integrator.integrate(
        current_state_train=current_state_train,
        next_state_train=next_state_train,
        integration_func=model_1d,
        dimension=1
    )
    
    print(f"Current state shape: {current_state_train.shape}")
    print(f"Next state shape: {next_state_train.shape}")
    print(f"Expression pred shape: {expression_pred.shape}")
    print(f"Label shape: {label.shape}")
    print(f"Model expression: {model_1d.expression_visualize()}")
    print(f"Expression simplified: {model_1d.expression_visualize_simplified()}")
    print()
    
    # Test with 3D FEX for equipart case
    print("="*60)
    print("Example 2: 3D FEX (equipart) - Integration-based method")
    print("="*60)
    op_seqs_3d = torch.tensor([2, 0, 1, 2,   # Dimension 0
                               2, 0, 1, 2,   # Dimension 1
                               2, 0, 1, 2])  # Dimension 2
    model_3d = FEX(op_seqs_3d, dim=3)
    
    # Create test data for 3D
    current_state_train_3d = torch.randn(100, 3)  # 100 samples, 3 dimensions
    next_state_train_3d = torch.randn(100, 3)
    
    integrator_3d = Body4TrainIntegrator(integratorParams, method="integration-based")
    expression_pred_3d, label_3d = integrator_3d.integrate(
        current_state_train=current_state_train_3d,
        next_state_train=next_state_train_3d,
        integration_func=model_3d,
        dimension=3
    )
    
    print(f"Current state shape: {current_state_train_3d.shape}")
    print(f"Next state shape: {next_state_train_3d.shape}")
    print(f"Expression pred shape: {expression_pred_3d.shape}")
    print(f"Label shape: {label_3d.shape}")
    print(f"Model expression: {model_3d.expression_visualize()}")
    print(f"Expression simplified: {model_3d.expression_visualize_simplified()}")
    
    # Test with derivative-based method
    print("="*60)
    print("Example 3: 1D FEX - Derivative-based method")
    print("="*60)
    integrator_deriv = Body4TrainIntegrator(integratorParams, method="derivative-based")
    expression_pred_deriv, label_deriv = integrator_deriv.integrate(
        current_state_train=current_state_train,
        next_state_train=next_state_train,
        integration_func=model_1d,
        dimension=1
    )
    
    print(f"Expression pred shape: {expression_pred_deriv.shape}")
    print(f"Label shape: {label_deriv.shape}")
    print("Label represents: (u_{i+1} - u_i) / dt")
    print("="*60)