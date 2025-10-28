"""
Example: Identifying Double-Well Potential System

This example demonstrates identification of a nonlinear system with a
double-well potential.
"""

import numpy as np
import matplotlib.pyplot as plt
from fexdm import StochasticDynamicsIdentifier
from fexdm.utils import generate_double_well


def main():
    print("="*70)
    print("FEX-DM: Identifying Double-Well Potential System")
    print("="*70)
    
    # Generate synthetic data from double-well system
    # dX = (alpha*X - beta*X^3)*dt + sigma*dW
    alpha = 1.0
    beta = 0.5
    sigma = 0.3
    
    print("\nGenerating data from double-well system:")
    print(f"  dX = ({alpha}*X - {beta}*X^3)*dt + {sigma}*dW")
    
    t, X = generate_double_well(
        alpha=alpha,
        beta=beta,
        sigma=sigma,
        x0=0.5,
        t_span=(0, 30),
        dt=0.01,
        seed=42
    )
    
    print(f"\nGenerated {len(X)} data points over time span [0, 30]")
    
    # Identify the dynamics
    print("\n" + "-"*70)
    print("Identifying dynamics using FEX-DM...")
    print("-"*70)
    
    identifier = StochasticDynamicsIdentifier(
        max_complexity=4,  # Higher complexity for cubic term
        dt=0.01
    )
    
    identifier.fit(X, t, verbose=True)
    
    # Print results
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"\nTrue drift:     {alpha}*x - {beta}*x^3")
    print(f"Identified:     {identifier.get_drift_expression()}")
    print(f"\nTrue diffusion: {sigma}")
    print(f"Identified:     {identifier.get_diffusion_expression()}")
    
    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Plot time series
    axes[0, 0].plot(t, X, 'b-', alpha=0.7, linewidth=0.5)
    axes[0, 0].set_xlabel('Time')
    axes[0, 0].set_ylabel('X')
    axes[0, 0].set_title('Generated Time Series')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot drift function
    x_range = np.linspace(-2, 2, 100)
    true_drift = alpha * x_range - beta * x_range**3
    pred_drift = identifier.predict_drift(x_range)
    
    axes[0, 1].plot(x_range, true_drift, 'k--', label='True', linewidth=2)
    axes[0, 1].plot(x_range, pred_drift, 'r-', label='Identified', linewidth=2)
    axes[0, 1].set_xlabel('X')
    axes[0, 1].set_ylabel('Drift f(X)')
    axes[0, 1].set_title('Drift Function')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot diffusion function
    true_diffusion = np.ones_like(x_range) * sigma
    pred_diffusion = identifier.predict_diffusion(x_range)
    
    axes[1, 0].plot(x_range, true_diffusion, 'k--', label='True', linewidth=2)
    axes[1, 0].plot(x_range, pred_diffusion, 'r-', label='Identified', linewidth=2)
    axes[1, 0].set_xlabel('X')
    axes[1, 0].set_ylabel('Diffusion g(X)')
    axes[1, 0].set_title('Diffusion Function')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Phase portrait
    axes[1, 1].plot(X[:-1], np.diff(X)/0.01, 'b.', alpha=0.1, markersize=1)
    axes[1, 1].plot(x_range, true_drift, 'k--', linewidth=2, label='True drift')
    axes[1, 1].set_xlabel('X')
    axes[1, 1].set_ylabel('dX/dt')
    axes[1, 1].set_title('Phase Portrait')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('double_well_identification.png', dpi=150)
    print("\nPlot saved as 'double_well_identification.png'")
    
    return identifier


if __name__ == "__main__":
    main()
