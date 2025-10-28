"""
Example: Identifying Ornstein-Uhlenbeck Process

This example demonstrates how to use FEX-DM to identify the drift and diffusion
functions of an Ornstein-Uhlenbeck process from simulated data.
"""

import numpy as np
import matplotlib.pyplot as plt
from fexdm import StochasticDynamicsIdentifier
from fexdm.utils import generate_ornstein_uhlenbeck


def main():
    print("="*70)
    print("FEX-DM: Identifying Ornstein-Uhlenbeck Process")
    print("="*70)
    
    # Generate synthetic data from Ornstein-Uhlenbeck process
    # dX = theta*(mu - X)*dt + sigma*dW
    theta = 1.0
    mu = 0.0
    sigma = 0.5
    
    print("\nGenerating data from Ornstein-Uhlenbeck process:")
    print(f"  dX = {theta}*({mu} - X)*dt + {sigma}*dW")
    print(f"  theta={theta}, mu={mu}, sigma={sigma}")
    
    t, X = generate_ornstein_uhlenbeck(
        theta=theta,
        mu=mu,
        sigma=sigma,
        x0=2.0,
        t_span=(0, 20),
        dt=0.01,
        seed=42
    )
    
    print(f"\nGenerated {len(X)} data points over time span [0, 20]")
    
    # Identify the dynamics
    print("\n" + "-"*70)
    print("Identifying dynamics using FEX-DM...")
    print("-"*70)
    
    identifier = StochasticDynamicsIdentifier(
        max_complexity=3,
        dt=0.01
    )
    
    identifier.fit(X, t, verbose=True)
    
    # Print results
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"\nTrue drift:     {theta}*({mu} - x) = {-theta}*x")
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
    x_range = np.linspace(X.min(), X.max(), 100)
    true_drift = -theta * x_range
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
    
    # Simulate from identified model
    sde = identifier.get_sde()
    t_sim, X_sim = sde.simulate(
        x0=2.0,
        t_span=(0, 20),
        dt=0.01,
        n_trajectories=3,
        seed=123
    )
    
    axes[1, 1].plot(t, X, 'b-', alpha=0.5, linewidth=0.5, label='Original')
    for i in range(3):
        axes[1, 1].plot(t_sim, X_sim[:, i], alpha=0.5, linewidth=0.5)
    axes[1, 1].set_xlabel('Time')
    axes[1, 1].set_ylabel('X')
    axes[1, 1].set_title('Simulated from Identified Model')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('ornstein_uhlenbeck_identification.png', dpi=150)
    print("\nPlot saved as 'ornstein_uhlenbeck_identification.png'")
    
    return identifier


if __name__ == "__main__":
    main()
