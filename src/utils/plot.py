"""
Plotting utilities for FEX-DM
"""
import os
import numpy as np
import matplotlib.pyplot as plt


def plot_training_data_histogram(current_state_train, 
                                 save_path, 
                                 model_name='OU1d',
                                 train_size=None,
                                 noise_level=None,
                                 bins=50,
                                 figsize=(8, 6),
                                 dpi=300):
    """
    Plot and save histogram of training data distribution.
    
    Args:
        current_state_train: numpy array of current state values for training
        save_path: directory path where the figure should be saved
        model_name: name of the model (default: 'OU1d')
        train_size: number of training samples (for title)
        noise_level: noise level used (for title and filename)
        bins: number of bins for histogram (default: 50)
        figsize: figure size tuple (default: (8, 6))
        dpi: resolution for saved figure (default: 300)
    
    Returns:
        str: path to the saved figure file
    """
    # Create save directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    
    # Generate filename
    if noise_level is not None:
        filename = f'{model_name}_training_data_histogram_noise_{noise_level}.pdf'
    else:
        filename = f'{model_name}_training_data_histogram.pdf'
    
    histogram_path = os.path.join(save_path, filename)
    
    # Create figure
    plt.figure(figsize=figsize)
    plt.hist(current_state_train.flatten(), bins=bins, edgecolor='black', alpha=0.7)
    plt.xlabel('Current State Value')
    plt.ylabel('Frequency')
    
    # Create title
    title_parts = [f'{model_name} Training Data Distribution']
    if train_size is not None:
        title_parts.append(f'Samples: {train_size}')
    if noise_level is not None:
        title_parts.append(f'Noise Level: {noise_level}')
    plt.title(', '.join(title_parts))
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(histogram_path, dpi=dpi, bbox_inches='tight')
    print(f"[INFO] Histogram saved to: {histogram_path}")
    
    
    # return histogram_path

