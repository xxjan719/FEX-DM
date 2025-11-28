import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
import torch.nn as nn
from typing import Tuple
from torch import Tensor

SAMPLER_EPSILON = 0.2  # Exploration probability (20% chance of uniform exploration)

class Sampler(nn.Module):
    """
    Epsilon-greedy sampler for probability mass functions (PMFs).
    
    This sampler implements an epsilon-greedy strategy:
    - With probability epsilon (20%): samples uniformly from all classes (exploration)
    - With probability (1-epsilon) (80%): samples from the given PMF (exploitation)
    
    This is commonly used in reinforcement learning and model selection to balance
    between exploring new options and exploiting known good options.
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, pmfs: Tuple[Tensor, ...], output: Tensor):
        """
        Sample from multiple PMFs using epsilon-greedy strategy.
        
        Args:
            pmfs: Tuple of probability mass function tensors, each representing 
                  probabilities over different classes/options
            output: Pre-allocated tensor to store sampled indices (one per PMF)
        
        Returns:
            output: Tensor containing sampled class indices for each PMF
        
        Example:
            pmf = [0.7, 0.1, 0.1, 0.1] means:
            - 70% chance of class 0
            - 10% chance each of classes 1, 2, 3
            
            With epsilon=0.2:
            - 20% of the time: sample uniformly (25% each class)
            - 80% of the time: sample from pmf (70% class 0, etc.)
        """
        for i, pmf in enumerate(pmfs):
            # Generate random number to decide: explore (uniform) or exploit (pmf)
            u = torch.rand(1, device=pmf.device)
            
            if u < SAMPLER_EPSILON:
                # EXPLORATION: Use uniform distribution (equal probability for all classes)
                classes: int = pmf.shape[0]  # Number of classes/options
                pmf_unif = torch.full((classes,), fill_value=1/classes, device=pmf.device)
                # Sample uniformly from all classes
                output[i] = torch.multinomial(pmf_unif, 1, replacement=True)
            else:
                # EXPLOITATION: Use the given PMF (follow the learned probabilities)
                output[i] = torch.multinomial(pmf, 1, replacement=True)
                
        return output

if __name__ == "__main__":
    """
    Example usage of the Sampler:
    
    Each PMF (Probability Mass Function) represents probabilities over 4 classes.
    - pmf[0] = [0.7, 0.1, 0.1, 0.1]: 70% chance of class 0, 10% each for others
    - pmf[1] = [0.1, 0.7, 0.1, 0.1]: 70% chance of class 1, 10% each for others  
    - pmf[2] = [0.1, 0.1, 0.7, 0.1]: 70% chance of class 2, 10% each for others
    
    With epsilon=0.2:
    - 20% of the time: samples uniformly (25% chance for each of 4 classes)
    - 80% of the time: samples from the given PMF (follows the 70/10/10/10 distribution)
    
    This balances exploration (trying all options) with exploitation (using learned preferences).
    """
    pmfs = (
        torch.tensor([0.7, 0.1, 0.1, 0.1]),  # PMF 1: prefers class 0
        torch.tensor([0.1, 0.7, 0.1, 0.1]),  # PMF 2: prefers class 1
        torch.tensor([0.1, 0.1, 0.7, 0.1]),  # PMF 3: prefers class 2
    )
    output = torch.zeros(3, dtype=torch.long)
    sampler = Sampler()
    sampled = sampler(pmfs, output)
    print(f"Sampled indices: {sampled}")
    print(f"  - PMF 1 sampled class: {sampled[0].item()}")
    print(f"  - PMF 2 sampled class: {sampled[1].item()}")
    print(f"  - PMF 3 sampled class: {sampled[2].item()}")