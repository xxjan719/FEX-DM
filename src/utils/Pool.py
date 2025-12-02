from dataclasses import dataclass
from typing import List, Tuple, Optional
import torch
import torch.nn as nn
import os
import sys

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Add the src directory to the path to handle imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try to get POOL_LIMIT from config
def get_pool_limit():
    """Get POOL_LIMIT from config, with fallback to default."""
    try:
        from config import parse_args
        args = parse_args()
        return args.POOL_LIMIT
    except (ImportError, AttributeError, SystemExit, ValueError):
        # Fallback to default if config is not available or args not parsed yet
        return 100

DEFAULT_POOL_LIMIT = get_pool_limit()


@dataclass
class Candidate:
    """
    Represents a candidate model in the pool with its score, model, loss, and action sequence.
    """
    score: float
    model: nn.Module
    loss: float
    action: List  # Operator sequence (action)
    expression: Optional[str] = None  # Store expression explicitly
    
    @property
    def error(self):
        """Alias for loss (for backward compatibility)."""
        return self.loss

    def get_expression(self):
        """Get the expression string, computing it if not already stored."""
        if self.expression is None:
            return self.model.expression_visualize_simplified()
        return self.expression
    

class Pool:
    """
    Pool of candidate models, maintaining the top N candidates by score.
    
    The pool keeps candidates sorted by score (descending - highest scores first).
    When the pool is full, only candidates with scores above the threshold are added.
    """
    def __init__(self, pool_limit: Optional[int] = None):
        """
        Initialize the pool.
        
        Args:
            pool_limit: Maximum number of candidates to keep. If None, uses DEFAULT_POOL_LIMIT.
        """
        self.POOL_LIMIT = pool_limit if pool_limit is not None else DEFAULT_POOL_LIMIT
        self.candidates: List[Candidate] = []
        self.min_score_threshold = float('-inf')  # Minimum score to be in the pool

    def add(self, score: float, model: nn.Module, loss: float, action: List):
        """
        Add a candidate to the pool if its score is high enough.
        
        Args:
            score: Score of the candidate (higher is better)
            model: The FEX model
            loss: Loss value
            action: Operator sequence (list of operator indices)
        """
        # Only add if score is above the minimum threshold (or pool is not full)
        if len(self.candidates) >= self.POOL_LIMIT and score <= self.min_score_threshold:
            return
        
        # Store the expression at the time of adding to preserve the sequence-expression correspondence
        expression = model.expression_visualize_simplified()
        self.candidates.append(Candidate(score, model, loss, action, expression))
        
        # Sort in descending order (highest scores first)
        self.sort()
        
        # Remove worst candidates if pool is too large
        if len(self.candidates) > self.POOL_LIMIT:
            # Remove the last (worst) candidates
            self.candidates = self.candidates[:self.POOL_LIMIT]
        
        # Update threshold to the worst score in the pool
        if len(self.candidates) == self.POOL_LIMIT:
            self.min_score_threshold = self.candidates[-1].score

    def sort(self):
        """Sort candidates by score in descending order (best first)."""
        self.candidates.sort(key=lambda c: c.score, reverse=True)

    def get_best(self) -> Optional[Candidate]:
        """Get the best candidate (highest score)."""
        return self.candidates[0] if self.candidates else None
    
    def get_worst(self) -> Optional[Candidate]:
        """Get the worst candidate (lowest score) in the pool."""
        return self.candidates[-1] if self.candidates else None

    def __len__(self):
        """Return the number of candidates in the pool."""
        return len(self.candidates)

    def __iter__(self):
        """Iterate over candidates (best first)."""
        return iter(self.candidates)
    
    def __getitem__(self, index: int) -> Candidate:
        """Get candidate by index (0 = best, -1 = worst)."""
        return self.candidates[index]


        