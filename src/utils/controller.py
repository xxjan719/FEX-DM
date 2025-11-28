import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import sys
import os

# Add the src directory to the path to handle imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from config import parse_args
    args = parse_args()
    CONTROLLER_INPUT_SIZE = args.CONTROLLER_INPUT_SIZE
    CONTROLLER_HIDDEN_SIZE = args.CONTROLLER_HIDDEN_SIZE
except ImportError:
    # Fallback values if config is not available
    CONTROLLER_INPUT_SIZE = 20
    CONTROLLER_HIDDEN_SIZE = 30

class Controller(nn.Module):
    def __init__(self, pmf_sizes: Tuple[int, ...]):
        super().__init__()

        self.splits = [0]
        for size in pmf_sizes:
            self.splits.append(self.splits[-1]+size)
        # print(self.splits)
        self.net = nn.Sequential(
            nn.Linear(CONTROLLER_INPUT_SIZE, CONTROLLER_HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(CONTROLLER_HIDDEN_SIZE, self.splits[-1])
        )
    
    def forward(self, x):
        logits = self.net(x)

        pmfs = []
        for i in range(len(self.splits)-1):
            pmf = F.softmax(logits[self.splits[i]:self.splits[i+1]], dim=0)
            pmfs.append(pmf)
        return pmfs

