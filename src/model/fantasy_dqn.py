from typing import List

import torch
import torch.nn as nn

from utils import NUM_POS

class FantasyDQN(nn.Module):
    """
    A simple deep Q network for the fantasy football draft. Observations are
    vectors of shape [B,13], represented as [num_qb_drafted, num_rb_drafted,
    num_wr_drafted, num_te_drafted, num_dst_drafted, num_k_drafted,
    best_qb_available_points, best_rb_available_points, best_wr_available_points,
    best_te_available_points, best_dst_available_points, best_k_available_points,
    round]. Note the counts for drafted players are each normalized by dividing
    by the max allowed.
    """
    
    def __init__(self, hidden_dims: List[int] = [64, 32]):
        """
        Initializes a FantasyDQN with the given width for each hidden layer.
        
        That is, depth is len(hidden_dims) and hidden_dims[i] contains the
        depth of the i-th hidden layer, 0-indexed. 
        """
        
        layers = []
        
        layers.append(nn.Linear(13, hidden_dims[0]))
        layers.append(nn.ReLU())
        
        for i in range(1, len(hidden_dims)):
            layers.append(nn.Linear(hidden_dims[i - 1], hidden_dims[i]))
            layers.append(nn.ReLU())
            
        layers.append(nn.Linear(hidden_dims[-1], NUM_POS))
        layers.append(nn.ReLU()) # No negative projected points
        
        self.dqn = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Given the observation array vector of shape [B,13], produces a [B,6] dimensional output
        vector containing the Q values per position, where out[b, i] is the Q
        value for selecting POS[i] for batch b. Q values for invalid actions
        (exceeding limit, i.e. normalized amount is 1) are set to negative infinity.
        """
        out = self.dqn(x)
        
        # Mask fully drafted positions to -inf
        out[x[:,:NUM_POS] == 1] = float('inf')
        
        return out