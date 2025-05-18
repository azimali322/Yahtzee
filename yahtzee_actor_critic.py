import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict
from collections import namedtuple

# Define a named tuple for storing transitions
Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'done'])

class YahtzeeActorCritic(nn.Module):
    """Actor-Critic network for Yahtzee environment with unified action space."""
    
    def __init__(self, observation_space, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()
        
        self.device = device
        input_size = self._get_obs_size(observation_space)
        
        # Calculate total action space size
        # 32 possible dice reroll combinations (2^5) + 13 categories = 45 total actions
        self.n_actions = 32 + 13  # Combined action space
        
        # Deeper shared layers with layer normalization
        self.shared = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU()
        ).to(device)
        
        # Single action head for combined action space
        self.action_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, self.n_actions)  # Output logits for all possible actions
        ).to(device)
        
        # Critic head
        self.value = nn.Linear(128, 1).to(device)
    
    def _get_obs_size(self, observation_space):
        """Calculate total observation size."""
        total_size = (
            len(observation_space['dice']) +  # 5 dice values
            1 +  # roll number
            len(observation_space['scoresheet']) +  # 13 category scores
            1 +  # upper bonus
            1 +  # yahtzee bonus
            len(observation_space['opponent_scores']) +  # opponent scores
            1   # relative rank
        )
        return total_size
    
    def _process_observation(self, obs):
        """Convert observation dict to tensor."""
        try:
            # If obs is already a tensor, return it
            if isinstance(obs, torch.Tensor):
                return obs.to(self.device)
                
            # Convert dict observation to tensor
            obs_array = np.concatenate([
                obs['dice'].astype(np.float32).flatten(),
                np.array([obs['roll_number']], dtype=np.float32).flatten(),
                obs['scoresheet'].astype(np.float32).flatten(),
                np.array([obs['upper_bonus']], dtype=np.float32).flatten(),
                obs['yahtzee_bonus'].astype(np.float32).flatten(),
                obs['opponent_scores'].astype(np.float32).flatten(),
                np.array([obs['relative_rank']], dtype=np.float32).flatten()
            ])
            
            # Replace any infinite or NaN values with 0
            obs_array = np.nan_to_num(obs_array, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Convert to tensor and move to device
            return torch.tensor(obs_array, dtype=torch.float32, device=self.device)
        except Exception as e:
            print(f"Error processing observation: {e}")
            print(f"Observation: {obs}")
            raise
    
    def forward(self, obs):
        """Forward pass through the network."""
        # Process observation
        x = self._process_observation(obs)
        
        # Add batch dimension if needed
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        
        # Shared layers
        shared_features = self.shared(x)
        
        # Action logits and value
        action_logits = self.action_head(shared_features)
        value = self.value(shared_features)
        
        return action_logits, value
    
    def get_action(self, obs, deterministic=False):
        """Get action from the network."""
        with torch.no_grad():
            action_logits, _ = self(obs)
            
            if deterministic:
                action = torch.argmax(action_logits)
            else:
                # Apply softmax and handle NaN values
                probs = F.softmax(action_logits, dim=-1)
                probs = torch.nan_to_num(probs, nan=1.0/self.n_actions, posinf=1.0, neginf=0.0)
                action = torch.multinomial(probs, 1).item()
            
            # Convert action index to dice reroll and category
            if action < 32:  # Dice reroll action
                # Convert to binary representation for dice reroll
                dice_action = torch.tensor(
                    [(action >> i) & 1 for i in range(5)],
                    dtype=torch.bool,
                    device=self.device
                )
                category = None
            else:  # Category action
                dice_action = None
                category = action - 32
            
            return action, dice_action, category, action_logits
    
    def evaluate_actions(self, obs, actions):
        """Evaluate actions and compute log probabilities and entropy."""
        action_logits, value = self(obs)
        
        # Calculate log probabilities using softmax
        log_probs = F.log_softmax(action_logits, dim=-1)
        action_log_probs = log_probs.gather(1, actions.unsqueeze(-1))
        
        # Calculate entropy
        probs = F.softmax(action_logits, dim=-1)
        entropy = -(probs * log_probs).sum(-1).mean()
        
        return action_log_probs, entropy, value 