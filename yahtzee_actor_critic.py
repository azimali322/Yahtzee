import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict
from collections import namedtuple

# Define a named tuple for storing transitions
Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'done'])

class YahtzeeActorCritic(nn.Module):
    """Actor-Critic network for Yahtzee environment."""
    
    def __init__(self, observation_space, n_actions_dice=5, n_actions_category=13, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()
        
        self.device = device
        input_size = self._get_obs_size(observation_space)
        
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
        
        # Separate dice policy layers (output size must be 5 for 5 dice)
        self.dice_layers = nn.Sequential(
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 5)  # Fixed to 5 for dice actions
        ).to(device)
        
        # Separate category policy layers (output size must be 13 for categories)
        self.category_layers = nn.Sequential(
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 13)  # Fixed to 13 for categories
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
        
        # Actor heads
        dice_logits = self.dice_layers(shared_features)
        category_logits = self.category_layers(shared_features)
        
        # Critic head
        value = self.value(shared_features)
        
        return dice_logits, category_logits, value
    
    def get_dice_action(self, obs, deterministic=False):
        """Get dice reroll action."""
        with torch.no_grad():
            dice_logits, _, _ = self(obs)
            if deterministic:
                action = torch.sigmoid(dice_logits) > 0.5
            else:
                # Clamp probabilities to [0, 1] range and handle NaN values
                probs = torch.clamp(torch.sigmoid(dice_logits), 0.0, 1.0)
                probs = torch.nan_to_num(probs, nan=0.5, posinf=1.0, neginf=0.0)
                action = torch.bernoulli(probs)
        return action.bool(), dice_logits
    
    def get_category_action(self, obs, deterministic=False):
        """Get category action."""
        with torch.no_grad():
            _, category_logits, _ = self(obs)
            if deterministic:
                action = torch.argmax(category_logits)
            else:
                # Apply softmax and handle NaN values
                probs = F.softmax(category_logits, dim=0)
                probs = torch.nan_to_num(probs, nan=1.0/len(probs), posinf=1.0, neginf=0.0)
                action = torch.multinomial(probs, 1).item()
        return action, category_logits
    
    def evaluate_actions(self, obs, dice_action, category_action):
        """Evaluate actions and compute log probabilities and entropy."""
        dice_logits, category_logits, value = self(obs)
        
        # Dice action log probabilities
        dice_probs = torch.sigmoid(dice_logits)
        dice_probs = torch.clamp(dice_probs, 1e-6, 1.0 - 1e-6)  # Prevent log(0)
        dice_log_probs = torch.sum(
            dice_action * torch.log(dice_probs) + 
            (1 - dice_action) * torch.log(1 - dice_probs)
        )
        
        # Category action log probabilities
        category_probs = F.softmax(category_logits, dim=0)
        category_probs = torch.clamp(category_probs, 1e-6, 1.0)  # Prevent log(0)
        category_log_probs = torch.log(category_probs[category_action])
        
        # Entropy for exploration
        dice_entropy = -(dice_probs * torch.log(dice_probs) + 
                        (1 - dice_probs) * torch.log(1 - dice_probs)).sum()
        category_entropy = -(category_probs * torch.log(category_probs)).sum()
        entropy = dice_entropy + category_entropy
        
        return dice_log_probs, category_log_probs, entropy, value 