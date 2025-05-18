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
    
    def __init__(
        self,
        observation_space: Dict,
        n_actions_dice: int,  # 32 possible dice reroll combinations (2^5)
        n_actions_category: int,  # 13 scoring categories
        hidden_size: int = 64
    ):
        super().__init__()
        
        # Calculate input size from observation space
        # Dice (5) + Roll number (1) + Scoresheet (13) + Upper bonus (1) + 
        # Yahtzee bonus (1) + Opponent scores (varies) + Relative rank (1)
        self.input_size = (
            5 +  # dice values
            1 +  # roll number
            13 +  # scoresheet
            1 +  # upper bonus
            1 +  # yahtzee bonus
            observation_space["opponent_scores"].shape[0] +  # opponent scores
            1  # relative rank
        )
        
        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(self.input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        # Actor heads (one for dice reroll, one for category selection)
        self.actor_dice = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions_dice)  # Logits for dice reroll combinations
        )
        
        self.actor_category = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions_category)  # Logits for category selection
        )
        
        # Critic head
        self.critic = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)  # State value
        )
        
    def _process_observation(self, obs: Dict[str, np.ndarray]) -> torch.Tensor:
        """Convert observation dictionary to tensor."""
        # Concatenate all observation components
        obs_list = [
            obs['dice'],
            [obs['roll_number']],
            obs['scoresheet'],
            [obs['upper_bonus']],
            obs['yahtzee_bonus'],
            obs['opponent_scores'],
            [obs['relative_rank']]
        ]
        
        # Flatten and convert to tensor
        obs_array = np.concatenate([
            arr.flatten() if isinstance(arr, np.ndarray) else np.array(arr).flatten()
            for arr in obs_list
        ])
        
        return torch.FloatTensor(obs_array)
    
    def forward(self, obs: Dict[str, np.ndarray]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the network.
        
        Args:
            obs: Dictionary containing the observation
            
        Returns:
            dice_logits: Logits for dice reroll actions
            category_logits: Logits for category selection
            value: State value estimate
        """
        x = self._process_observation(obs)
        
        # Shared features
        features = self.shared(x)
        
        # Actor outputs
        dice_logits = self.actor_dice(features)
        category_logits = self.actor_category(features)
        
        # Critic output
        value = self.critic(features)
        
        return dice_logits, category_logits, value
    
    def get_dice_action(self, obs: Dict[str, np.ndarray], deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get dice reroll action and its log probability."""
        dice_logits, _, _ = self(obs)
        
        # Create distribution for log probability calculation
        dist = torch.distributions.Categorical(logits=dice_logits)
        
        if deterministic:
            action = torch.argmax(dice_logits)
        else:
            # Sample from categorical distribution
            action = dist.sample()
            
        # Convert action to binary array for dice reroll
        binary_action = torch.tensor(
            [(action.item() >> i) & 1 for i in range(5)],
            dtype=torch.int8
        )
        
        return binary_action, dist.log_prob(action)
    
    def get_category_action(self, obs: Dict[str, np.ndarray], deterministic: bool = False) -> Tuple[int, torch.Tensor]:
        """Get category selection action and its log probability."""
        _, category_logits, _ = self(obs)
        
        # Create distribution for log probability calculation
        dist = torch.distributions.Categorical(logits=category_logits)
        
        if deterministic:
            action = torch.argmax(category_logits)
        else:
            # Sample from categorical distribution
            action = dist.sample()
        
        return action.item(), dist.log_prob(action)
    
    def evaluate_actions(
        self,
        obs: Dict[str, np.ndarray],
        dice_action: torch.Tensor,
        category_action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for training.
        
        Returns:
            dice_log_probs: Log probabilities of dice actions
            category_log_probs: Log probabilities of category actions
            entropy: Combined entropy of both action distributions
            values: State values
        """
        dice_logits, category_logits, values = self(obs)
        
        # Create distributions
        dice_dist = torch.distributions.Categorical(logits=dice_logits)
        category_dist = torch.distributions.Categorical(logits=category_logits)
        
        # Convert binary dice action to integer
        dice_int = sum([int(x) * (2 ** i) for i, x in enumerate(dice_action)])
        
        # Get log probabilities
        dice_log_probs = dice_dist.log_prob(torch.tensor(dice_int))
        category_log_probs = category_dist.log_prob(category_action)
        
        # Calculate entropy (for exploration)
        entropy = dice_dist.entropy().mean() + category_dist.entropy().mean()
        
        return dice_log_probs, category_log_probs, entropy, values 