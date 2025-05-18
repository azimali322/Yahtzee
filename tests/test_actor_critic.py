import pytest
import torch
import numpy as np
import sys
from pathlib import Path

# Add the root directory to Python path
root_dir = str(Path(__file__).parent.parent)
if root_dir not in sys.path:
    sys.path.append(root_dir)

from yahtzee_actor_critic import YahtzeeActorCritic
from game_logic import ALL_CATEGORIES

@pytest.fixture
def observation_space():
    """Create a dummy observation space for testing."""
    return {
        "dice": np.zeros(5),
        "roll_number": 0,
        "scoresheet": np.zeros(13),
        "upper_bonus": 0,
        "yahtzee_bonus": np.array([0]),
        "opponent_scores": np.array([0]),
        "relative_rank": 0
    }

@pytest.fixture
def actor_critic(observation_space):
    """Create an actor-critic network instance for testing."""
    return YahtzeeActorCritic(
        observation_space=observation_space,
        n_actions_dice=32,  # 2^5 possible dice reroll combinations
        n_actions_category=len(ALL_CATEGORIES)
    )

def test_network_initialization(actor_critic):
    """Test that the network is properly initialized."""
    # Check input size calculation
    expected_input_size = (
        5 +  # dice
        1 +  # roll number
        13 +  # scoresheet
        1 +  # upper bonus
        1 +  # yahtzee bonus
        1 +  # opponent scores (1 opponent)
        1    # relative rank
    )
    assert actor_critic.input_size == expected_input_size

    # Check network architecture
    assert isinstance(actor_critic.shared, torch.nn.Sequential)
    assert isinstance(actor_critic.actor_dice, torch.nn.Sequential)
    assert isinstance(actor_critic.actor_category, torch.nn.Sequential)
    assert isinstance(actor_critic.critic, torch.nn.Sequential)

def test_observation_processing(actor_critic, observation_space):
    """Test observation processing."""
    processed_obs = actor_critic._process_observation(observation_space)
    
    # Check that the output is a tensor with the correct shape
    assert isinstance(processed_obs, torch.Tensor)
    assert processed_obs.shape == (actor_critic.input_size,)
    assert processed_obs.dtype == torch.float32

def test_forward_pass(actor_critic, observation_space):
    """Test the forward pass of the network."""
    dice_logits, category_logits, value = actor_critic(observation_space)
    
    # Check output shapes
    assert dice_logits.shape == (32,)  # 32 possible dice combinations
    assert category_logits.shape == (len(ALL_CATEGORIES),)  # Number of categories
    assert value.shape == (1,)  # Single value estimate
    
    # Check that outputs are finite
    assert torch.all(torch.isfinite(dice_logits))
    assert torch.all(torch.isfinite(category_logits))
    assert torch.all(torch.isfinite(value))

def test_dice_action_selection(actor_critic, observation_space):
    """Test dice action selection."""
    # Test deterministic action
    binary_action, log_prob = actor_critic.get_dice_action(observation_space, deterministic=True)
    assert isinstance(binary_action, torch.Tensor)
    assert binary_action.shape == (5,)  # 5 dice
    assert binary_action.dtype == torch.int8
    assert all(x in [0, 1] for x in binary_action)
    assert isinstance(log_prob, torch.Tensor)
    
    # Test stochastic action
    binary_action, log_prob = actor_critic.get_dice_action(observation_space, deterministic=False)
    assert isinstance(binary_action, torch.Tensor)
    assert binary_action.shape == (5,)
    assert binary_action.dtype == torch.int8
    assert all(x in [0, 1] for x in binary_action)
    assert isinstance(log_prob, torch.Tensor)

def test_category_action_selection(actor_critic, observation_space):
    """Test category action selection."""
    # Test deterministic action
    category_idx, log_prob = actor_critic.get_category_action(observation_space, deterministic=True)
    assert isinstance(category_idx, int)
    assert 0 <= category_idx < len(ALL_CATEGORIES)
    assert isinstance(log_prob, torch.Tensor)
    
    # Test stochastic action
    category_idx, log_prob = actor_critic.get_category_action(observation_space, deterministic=False)
    assert isinstance(category_idx, int)
    assert 0 <= category_idx < len(ALL_CATEGORIES)
    assert isinstance(log_prob, torch.Tensor)

def test_action_evaluation(actor_critic, observation_space):
    """Test action evaluation for training."""
    # Create dummy actions
    dice_action = torch.tensor([1, 0, 1, 0, 1], dtype=torch.int8)
    category_action = torch.tensor(0)
    
    # Evaluate actions
    dice_log_probs, category_log_probs, entropy, values = actor_critic.evaluate_actions(
        observation_space,
        dice_action,
        category_action
    )
    
    # Check outputs
    assert isinstance(dice_log_probs, torch.Tensor)
    assert isinstance(category_log_probs, torch.Tensor)
    assert isinstance(entropy, torch.Tensor)
    assert isinstance(values, torch.Tensor)
    assert values.shape == (1,)
    
    # Check that outputs are finite
    assert torch.isfinite(dice_log_probs)
    assert torch.isfinite(category_log_probs)
    assert torch.isfinite(entropy)
    assert torch.all(torch.isfinite(values))

def test_multiple_observations(actor_critic):
    """Test handling of multiple observations with different values."""
    # Create a batch of observations with different values
    observations = [
        {
            "dice": np.array([1, 2, 3, 4, 5]),
            "roll_number": 1,
            "scoresheet": np.array([10, -1, 15, -1, 20, -1, 25, -1, 30, -1, 35, -1, 40]),
            "upper_bonus": 1,
            "yahtzee_bonus": np.array([1]),
            "opponent_scores": np.array([50]),
            "relative_rank": 1
        },
        {
            "dice": np.array([6, 6, 6, 6, 6]),
            "roll_number": 2,
            "scoresheet": np.array([-1] * 13),
            "upper_bonus": 0,
            "yahtzee_bonus": np.array([0]),
            "opponent_scores": np.array([0]),
            "relative_rank": 0
        }
    ]
    
    for obs in observations:
        # Test forward pass
        dice_logits, category_logits, value = actor_critic(obs)
        assert torch.all(torch.isfinite(dice_logits))
        assert torch.all(torch.isfinite(category_logits))
        assert torch.all(torch.isfinite(value))
        
        # Test action selection
        binary_action, log_prob = actor_critic.get_dice_action(obs)
        assert all(x in [0, 1] for x in binary_action)
        assert torch.isfinite(log_prob)
        
        category_idx, log_prob = actor_critic.get_category_action(obs)
        assert 0 <= category_idx < len(ALL_CATEGORIES)
        assert torch.isfinite(log_prob) 