import pytest
import numpy as np
from yahtzee_env import YahtzeeEnv
from game_logic import (
    ALL_CATEGORIES, ONES, TWOS, THREES, FOURS, FIVES, SIXES,
    THREE_OF_A_KIND, FOUR_OF_A_KIND, FULL_HOUSE,
    SMALL_STRAIGHT, LARGE_STRAIGHT, YAHTZEE, CHANCE
)

class TestYahtzeeEnv:
    @pytest.fixture
    def env(self):
        """Create a basic environment with one medium opponent"""
        return YahtzeeEnv(num_opponents=1, opponent_difficulties=["medium"])
    
    @pytest.fixture
    def multi_opponent_env(self):
        """Create an environment with multiple opponents of different difficulties"""
        return YahtzeeEnv(num_opponents=3, opponent_difficulties=["easy", "medium", "hard"])
    
    def test_initialization(self, env, multi_opponent_env):
        """Test environment initialization"""
        # Test single opponent env
        assert env.num_opponents == 1
        assert env.opponent_difficulties == ["medium"]
        assert len(env.opponents) == 1
        
        # Test multi-opponent env
        assert multi_opponent_env.num_opponents == 3
        assert multi_opponent_env.opponent_difficulties == ["easy", "medium", "hard"]
        assert len(multi_opponent_env.opponents) == 3
        
        # Test action space
        assert isinstance(env.action_space.spaces[0].n, int)  # MultiBinary space
        assert env.action_space.spaces[1].n == len(ALL_CATEGORIES)  # Category space
        
        # Test observation space
        obs_space = env.observation_space.spaces
        assert obs_space["dice"].shape == (5,)
        assert obs_space["roll_number"].n == 3
        assert obs_space["scoresheet"].shape == (len(ALL_CATEGORIES),)
        assert obs_space["opponent_scores"].shape == (1,)  # Single opponent
        assert obs_space["relative_rank"].n == 2  # 0 or 1 for single opponent
        
        # Test multi-opponent observation space
        multi_obs_space = multi_opponent_env.observation_space.spaces
        assert multi_obs_space["opponent_scores"].shape == (3,)  # Three opponents
        assert multi_obs_space["relative_rank"].n == 4  # 0 to 3 for three opponents
    
    def test_reset(self, env):
        """Test environment reset"""
        obs, info = env.reset()
        
        # Test observation structure
        assert isinstance(obs, dict)
        assert all(key in obs for key in ["dice", "roll_number", "scoresheet", 
                                        "upper_bonus", "yahtzee_bonus", 
                                        "opponent_scores", "relative_rank"])
        
        # Test initial state
        assert obs["roll_number"] == 0
        assert obs["dice"].shape == (5,)
        assert all(1 <= x <= 6 for x in obs["dice"])
        assert obs["scoresheet"].shape == (len(ALL_CATEGORIES),)
        assert all(x == -1 for x in obs["scoresheet"])  # All categories unscored
        assert obs["upper_bonus"] == 0
        assert obs["yahtzee_bonus"][0] == 0
        assert obs["opponent_scores"].shape == (1,)
        assert obs["opponent_scores"][0] == 0  # Initial opponent score
        assert obs["relative_rank"] == 0  # Tied for first at start
    
    def test_relative_rank(self, env):
        """Test relative rank calculation"""
        env.reset()
        
        # Set up scores
        env.scoresheet.scores[ONES] = 3
        env.opponent_scoresheets[0].scores[ONES] = 2
        assert env._get_relative_rank(3) == 0  # Agent winning
        
        env.opponent_scoresheets[0].scores[TWOS] = 8
        assert env._get_relative_rank(3) == 1  # Agent losing
    
    def test_relative_reward(self, env):
        """Test reward calculation with relative performance"""
        env.reset()
        
        # Test reward when leading
        env.scoresheet.scores[ONES] = 3
        env.opponent_scoresheets[0].scores[ONES] = 2
        reward = env._calculate_relative_reward(5.0)
        assert reward == 6.0  # 5.0 * 1.2 (20% bonus for leading)
        
        # Test reward when behind
        env.opponent_scoresheets[0].scores[TWOS] = 8
        reward = env._calculate_relative_reward(5.0)
        assert reward < 5.0  # Should be penalized for being behind
    
    def test_step_reroll(self, env):
        """Test rerolling dice"""
        env.reset()
        initial_dice = env.dice.get_values().copy()
        
        # Reroll first three dice
        action = (np.array([1, 1, 1, 0, 0]), 0)  # Reroll first 3 dice, category doesn't matter
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Check that only specified dice were rerolled
        current_dice = obs["dice"]
        assert not np.array_equal(current_dice[:3], initial_dice[:3])  # First 3 should be different
        assert np.array_equal(current_dice[3:], initial_dice[3:])  # Last 2 should be same
        assert obs["roll_number"] == 1
        assert reward == 0  # No reward for rerolling
        assert not terminated
    
    def test_step_scoring(self, env):
        """Test scoring a category"""
        env.reset()
        
        # Force dice values for predictable test
        for i, value in enumerate([1, 1, 1, 2, 2]):
            env.dice.dice[i].value = value
        
        # Score three-of-a-kind
        action = (np.array([0, 0, 0, 0, 0]), ALL_CATEGORIES.index(THREE_OF_A_KIND))
        obs, reward, terminated, truncated, info = env.step(action)
        
        assert obs["roll_number"] == 0  # Reset for next turn
        assert reward > 0  # Should get positive reward for scoring
        assert not terminated  # Game shouldn't be over
        
        # Verify opponent took their turn
        assert obs["opponent_scores"][0] > 0
    
    def test_invalid_action(self, env):
        """Test attempting invalid actions"""
        env.reset()
        
        # Try to score in ONES
        action = (np.array([0, 0, 0, 0, 0]), ALL_CATEGORIES.index(ONES))
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Try to score in ONES again (invalid)
        action = (np.array([0, 0, 0, 0, 0]), ALL_CATEGORIES.index(ONES))
        obs, reward, terminated, truncated, info = env.step(action)
        
        assert reward == -10.0  # Penalty for invalid action
        assert not terminated
    
    def test_game_completion(self, env):
        """Test game completion and final rewards"""
        env.reset()
        
        # Fill all categories except one
        for i, category in enumerate(ALL_CATEGORIES[:-1]):
            env.dice._values = [1, 1, 1, 1, 1]  # Yahtzee for max score
            action = (np.array([0, 0, 0, 0, 0]), i)
            obs, reward, terminated, truncated, info = env.step(action)
            assert not terminated
        
        # Fill final category
        env.dice._values = [1, 1, 1, 1, 1]
        action = (np.array([0, 0, 0, 0, 0]), len(ALL_CATEGORIES) - 1)
        obs, reward, terminated, truncated, info = env.step(action)
        
        assert terminated  # Game should be over
        assert reward != 0  # Should get final reward
        # Final reward should include:
        # - Score for final category
        # - Upper section bonus if applicable
        # - Win/lose bonus
        
    def test_render_modes(self, env):
        """Test rendering modes"""
        env.reset()
        
        # Test human mode
        env.render_mode = "human"
        env.render()  # Should print to console
        
        # Test ansi mode
        env.render_mode = "ansi"
        output = env.render()
        assert isinstance(output, str)
        assert "Current Dice" in output
        assert "Roll" in output 