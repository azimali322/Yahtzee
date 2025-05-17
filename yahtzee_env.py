import gymnasium as gym
from gymnasium import spaces
import numpy as np
from game_logic import ScoreSheet, Dice, ALL_CATEGORIES
from typing import Optional, Dict, List, Tuple

class YahtzeeEnv(gym.Env):
    """Yahtzee environment following gym interface"""
    metadata = {"render_modes": ["human", "ansi"], "render_fps": 4}

    def __init__(self, render_mode: Optional[str] = None):
        super().__init__()
        
        # Define action spaces
        # Action space is a tuple of:
        # 1. Dice to reroll (binary mask of length 5)
        # 2. Category to score (one of 13 categories)
        self.action_space = spaces.Tuple((
            spaces.MultiBinary(5),  # Which dice to reroll
            spaces.Discrete(len(ALL_CATEGORIES))  # Which category to score
        ))

        # Observation space is a tuple of:
        # 1. Current dice values (5 dice, values 1-6)
        # 2. Current roll number (0-2)
        # 3. Scoresheet state (13 categories, -1 if not scored, actual score if scored)
        # 4. Upper section bonus achieved (binary)
        # 5. Yahtzee bonus count (int)
        self.observation_space = spaces.Dict({
            "dice": spaces.Box(low=1, high=6, shape=(5,), dtype=np.int32),
            "roll_number": spaces.Discrete(3),
            "scoresheet": spaces.Box(low=-1, high=50, shape=(len(ALL_CATEGORIES),), dtype=np.int32),
            "upper_bonus": spaces.Discrete(2),
            "yahtzee_bonus": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.int32)
        })

        self.render_mode = render_mode
        
        # Initialize game components
        self.dice = None
        self.scoresheet = None
        self.current_roll = None
        
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict, Dict]:
        """Reset the environment to initial state"""
        super().reset(seed=seed)
        
        # Initialize game state
        self.dice = Dice()
        self.scoresheet = ScoreSheet()
        self.current_roll = 0
        
        # Get initial observation
        observation = self._get_observation()
        
        # Additional info
        info = {}
        
        return observation, info
    
    def step(self, action: Tuple[np.ndarray, int]) -> Tuple[Dict, float, bool, bool, Dict]:
        """
        Take a step in the environment
        
        Args:
            action: Tuple of (dice_to_reroll, category_to_score)
                   dice_to_reroll: binary array of length 5
                   category_to_score: int (0-12)
        
        Returns:
            observation: Dict containing the new state
            reward: Float reward signal
            terminated: Whether the episode is done
            truncated: Whether the episode was truncated
            info: Additional information
        """
        dice_to_reroll, category_to_score = action
        
        # Convert category index to category name
        category = ALL_CATEGORIES[category_to_score]
        
        # If we still have rolls left and dice were selected to reroll
        if self.current_roll < 2 and np.any(dice_to_reroll):
            # Convert binary mask to indices
            reroll_indices = np.where(dice_to_reroll)[0]
            # Reroll selected dice
            self.dice.roll_specific(reroll_indices)
            self.current_roll += 1
            
            reward = 0  # No reward for rerolling
            terminated = False
            
        else:
            # Score the category if it's available
            if category in self.scoresheet.get_available_categories():
                # Get current dice values
                dice_values = self.dice.get_values()
                # Record the score
                score = self.scoresheet.get_potential_score(category, dice_values)
                self.scoresheet.record_score(category, dice_values)
                
                # Reset dice for next turn
                self.dice.reset_roll_count()
                self.current_roll = 0
                
                # Reward is the score achieved
                reward = float(score)
                
                # Check if game is done (all categories filled)
                terminated = self.scoresheet.is_complete()
                
                if terminated:
                    # Add bonus to final reward
                    reward += float(self.scoresheet.get_upper_section_bonus())
                    reward += float(self.scoresheet.yahtzee_bonus_score)
                
            else:
                # Trying to score in an already scored category
                reward = -10.0  # Penalty for invalid action
                terminated = False
        
        truncated = False
        observation = self._get_observation()
        info = {}
        
        if self.render_mode == "human":
            self.render()
        
        return observation, reward, terminated, truncated, info
    
    def _get_observation(self) -> Dict:
        """Get the current state observation"""
        # Get current dice values
        dice_values = np.array(self.dice.get_values())
        
        # Get scoresheet state
        scoresheet_state = np.array([
            self.scoresheet.scores[category] if self.scoresheet.scores[category] is not None else -1
            for category in ALL_CATEGORIES
        ])
        
        # Get upper bonus state
        upper_bonus = 1 if self.scoresheet.get_upper_section_bonus() > 0 else 0
        
        # Get Yahtzee bonus count
        yahtzee_bonus = np.array([self.scoresheet.yahtzee_bonus_score // 100])
        
        return {
            "dice": dice_values,
            "roll_number": self.current_roll,
            "scoresheet": scoresheet_state,
            "upper_bonus": upper_bonus,
            "yahtzee_bonus": yahtzee_bonus
        }
    
    def render(self):
        """Render the current state"""
        if self.render_mode == "human":
            print("\nCurrent Dice:", self.dice)
            print(f"Roll {self.current_roll + 1}/3")
            self.scoresheet.display_scoresheet()
        elif self.render_mode == "ansi":
            output = f"\nCurrent Dice: {self.dice}\n"
            output += f"Roll {self.current_roll + 1}/3\n"
            return output
    
    def close(self):
        """Clean up resources"""
        pass 