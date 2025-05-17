import numpy as np
from yahtzee_ai import YahtzeeAI
from game_logic import ALL_CATEGORIES

class YahtzeeAIGymWrapper:
    """Wrapper to make YahtzeeAI compatible with Gym environment"""
    
    def __init__(self, difficulty="medium"):
        """Initialize the wrapper with a YahtzeeAI agent"""
        self.agent = YahtzeeAI(difficulty)
        
    def reset(self):
        """Reset the agent's state"""
        self.agent.scoresheet = None
        self.agent.dice = None
    
    def act(self, observation):
        """
        Convert Gym observation to action
        
        Args:
            observation: Dict containing:
                - dice: array of 5 dice values
                - roll_number: current roll number (0-2)
                - scoresheet: array of 13 category scores (-1 if not scored)
                - upper_bonus: whether upper bonus is achieved
                - yahtzee_bonus: number of yahtzee bonuses
                
        Returns:
            Tuple of (dice_to_reroll, category_to_score)
        """
        # Create temporary scoresheet from observation
        from game_logic import ScoreSheet
        scoresheet = ScoreSheet()
        for i, score in enumerate(observation['scoresheet']):
            if score >= 0:  # Category has been scored
                scoresheet.scores[ALL_CATEGORIES[i]] = int(score)
        
        # Create temporary dice from observation
        from game_logic import Dice
        dice = Dice()
        dice.roll_count = observation['roll_number']
        # Set dice values directly
        for i, value in enumerate(observation['dice']):
            dice.dice[i].value = int(value)
        
        # Update agent's game state
        self.agent.set_game_state(scoresheet, dice)
        
        # Get reroll decision if not last roll
        dice_to_reroll = np.zeros(5, dtype=np.int8)
        if observation['roll_number'] < 2:
            reroll_indices = self.agent.decide_reroll(
                observation['dice'],
                observation['roll_number'] + 1
            )
            if reroll_indices:
                dice_to_reroll[reroll_indices] = 1
        
        # Get category choice
        category = self.agent.choose_category(observation['dice'])
        category_index = ALL_CATEGORIES.index(category)
        
        return dice_to_reroll, category_index 