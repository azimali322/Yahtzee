import gymnasium as gym
from gymnasium import spaces
import numpy as np
from game_logic import ScoreSheet, Dice, ALL_CATEGORIES
from typing import Optional, Dict, List, Tuple
from yahtzee_ai import YahtzeeAI

class YahtzeeEnv(gym.Env):
    """Yahtzee environment following gym interface"""
    metadata = {"render_modes": ["human", "ansi"], "render_fps": 4}

    def __init__(self, render_mode: Optional[str] = None, num_opponents: int = 1, opponent_difficulties: Optional[List[str]] = None):
        super().__init__()
        
        # Initialize opponents
        self.num_opponents = num_opponents
        self.opponent_difficulties = opponent_difficulties or ["medium"] * num_opponents
        self.opponents = [YahtzeeAI(diff) for diff in self.opponent_difficulties]
        self.opponent_scoresheets = None
        
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
        # 6. Opponent scores (num_opponents values)
        # 7. Relative rank (0 to num_opponents, where 0 is first place)
        self.observation_space = spaces.Dict({
            "dice": spaces.Box(low=1, high=6, shape=(5,), dtype=np.int32),
            "roll_number": spaces.Discrete(3),
            "scoresheet": spaces.Box(low=-1, high=50, shape=(len(ALL_CATEGORIES),), dtype=np.int32),
            "upper_bonus": spaces.Discrete(2),
            "yahtzee_bonus": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.int32),
            "opponent_scores": spaces.Box(low=0, high=np.inf, shape=(num_opponents,), dtype=np.int32),
            "relative_rank": spaces.Discrete(num_opponents + 1)
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
        
        # Initialize opponent scoresheets
        self.opponent_scoresheets = [ScoreSheet() for _ in range(self.num_opponents)]
        for opponent, scoresheet in zip(self.opponents, self.opponent_scoresheets):
            opponent.set_game_state(scoresheet, self.dice)
        
        # Get initial observation
        observation = self._get_observation()
        
        # Additional info
        info = {}
        
        return observation, info
    
    def _get_relative_rank(self, agent_score: float) -> int:
        """Calculate the agent's rank (0-based, where 0 is first place)"""
        opponent_scores = [sheet.get_grand_total() for sheet in self.opponent_scoresheets]
        all_scores = [agent_score] + opponent_scores
        sorted_scores = sorted(all_scores, reverse=True)
        return sorted_scores.index(agent_score)
    
    def _calculate_relative_reward(self, score_gain: float) -> float:
        """Calculate reward considering relative performance against opponents"""
        agent_score = self.scoresheet.get_grand_total()
        opponent_scores = [sheet.get_grand_total() for sheet in self.opponent_scoresheets]
        
        # Base reward is the score gained
        reward = score_gain
        
        # Add bonus for being in the lead or penalty for being behind
        max_opponent_score = max(opponent_scores)
        if agent_score > max_opponent_score:
            reward *= 1.2  # 20% bonus for being in the lead
        elif agent_score < max_opponent_score:
            score_diff = max_opponent_score - agent_score
            # Small penalty that scales with how far behind we are
            penalty_factor = 1.0 - min(0.2, score_diff / 100)  # Max 20% penalty
            reward *= penalty_factor
        
        return reward
    
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
        initial_score = self.scoresheet.get_grand_total()
        
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
            game_score = self.scoresheet.get_grand_total()
            
        else:
            # Score the category if it's available
            if category in self.scoresheet.get_available_categories():
                # Get current dice values
                dice_values = self.dice.get_values()
                # Get the score before recording
                score = self.scoresheet.get_potential_score(category, dice_values)
                # Record the score
                self.scoresheet.record_score(category, dice_values)
                
                # Let opponents take their turns
                for opponent, scoresheet in zip(self.opponents, self.opponent_scoresheets):
                    opponent.dice.reset_roll_count()
                    # Simulate opponent's turn
                    for _ in range(3):
                        reroll = opponent.decide_reroll(opponent.dice.get_values(), opponent.dice.roll_count + 1)
                        if not reroll:
                            break
                        opponent.dice.roll_specific(reroll)
                    # Score opponent's turn
                    opp_category = opponent.choose_category(opponent.dice.get_values())
                    scoresheet.record_score(opp_category, opponent.dice.get_values())
                
                # Reset dice for next turn
                self.dice.reset_roll_count()
                self.current_roll = 0
                
                # Calculate score gain and relative reward
                score_gain = score  # Use the actual score gained, not the difference in total
                reward = self._calculate_relative_reward(float(score_gain))
                
                # Get current game score
                game_score = self.scoresheet.get_grand_total()
                
                # Check if game is done (all categories filled)
                terminated = self.scoresheet.is_complete()
                
                if terminated:
                    # Add final bonuses to reward
                    final_bonus = float(self.scoresheet.get_upper_section_bonus())
                    final_bonus += float(self.scoresheet.yahtzee_bonus_score)
                    reward += self._calculate_relative_reward(final_bonus)
                    
                    # Add win/lose bonus
                    agent_score = self.scoresheet.get_grand_total()
                    opponent_scores = [sheet.get_grand_total() for sheet in self.opponent_scoresheets]
                    if agent_score > max(opponent_scores):
                        reward += 50.0  # Bonus for winning
                    elif agent_score < max(opponent_scores):
                        reward -= 25.0  # Penalty for losing
                
            else:
                # Trying to score in an already scored category
                reward = -10.0  # Penalty for invalid action
                terminated = False
                game_score = self.scoresheet.get_grand_total()
        
        truncated = False
        observation = self._get_observation()
        info = {
            "game_score": game_score,  # Actual Yahtzee score
            "reward_score": reward,    # RL reward score
            "agent_score": self.scoresheet.get_grand_total(),
            "opponent_scores": [sheet.get_grand_total() for sheet in self.opponent_scoresheets]
        }
        
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
        
        # Get opponent scores
        opponent_scores = np.array([
            sheet.get_grand_total() for sheet in self.opponent_scoresheets
        ])
        
        # Get relative rank
        relative_rank = self._get_relative_rank(self.scoresheet.get_grand_total())
        
        return {
            "dice": dice_values,
            "roll_number": self.current_roll,
            "scoresheet": scoresheet_state,
            "upper_bonus": upper_bonus,
            "yahtzee_bonus": yahtzee_bonus,
            "opponent_scores": opponent_scores,
            "relative_rank": relative_rank
        }
    
    def render(self):
        """Render the current state"""
        if self.render_mode == "human":
            print("\nCurrent Dice:", self.dice)
            print(f"Roll {self.current_roll + 1}/3")
            print("\nYour scoresheet:")
            self.scoresheet.display_scoresheet()
            print("\nOpponent scoresheets:")
            for i, sheet in enumerate(self.opponent_scoresheets):
                print(f"\nOpponent {i+1} ({self.opponent_difficulties[i]}):")
                sheet.display_scoresheet()
        elif self.render_mode == "ansi":
            output = f"\nCurrent Dice: {self.dice}\n"
            output += f"Roll {self.current_roll + 1}/3\n"
            return output
    
    def close(self):
        """Clean up resources"""
        pass 