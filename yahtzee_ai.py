from game_logic import (
    ScoreSheet, Dice, 
    ONES, TWOS, THREES, FOURS, FIVES, SIXES,
    THREE_OF_A_KIND, FOUR_OF_A_KIND, FULL_HOUSE,
    SMALL_STRAIGHT, LARGE_STRAIGHT, YAHTZEE, CHANCE
)
from collections import Counter

class YahtzeeAI:
    """AI agent for playing Yahtzee using heuristic strategies."""
    
    def __init__(self, difficulty="medium"):
        """Initialize the AI agent with a difficulty level."""
        self.difficulty = difficulty  # "easy", "medium", or "hard"
        self.scoresheet = None
        self.dice = None
    
    def set_game_state(self, scoresheet, dice):
        """Set the current game state for the AI to make decisions."""
        self.scoresheet = scoresheet
        self.dice = dice
    
    def decide_reroll(self, current_dice_values, roll_number):
        """
        Decide which dice to reroll based on the current dice values and roll number.
        Returns a list of dice indices (0-based) to reroll.
        """
        if self.difficulty == "easy":
            return self._decide_reroll_easy(current_dice_values, roll_number)
        elif self.difficulty == "hard":
            return self._decide_reroll_hard(current_dice_values, roll_number)
        else:  # medium difficulty
            return self._decide_reroll_medium(current_dice_values, roll_number)
    
    def choose_category(self, dice_values):
        """
        Choose the best category to score based on current dice values.
        Returns the chosen category name.
        """
        if self.difficulty == "easy":
            return self._choose_category_easy(dice_values)
        elif self.difficulty == "hard":
            return self._choose_category_hard(dice_values)
        else:  # medium difficulty
            return self._choose_category_medium(dice_values)
    
    def _decide_reroll_easy(self, current_dice_values, roll_number):
        """Simple strategy: Keep the most frequent value, reroll others."""
        counter = Counter(current_dice_values)
        most_common_value = counter.most_common(1)[0][0]
        
        # If we have three or more of a kind, keep those and reroll others
        if counter[most_common_value] >= 3:
            return [i for i, val in enumerate(current_dice_values) if val != most_common_value]
        
        # If it's the last roll, don't reroll anything
        if roll_number == 3:
            return []
        
        # Otherwise, keep the two most common values if we have pairs
        if roll_number == 2 and counter[most_common_value] == 2:
            return [i for i, val in enumerate(current_dice_values) if val != most_common_value]
        
        # For first roll or no good combinations, reroll everything
        return list(range(len(current_dice_values)))
    
    def _decide_reroll_medium(self, current_dice_values, roll_number):
        """
        Medium strategy: Consider straight possibilities and high-value combinations.
        """
        counter = Counter(current_dice_values)
        sorted_vals = sorted(current_dice_values)
        
        # Check for potential straights
        unique_vals = len(set(current_dice_values))
        if unique_vals >= 4:
            # Potential straight, keep sequential numbers
            sequence = self._find_longest_sequence(sorted_vals)
            if len(sequence) >= 4:
                return [i for i, val in enumerate(current_dice_values) if val not in sequence]
        
        # Check for three or more of a kind
        most_common_val, most_common_count = counter.most_common(1)[0]
        if most_common_count >= 3:
            # Keep three of a kind, reroll others for potential full house or four of a kind
            return [i for i, val in enumerate(current_dice_values) if val != most_common_val]
        
        # Check for pairs in second roll
        if roll_number == 2:
            pairs = [val for val, count in counter.items() if count >= 2]
            if pairs:
                # Keep highest pair
                highest_pair = max(pairs)
                return [i for i, val in enumerate(current_dice_values) if val != highest_pair]
        
        # Default: reroll lowest values
        if roll_number < 3:
            return [i for i, val in enumerate(current_dice_values) if val < 4]
        
        return []
    
    def _decide_reroll_hard(self, current_dice_values, roll_number):
        """
        Advanced strategy: Consider all possible scoring combinations and probability.
        Implements optimal strategy based on expected value calculations.
        """
        # TODO: Implement advanced strategy considering:
        # 1. Probability of improving current hand
        # 2. Available categories and their potential scores
        # 3. Game state (e.g., upper section bonus possibility)
        # 4. Expected value of different combinations
        pass
    
    def _choose_category_easy(self, dice_values):
        """Simple strategy: Choose the highest scoring available category."""
        available_categories = self.scoresheet.get_available_categories()
        best_score = -1
        best_category = None
        
        for category in available_categories:
            score = self.scoresheet.get_potential_score(category, dice_values)
            if score > best_score:
                best_score = score
                best_category = category
        
        return best_category
    
    def _choose_category_medium(self, dice_values):
        """
        Medium strategy: Consider both current score and future possibilities.
        Prioritizes completing Yahtzee and straights when close.
        """
        available_categories = self.scoresheet.get_available_categories()
        scores = {cat: self.scoresheet.get_potential_score(cat, dice_values) 
                 for cat in available_categories}
        
        # First priority: Yahtzee with actual Yahtzee
        if YAHTZEE in scores and scores[YAHTZEE] == 50:
            return YAHTZEE
        
        # Second priority: Large straight if available
        if LARGE_STRAIGHT in scores and scores[LARGE_STRAIGHT] == 40:
            return LARGE_STRAIGHT
        
        # Third priority: Small straight if available
        if SMALL_STRAIGHT in scores and scores[SMALL_STRAIGHT] == 30:
            return SMALL_STRAIGHT
        
        # Fourth priority: Full house if available
        if FULL_HOUSE in scores and scores[FULL_HOUSE] == 25:
            return FULL_HOUSE
        
        # Check upper section bonus potential
        upper_section_score = sum(self.scoresheet.scores.get(cat, 0) 
                                for cat in [ONES, TWOS, THREES, FOURS, FIVES, SIXES]
                                if self.scoresheet.scores.get(cat) is not None)
        
        # If we're close to upper section bonus, prioritize upper section
        if upper_section_score < 63 and upper_section_score >= 43:
            # Look for good upper section scores
            for cat in [SIXES, FIVES, FOURS, THREES, TWOS, ONES]:
                if cat in scores and scores[cat] >= 8:
                    return cat
        
        # Default to highest scoring category
        return max(scores.items(), key=lambda x: x[1])[0]
    
    def _choose_category_hard(self, dice_values):
        """
        Advanced strategy: Uses expected value calculations and game state analysis.
        Considers optimal long-term strategy.
        """
        # TODO: Implement advanced category selection considering:
        # 1. Expected value of remaining rolls
        # 2. Probability of achieving better scores in other categories
        # 3. Game state optimization
        # 4. Upper section bonus strategy
        pass
    
    def _find_longest_sequence(self, sorted_vals):
        """Helper function to find the longest sequence in sorted dice values."""
        if not sorted_vals:
            return []
            
        sequences = []
        current_sequence = [sorted_vals[0]]
        
        for i in range(1, len(sorted_vals)):
            if sorted_vals[i] == sorted_vals[i-1] + 1:
                current_sequence.append(sorted_vals[i])
            elif sorted_vals[i] != sorted_vals[i-1]:  # Skip duplicates
                if current_sequence:
                    sequences.append(current_sequence)
                current_sequence = [sorted_vals[i]]
        
        sequences.append(current_sequence)
        return max(sequences, key=len) 