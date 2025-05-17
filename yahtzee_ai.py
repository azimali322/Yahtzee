from game_logic import (
    ScoreSheet, Dice, 
    ONES, TWOS, THREES, FOURS, FIVES, SIXES,
    THREE_OF_A_KIND, FOUR_OF_A_KIND, FULL_HOUSE,
    SMALL_STRAIGHT, LARGE_STRAIGHT, YAHTZEE, CHANCE
)
from collections import Counter
from itertools import combinations_with_replacement
import math

class YahtzeeAI:
    """AI agent for playing Yahtzee using heuristic strategies."""
    
    # Add class constants for probabilities
    UPPER_SECTION = [ONES, TWOS, THREES, FOURS, FIVES, SIXES]
    LOWER_SECTION = [THREE_OF_A_KIND, FOUR_OF_A_KIND, FULL_HOUSE, 
                    SMALL_STRAIGHT, LARGE_STRAIGHT, YAHTZEE, CHANCE]
    
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
    
    def _calculate_probability_of_value(self, target_count, current_count, dice_to_roll):
        """Calculate probability of getting exactly target_count of a value after rerolling."""
        if target_count < current_count:
            return 0.0  # Can't decrease count through rerolls
        
        needed = target_count - current_count
        if needed > dice_to_roll:
            return 0.0  # Need more dice than available to roll
        
        # Probability of rolling the specific value on one die
        p_success = 1/6
        
        # Use binomial probability formula
        ways = math.comb(dice_to_roll, needed)
        prob = ways * (p_success ** needed) * ((1 - p_success) ** (dice_to_roll - needed))
        return prob

    def _calculate_straight_probability(self, current_values, dice_to_roll, straight_length):
        """Calculate probability of completing a straight of given length."""
        current_unique = sorted(set(current_values))
        max_sequence = len(self._find_longest_sequence(current_unique))
        needed_unique = straight_length - max_sequence
        
        if needed_unique > dice_to_roll:
            return 0.0  # Need more dice than available
        
        # Calculate probability based on needed unique values and available dice
        if straight_length == 4:  # Small straight
            if max_sequence == 3:
                return 0.5  # Need one specific value out of two possibilities
            elif max_sequence == 2:
                return 0.3  # Need two specific values with some flexibility
        else:  # Large straight
            if max_sequence == 4:
                return 1/3  # Need one specific value
            elif max_sequence == 3:
                return 0.1  # Need two specific values in order
        
        return 0.0  # Very unlikely to complete from scratch

    def _calculate_expected_value(self, category, current_values, remaining_rolls):
        """Calculate expected value for a category considering possible rerolls."""
        if remaining_rolls == 0:
            return self.scoresheet.get_potential_score(category, current_values)
        
        current_score = self.scoresheet.get_potential_score(category, current_values)
        counter = Counter(current_values)
        
        # Upper section expected values
        if category in self.UPPER_SECTION:
            target_value = int(category[-1])  # Extract number from category name
            current_count = counter[target_value]
            max_possible = current_count * target_value
            
            # Calculate probability of improving through rerolls
            for additional in range(1, 6 - current_count):
                prob = self._calculate_probability_of_value(current_count + additional, 
                                                         current_count, 
                                                         5 - current_count)
                max_possible += prob * ((current_count + additional) * target_value)
            
            return max_possible
        
        # Lower section expected values
        if category == YAHTZEE:
            if current_score == 50:  # Already have Yahtzee
                return 50
            most_common = counter.most_common(1)[0][1]
            prob = self._calculate_probability_of_value(5, most_common, 5 - most_common)
            return 50 * prob
            
        elif category in [THREE_OF_A_KIND, FOUR_OF_A_KIND]:
            target_count = 3 if category == THREE_OF_A_KIND else 4
            most_common = counter.most_common(1)[0][1]
            if most_common >= target_count:
                return sum(current_values)
            
            prob = self._calculate_probability_of_value(target_count, 
                                                     most_common, 
                                                     5 - most_common)
            return prob * sum(current_values)
            
        elif category == FULL_HOUSE:
            if len(counter) == 2 and 2 in counter.values() and 3 in counter.values():
                return 25
            # Calculate probability of completing full house from current state
            prob = 0.1 if len(counter) == 2 else 0.05  # Simplified probability
            return 25 * prob
            
        elif category in [SMALL_STRAIGHT, LARGE_STRAIGHT]:
            score = 30 if category == SMALL_STRAIGHT else 40
            prob = self._calculate_straight_probability(current_values, 
                                                     remaining_rolls * 5,
                                                     4 if category == SMALL_STRAIGHT else 5)
            return score * prob
            
        elif category == CHANCE:
            # For chance, consider average roll value (3.5) for rerollable dice
            current_sum = sum(current_values)
            rerollable = min(2, remaining_rolls) * 5  # Max 2 rerolls
            return current_sum + (rerollable * 3.5)
        
        return 0

    def _choose_category_hard(self, dice_values):
        """
        Advanced strategy: Uses expected value calculations and game state analysis.
        Considers optimal long-term strategy, category synergies, and risk assessment.
        """
        available_categories = self.scoresheet.get_available_categories()
        if not available_categories:
            return None
            
        # Calculate game progress
        total_categories = 13
        filled_categories = total_categories - len(available_categories)
        game_progress = filled_categories / total_categories
            
        # Calculate expected values for each available category
        expected_values = {}
        for category in available_categories:
            # Base expected value
            base_value = self._calculate_expected_value(category, dice_values, 0)
            
            # Apply strategic weights
            weight = 1.0
            
            # Upper section bonus strategy
            if category in self.UPPER_SECTION:
                upper_total = sum(self.scoresheet.scores.get(cat, 0) or 0 
                                for cat in self.UPPER_SECTION 
                                if self.scoresheet.scores.get(cat) is not None)
                remaining_upper = sum(1 for cat in self.UPPER_SECTION 
                                   if cat in available_categories)
                
                if upper_total < 63:  # Haven't secured bonus yet
                    needed_per_remaining = (63 - upper_total) / max(1, remaining_upper)
                    if base_value >= needed_per_remaining:
                        weight *= 1.2  # Boost weight if this helps achieve bonus
                    elif game_progress > 0.5 and base_value >= 0.8 * needed_per_remaining:
                        weight *= 1.1  # Slightly boost if close to needed average late game
            
            # Yahtzee and bonus Yahtzee strategy
            if category == YAHTZEE:
                if self.scoresheet.scores.get(YAHTZEE) == 50:
                    weight *= 1.3  # Prioritize bonus Yahtzees
                elif game_progress < 0.5:
                    weight *= 1.1  # Slightly prioritize first Yahtzee early game
            
            # Consider future potential
            future_value = self._calculate_future_potential(category, available_categories)
            opportunity_cost = future_value * (1 - game_progress)
            weight *= (1 - opportunity_cost/50)  # Scale down if high opportunity cost
            
            # Add category synergy analysis
            synergy_bonus = self._calculate_category_synergy(category, dice_values)
            weight *= (1 + synergy_bonus)
            
            # Add risk assessment
            risk_factor = self._assess_risk(category, dice_values, game_progress)
            weight *= (1 + risk_factor)
            
            # Late game adjustments
            if game_progress > 0.7:
                if category in [CHANCE, THREE_OF_A_KIND]:
                    weight *= 1.1  # Prefer reliable scores late
                if category == YAHTZEE and self.scoresheet.scores.get(YAHTZEE) is None:
                    weight *= 0.7  # Penalize attempting first Yahtzee late
            
            # Early game adjustments
            if game_progress < 0.3:
                if category == CHANCE:
                    weight *= 0.8  # Save Chance for later
                if category in self.UPPER_SECTION and base_value >= 8:
                    weight *= 1.1  # Prioritize good upper section early
            
            # Calculate final weighted value
            expected_values[category] = base_value * weight
            
            # Add minimum score protection
            if base_value > 0 and expected_values[category] < base_value * 0.5:
                expected_values[category] = base_value * 0.5  # Don't reduce too much
        
        # Choose category with highest expected value
        return max(expected_values.items(), key=lambda x: x[1])[0]
    
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