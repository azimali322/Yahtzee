from game_logic import (
    ScoreSheet, Dice, 
    ONES, TWOS, THREES, FOURS, FIVES, SIXES,
    THREE_OF_A_KIND, FOUR_OF_A_KIND, FULL_HOUSE,
    SMALL_STRAIGHT, LARGE_STRAIGHT, YAHTZEE, CHANCE
)
from collections import Counter
from itertools import combinations_with_replacement
import math
import itertools
import logging
import random

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class YahtzeeAI:
    """AI agent for playing Yahtzee using heuristic strategies."""
    
    # Add class constants for probabilities
    UPPER_SECTION = [ONES, TWOS, THREES, FOURS, FIVES, SIXES]
    LOWER_SECTION = [THREE_OF_A_KIND, FOUR_OF_A_KIND, FULL_HOUSE, 
                    SMALL_STRAIGHT, LARGE_STRAIGHT, YAHTZEE, CHANCE]
    
    def __init__(self, difficulty="medium"):
        """Initialize the AI agent with a difficulty level."""
        self.difficulty = difficulty  # "easy", "medium", "hard", "greedy1", "greedy2", "greedy3", "random"
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
        elif self.difficulty == "greedy1":
            return []  # Never reroll
        elif self.difficulty == "greedy2":
            return self._decide_reroll_greedy2(current_dice_values, roll_number)
        elif self.difficulty == "greedy3":
            return self._decide_reroll_greedy3(current_dice_values, roll_number)
        elif self.difficulty == "random":
            return self._decide_reroll_random(current_dice_values, roll_number)
        else:  # medium difficulty
            return self._decide_reroll_medium(current_dice_values, roll_number)
    
    def choose_category(self, dice_values):
        """
        Choose the best category to score based on current dice values.
        Returns the chosen category name.
        """
        if self.difficulty in ["greedy1", "greedy2", "greedy3"]:
            return self._choose_category_greedy(dice_values)
        elif self.difficulty == "easy":
            return self._choose_category_easy(dice_values)
        elif self.difficulty == "hard":
            return self._choose_category_hard(dice_values)
        elif self.difficulty == "random":
            return self._choose_category_random(dice_values)
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

    def _calculate_expected_score_for_reroll(self, keep_indices, current_values):
        """Calculate expected score for a given set of dice to keep using complete enumeration."""
        # Convert to list for easier manipulation
        reroll_count = 5 - len(keep_indices)
        
        if reroll_count == 0:
            return self._get_max_score(current_values)
        
        # Keep track of which values we're keeping
        kept_values = [current_values[i] for i in keep_indices]
        logger.debug(f"Keeping dice values: {kept_values}")
        
        # Generate all possible combinations for the rerolled dice
        total_score = 0
        total_combinations = 0
        
        # Use combinations_with_replacement to generate all possible reroll outcomes
        for reroll_values in itertools.combinations_with_replacement(range(1, 7), reroll_count):
            # Combine kept values with current reroll combination
            combined_values = kept_values + list(reroll_values)
            score = self._get_max_score(combined_values)
            total_score += score
            total_combinations += 1
        
        # Calculate average score across all possible combinations
        expected_score = total_score / total_combinations if total_combinations > 0 else 0
        logger.debug(f"Expected score for keeping {kept_values}: {expected_score:.2f}")
        
        return expected_score

    def _get_max_score(self, dice_values):
        """Get the maximum possible score across all available categories."""
        available_categories = self.scoresheet.get_available_categories()
        scores = {cat: self.scoresheet.get_potential_score(cat, dice_values) 
                 for cat in available_categories}
        best_category = max(scores.items(), key=lambda x: x[1])
        logger.debug(f"Best score for {dice_values}: {best_category[1]} in category {best_category[0]}")
        return best_category[1]

    def _decide_reroll_greedy2(self, current_dice_values, roll_number):
        """
        Greedy level-2 strategy: Only one reroll allowed.
        Choose reroll that maximizes expected score this round.
        """
        logger.info(f"\nGreedy2 AI deciding reroll for {current_dice_values} (roll {roll_number})")
        
        if roll_number >= 2:
            logger.info("No more rerolls allowed for greedy2")
            return []
            
        best_score = self._get_max_score(current_dice_values)
        best_reroll = []
        logger.info(f"Current best score (no reroll): {best_score}")
        
        # Check for special cases first
        counter = Counter(current_dice_values)
        pairs = [val for val, count in counter.items() if count >= 2]
        
        # If we have two pairs, only consider rerolling the fifth die
        if len(pairs) == 2 and all(counter[p] == 2 for p in pairs):
            non_pair_indices = [i for i, val in enumerate(current_dice_values) if val not in pairs]
            if non_pair_indices:
                expected_score = self._calculate_expected_score_for_reroll(
                    [i for i in range(5) if i not in non_pair_indices],
                    current_dice_values
                )
                if expected_score > best_score:
                    best_score = expected_score
                    best_reroll = non_pair_indices
                    logger.info(f"New best reroll found: {best_reroll} with expected score {best_score:.2f}")
            return best_reroll
        
        # For all other cases, try all possible combinations of dice to keep
        for keep_count in range(6):
            for keep_indices in itertools.combinations(range(5), keep_count):
                expected_score = self._calculate_expected_score_for_reroll(keep_indices, current_dice_values)
                logger.debug(f"Keeping indices {keep_indices}, expected score: {expected_score:.2f}")
                if expected_score > best_score:
                    best_score = expected_score
                    best_reroll = [i for i in range(5) if i not in keep_indices]
                    logger.info(f"New best reroll found: {best_reroll} with expected score {best_score:.2f}")
        
        logger.info(f"Final decision - reroll indices: {best_reroll}")
        return best_reroll

    def _decide_reroll_greedy3(self, current_dice_values, roll_number):
        """
        Greedy level-3 strategy: Up to two rerolls allowed.
        Choose reroll that maximizes expected score this round.
        """
        logger.info(f"\nGreedy3 AI deciding reroll for {current_dice_values} (roll {roll_number})")
        
        if roll_number >= 3:
            logger.info("No more rerolls allowed")
            return []
            
        best_score = self._get_max_score(current_dice_values)
        best_reroll = []
        logger.info(f"Current best score (no reroll): {best_score}")
        
        # Similar to greedy2, but considers the possibility of a second reroll
        for keep_count in range(6):
            for keep_indices in itertools.combinations(range(5), keep_count):
                expected_score = self._calculate_expected_score_for_reroll(keep_indices, current_dice_values)
                logger.debug(f"Keeping indices {keep_indices}, expected score: {expected_score:.2f}")
                if expected_score > best_score:
                    best_score = expected_score
                    best_reroll = [i for i in range(5) if i not in keep_indices]
                    logger.info(f"New best reroll found: {best_reroll} with expected score {best_score:.2f}")
        
        logger.info(f"Final decision - reroll indices: {best_reroll}")
        return best_reroll

    def _choose_category_greedy(self, dice_values):
        """
        Greedy strategy: Simply choose the category that gives the highest score
        for the current dice values.
        """
        logger.info(f"\nGreedy AI choosing category for dice values: {dice_values}")
        available_categories = self.scoresheet.get_available_categories()
        scores = {}
        
        # Calculate scores for each available category
        for cat in available_categories:
            score = self.scoresheet.get_potential_score(cat, dice_values)
            scores[cat] = score
            logger.debug(f"Category {cat}: {score} points")
        
        # Special handling for Three of a Kind vs Four of a Kind
        if THREE_OF_A_KIND in scores and FOUR_OF_A_KIND in scores:
            # If the scores are equal, prefer Four of a Kind
            if scores[THREE_OF_A_KIND] == scores[FOUR_OF_A_KIND]:
                scores[THREE_OF_A_KIND] = 0  # Effectively remove Three of a Kind from consideration
        
        best_category = max(scores.items(), key=lambda x: x[1])
        logger.info(f"Chose category {best_category[0]} with score {best_category[1]}")
        return best_category[0]

    def test_greedy_expected_value_accuracy(self):
        """Test accuracy of expected value calculations."""
        test_cases = [
            # Format: (current_dice, keep_indices, min_expected_score)
            ([6, 6, 6, 1, 1], [0, 1, 2], 25),  # Three 6s
            ([5, 5, 4, 4, 1], [0, 1, 2, 3], 25),  # Two pairs
            ([1, 2, 3, 4, 6], [0, 1, 2, 3], 30),  # Potential straight
        ]
        
        for dice_values, keep_indices, min_expected in test_cases:
            score = self._calculate_expected_score_for_reroll(keep_indices, dice_values)
            self.assertGreaterEqual(score, min_expected,
                f"Expected value too low for {dice_values} keeping {keep_indices}")

    def _decide_reroll_random(self, current_dice_values, roll_number):
        """
        Random strategy: Randomly decide which dice to reroll.
        Each die has a 50% chance of being rerolled.
        """
        if roll_number >= 3:
            logger.info("No more rerolls allowed")
            return []
            
        # For each die, randomly decide whether to reroll it
        reroll_indices = []
        for i in range(len(current_dice_values)):
            if random.random() < 0.5:  # 50% chance to reroll each die
                reroll_indices.append(i)
                
        logger.info(f"\nRandom AI deciding reroll for {current_dice_values} (roll {roll_number})")
        logger.info(f"Randomly chose to reroll indices: {reroll_indices}")
        return reroll_indices

    def _choose_category_random(self, dice_values):
        """
        Random strategy: Choose a random available category.
        """
        available_categories = self.scoresheet.get_available_categories()
        if not available_categories:
            return None
            
        chosen_category = random.choice(list(available_categories))
        score = self.scoresheet.get_potential_score(chosen_category, dice_values)
        
        logger.info(f"\nRandom AI choosing category for dice values: {dice_values}")
        logger.info(f"Randomly chose category {chosen_category} with score {score}")
        
        return chosen_category 