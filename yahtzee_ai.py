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

    def _calculate_expected_score_for_reroll(self, keep_indices, current_values):
        """Calculate expected score for a given set of dice to keep."""
        # Convert to list for easier manipulation
        reroll_count = 5 - len(keep_indices)
        
        if reroll_count == 0:
            logger.debug(f"No dice to reroll, returning current max score")
            return self._get_max_score(current_values)
        
        # Keep track of which values we're keeping
        kept_values = [current_values[i] for i in keep_indices]
        logger.debug(f"Keeping dice values: {kept_values}")
        
        # Early pruning: If keeping a good combination, don't reroll
        kept_counter = Counter(kept_values)
        all_counter = Counter(current_values)
        
        # Check for four of a kind
        max_count = max(all_counter.values(), default=0)
        if max_count >= 4:
            # Find the value that appears 4 or more times
            four_value = [val for val, count in all_counter.items() if count >= 4][0]
            # Keep only the four of a kind dice
            four_indices = [i for i, val in enumerate(current_values) if val == four_value][:4]
            if set(keep_indices) != set(four_indices):
                return -1  # Signal that this is not an optimal keep
        
        # Check for full house
        if len(all_counter) == 2 and 2 in all_counter.values() and 3 in all_counter.values():
            # Should keep all dice for a full house
            if len(keep_indices) != 5:
                return -1
        
        # Check for large straight
        sorted_vals = sorted(set(current_values))
        if len(sorted_vals) >= 5 and sorted_vals[-1] - sorted_vals[0] == 4:
            # Should keep all dice for a large straight
            if len(keep_indices) != 5:
                return -1
        
        # Check for small straight
        if len(self._find_longest_sequence(sorted_vals)) >= 4:
            # Should keep at least the straight dice
            straight_indices = set()
            for i, val in enumerate(current_values):
                if val in sorted_vals[:4]:
                    straight_indices.add(i)
            if not straight_indices.issubset(set(keep_indices)):
                return -1
        
        # Calculate expected score using probability-based approach
        total_score = 0
        
        # For upper section categories
        for value in range(1, 7):
            prob = 1/6  # Probability of rolling this value
            for count in range(reroll_count + 1):
                # Probability of getting exactly 'count' of this value
                count_prob = self._calculate_probability_of_value(count, 0, reroll_count)
                # Add kept values of this number
                total_count = count + kept_counter[value]
                # Calculate score if using this as upper section
                score = total_count * value
                total_score += prob * count_prob * score
        
        # For lower section categories
        # Three of a kind
        if max(kept_counter.values(), default=0) >= 2:
            prob_three = self._calculate_probability_of_value(1, 0, reroll_count)
            total_score = max(total_score, prob_three * 3 * max(kept_counter.keys()))
        
        # Four of a kind
        if max(kept_counter.values(), default=0) >= 3:
            prob_four = self._calculate_probability_of_value(1, 0, reroll_count)
            total_score = max(total_score, prob_four * 4 * max(kept_counter.keys()))
        
        # Full house
        if len(kept_counter) == 2 and 2 in kept_counter.values() and 3 in kept_counter.values():
            total_score = max(total_score, 25)
        
        # Small straight
        kept_sequence = len(self._find_longest_sequence(sorted(kept_values)))
        if kept_sequence >= 3:
            prob_straight = self._calculate_straight_probability(kept_values, reroll_count, 4)
            total_score = max(total_score, prob_straight * 30)
        
        # Large straight
        if kept_sequence >= 4:
            prob_straight = self._calculate_straight_probability(kept_values, reroll_count, 5)
            total_score = max(total_score, prob_straight * 40)
        
        # Yahtzee
        if max(kept_counter.values(), default=0) >= 4:
            prob_yahtzee = self._calculate_probability_of_value(1, 0, reroll_count)
            total_score = max(total_score, prob_yahtzee * 50)
        
        # Chance (always possible)
        chance_score = sum(kept_values) + reroll_count * 3.5  # 3.5 is average die value
        total_score = max(total_score, chance_score)
        
        logger.debug(f"Expected score for keeping {kept_values}: {total_score:.2f}")
        return total_score

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
        
        # Try all possible combinations of dice to keep
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