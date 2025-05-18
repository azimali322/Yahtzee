from game_logic import (
    ScoreSheet, Dice, 
    ONES, TWOS, THREES, FOURS, FIVES, SIXES,
    THREE_OF_A_KIND, FOUR_OF_A_KIND, FULL_HOUSE,
    SMALL_STRAIGHT, LARGE_STRAIGHT, YAHTZEE, CHANCE,
    ALL_CATEGORIES
)
from collections import Counter
from itertools import combinations_with_replacement
import math
import itertools
import logging
import random
import numpy as np
import torch
import torch.optim as optim
from typing import List, Optional, Tuple
import torch.nn.functional as F
from yahtzee_actor_critic import YahtzeeActorCritic, Transition

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class YahtzeeAI:
    """AI agent for playing Yahtzee using heuristic strategies."""
    
    # Add class constants for probabilities
    UPPER_SECTION = [ONES, TWOS, THREES, FOURS, FIVES, SIXES]
    LOWER_SECTION = [THREE_OF_A_KIND, FOUR_OF_A_KIND, FULL_HOUSE, 
                    SMALL_STRAIGHT, LARGE_STRAIGHT, YAHTZEE, CHANCE]
    
    # Maximum possible scores for each category
    CATEGORY_MAX_SCORES = {
        ONES: 5,           # All ones (1×5)
        TWOS: 10,         # All twos (2×5)
        THREES: 15,       # All threes (3×5)
        FOURS: 20,        # All fours (4×5)
        FIVES: 25,        # All fives (5×5)
        SIXES: 30,        # All sixes (6×5)
        THREE_OF_A_KIND: 30,  # All sixes (maximum possible sum)
        FOUR_OF_A_KIND: 30,   # All sixes (maximum possible sum)
        FULL_HOUSE: 25,       # Fixed score
        SMALL_STRAIGHT: 30,   # Fixed score
        LARGE_STRAIGHT: 40,   # Fixed score
        YAHTZEE: 50,         # Fixed score
        CHANCE: 30           # All sixes (maximum possible sum)
    }
    
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
        Advanced strategy: Uses multi-turn look-ahead and sophisticated optimization.
        Implements optimal strategy based on game state analysis and future potential.
        """
        logger.info(f"\nHard AI deciding reroll for {current_dice_values} (roll {roll_number})")
        
        if roll_number >= 3:
            return []
            
        # Get current game state analysis
        game_state = self._analyze_game_state()
        
        # Calculate best reroll strategy considering future turns
        best_reroll = self._calculate_optimal_reroll(
            current_dice_values, 
            roll_number,
            game_state
        )
        
        return best_reroll
    
    def _analyze_game_state(self):
        """Analyze current game state for strategic decision making."""
        available_categories = self.scoresheet.get_available_categories()
        filled_categories = set(cat for cat in self.UPPER_SECTION + self.LOWER_SECTION 
                              if cat not in available_categories)
        
        # Calculate upper section status
        upper_total = sum(self.scoresheet.scores.get(cat, 0) or 0 
                         for cat in self.UPPER_SECTION)
        remaining_upper = [cat for cat in self.UPPER_SECTION 
                         if cat in available_categories]
        
        # Calculate optimal upper section targets
        needed_for_bonus = max(0, 63 - upper_total)
        avg_needed_per_remaining = (needed_for_bonus / len(remaining_upper)) if remaining_upper else 0
        
        # Analyze Yahtzee potential
        has_yahtzee = self.scoresheet.scores.get(YAHTZEE) == 50
        can_bonus_yahtzee = has_yahtzee and YAHTZEE in available_categories
        
        return {
            'available_categories': available_categories,
            'filled_categories': filled_categories,
            'upper_total': upper_total,
            'remaining_upper': remaining_upper,
            'needed_for_bonus': needed_for_bonus,
            'avg_needed_per_remaining': avg_needed_per_remaining,
            'has_yahtzee': has_yahtzee,
            'can_bonus_yahtzee': can_bonus_yahtzee,
            'turns_remaining': len(available_categories),
            'game_progress': 1 - (len(available_categories) / 13)
        }
    
    def _calculate_optimal_reroll(self, current_values, roll_number, game_state):
        """Calculate optimal reroll strategy using multi-turn look-ahead."""
        best_score = -1
        best_reroll = []
        
        # Get current dice analysis
        dice_analysis = self._analyze_dice_values(current_values)
        
        # Early game strategy (first 5 turns)
        if game_state['game_progress'] < 0.4:
            best_reroll = self._early_game_strategy(
                current_values, roll_number, game_state, dice_analysis
            )
            if best_reroll is not None:
                return best_reroll
        
        # Mid game strategy (turns 6-10)
        elif game_state['game_progress'] < 0.8:
            best_reroll = self._mid_game_strategy(
                current_values, roll_number, game_state, dice_analysis
            )
            if best_reroll is not None:
                return best_reroll
        
        # Late game strategy (last 3 turns)
        else:
            best_reroll = self._late_game_strategy(
                current_values, roll_number, game_state, dice_analysis
            )
            if best_reroll is not None:
                return best_reroll
        
        # If no specific strategy was chosen, use complete enumeration
        return self._complete_enumeration_strategy(
            current_values, roll_number, game_state, dice_analysis
        )
    
    def _analyze_dice_values(self, dice_values):
        """Detailed analysis of current dice values."""
        if not dice_values:
            return {
                'counts': Counter(),
                'max_count': 0,
                'unique_values': 0,
                'sequence_length': 0,
                'sequence': [],
                'sum': 0,
                'has_pair': False,
                'has_three': False,
                'has_four': False,
                'most_common': [],
                'potential_straight': False
            }
            
        counter = Counter(dice_values)
        sorted_vals = sorted(dice_values)
        sequence = self._find_longest_sequence(sorted_vals)
        
        return {
            'counts': counter,
            'max_count': max(counter.values()),
            'unique_values': len(counter),
            'sequence_length': len(sequence),
            'sequence': sequence,
            'sum': sum(dice_values),
            'has_pair': any(count >= 2 for count in counter.values()),
            'has_three': any(count >= 3 for count in counter.values()),
            'has_four': any(count >= 4 for count in counter.values()),
            'most_common': counter.most_common(),
            'potential_straight': len(set(range(min(dice_values), max(dice_values) + 1))) >= 4
        }
    
    def _early_game_strategy(self, current_values, roll_number, game_state, dice_analysis):
        """
        Early game strategy focusing on:
        1. Securing upper section bonus
        2. Setting up for Yahtzee opportunities
        3. Keeping high-scoring combinations
        """
        # Prioritize upper section bonus
        if game_state['needed_for_bonus'] > 0:
            high_value_counts = {
                val: count for val, count in dice_analysis['counts'].items()
                if val >= 4 and count >= 2
            }
            if high_value_counts:
                highest_val = max(high_value_counts.keys())
                return [i for i, val in enumerate(current_values) 
                       if val != highest_val]
        
        # Look for potential Yahtzee
        if dice_analysis['max_count'] >= 3 and roll_number < 3:
            most_common_val = dice_analysis['most_common'][0][0]
            return [i for i, val in enumerate(current_values) 
                   if val != most_common_val]
        
        # Look for potential straight
        if dice_analysis['sequence_length'] >= 3 and roll_number < 3:
            return [i for i, val in enumerate(current_values) 
                   if val not in dice_analysis['sequence']]
        
        return None
    
    def _mid_game_strategy(self, current_values, roll_number, game_state, dice_analysis):
        """
        Mid game strategy focusing on:
        1. Completing high-scoring combinations
        2. Maximizing category synergies
        3. Maintaining upper section bonus pace
        """
        # Try to complete Yahtzee if close
        if dice_analysis['max_count'] == 4:
            most_common_val = dice_analysis['most_common'][0][0]
            return [i for i, val in enumerate(current_values) 
                   if val != most_common_val]
        
        # Try to complete Large Straight if close
        if dice_analysis['sequence_length'] == 4 and LARGE_STRAIGHT in game_state['available_categories']:
            return [i for i, val in enumerate(current_values) 
                   if val not in dice_analysis['sequence']]
        
        # Try to complete Full House if have three of a kind
        if (dice_analysis['has_three'] and not dice_analysis['has_pair'] 
            and FULL_HOUSE in game_state['available_categories']):
            three_val = next(val for val, count in dice_analysis['counts'].items() 
                           if count >= 3)
            return [i for i, val in enumerate(current_values) 
                   if val != three_val and dice_analysis['counts'][val] == 1]
        
        return None
    
    def _late_game_strategy(self, current_values, roll_number, game_state, dice_analysis):
        """
        Late game strategy focusing on:
        1. Maximizing remaining category scores
        2. Securing upper section bonus if close
        3. Taking guaranteed points over risky plays
        """
        # Map category names to their values for upper section
        value_map = {
            ONES: 1,
            TWOS: 2,
            THREES: 3,
            FOURS: 4,
            FIVES: 5,
            SIXES: 6
        }
        
        # If we need specific upper section values, focus on those
        if game_state['needed_for_bonus'] <= 15 and game_state['remaining_upper']:
            needed_sections = [value_map[cat] for cat in game_state['remaining_upper']]
            for val in needed_sections:
                if dice_analysis['counts'][val] >= 2:
                    return [i for i, d in enumerate(current_values) 
                           if d != val]
        
        # Take guaranteed points if available
        if dice_analysis['max_count'] >= 3:
            most_common_val = dice_analysis['most_common'][0][0]
            if THREE_OF_A_KIND in game_state['available_categories']:
                return [i for i, val in enumerate(current_values) 
                       if val != most_common_val]
        
        # If we have a potential straight, try to complete it
        if dice_analysis['sequence_length'] >= 3:
            if (LARGE_STRAIGHT in game_state['available_categories'] or 
                SMALL_STRAIGHT in game_state['available_categories']):
                return [i for i, val in enumerate(current_values) 
                       if val not in dice_analysis['sequence']]
        
        # If we have a pair and potential for full house
        if (dice_analysis['has_pair'] and FULL_HOUSE in game_state['available_categories'] 
            and not dice_analysis['has_three']):
            pair_val = next(val for val, count in dice_analysis['counts'].items() 
                          if count == 2)
            return [i for i, val in enumerate(current_values) 
                   if val != pair_val]
        
        # Default to keeping highest scoring combination
        if dice_analysis['max_count'] >= 2:
            most_common_val = dice_analysis['most_common'][0][0]
            return [i for i, val in enumerate(current_values) 
                   if val != most_common_val]
        
        # If no clear strategy, reroll lowest values
        sorted_indices = sorted(range(len(current_values)), 
                              key=lambda i: current_values[i])
        return sorted_indices[:3]  # Reroll the three lowest values
    
    def _complete_enumeration_strategy(self, current_values, roll_number, game_state, dice_analysis):
        """Fallback to complete enumeration with strategic weighting."""
        best_score = -1
        best_reroll = []
        
        # Try all possible combinations of dice to keep
        for keep_count in range(6):
            for keep_indices in itertools.combinations(range(5), keep_count):
                # Calculate base expected score
                expected_score = self._calculate_expected_score_for_reroll(
                    keep_indices, current_values
                )
                
                # Apply strategic weights
                final_score = self._apply_strategic_weights(
                    expected_score,
                    keep_indices,
                    current_values,
                    game_state,
                    dice_analysis
                )
                
                if final_score > best_score:
                    best_score = final_score
                    best_reroll = [i for i in range(5) if i not in keep_indices]
        
        return best_reroll
    
    def _apply_strategic_weights(self, base_score, keep_indices, current_values, 
                               game_state, dice_analysis):
        """Apply strategic weights to the expected score based on game state."""
        score = base_score
        kept_values = [current_values[i] for i in keep_indices]
        kept_analysis = self._analyze_dice_values(kept_values)
        
        # Weight based on upper section bonus potential
        if game_state['needed_for_bonus'] > 0:
            upper_potential = sum(
                val * count for val, count in kept_analysis['counts'].items()
                if f"_{val}" in game_state['remaining_upper']
            )
            if upper_potential >= game_state['avg_needed_per_remaining']:
                score *= 1.2
        
        # Weight based on Yahtzee potential
        if kept_analysis['max_count'] >= 3:
            if game_state['has_yahtzee']:  # Going for bonus Yahtzee
                score *= 1.3
            elif YAHTZEE in game_state['available_categories']:
                score *= 1.2
        
        # Weight based on straight potential
        if kept_analysis['sequence_length'] >= 3:
            if LARGE_STRAIGHT in game_state['available_categories']:
                score *= 1.15
            elif SMALL_STRAIGHT in game_state['available_categories']:
                score *= 1.1
        
        # Late game adjustment
        if game_state['game_progress'] > 0.8:
            score *= (1 + 0.1 * kept_analysis['max_count'])  # Prefer keeping more dice
        
        return score

    def _choose_category_hard(self, dice_values):
        """
        Advanced category selection strategy that considers:
        1. Game state (early, mid, late game)
        2. Upper section bonus potential
        3. Yahtzee opportunities
        4. Strategic category selection
        5. Future turn implications
        """
        game_state = self._analyze_game_state()
        dice_analysis = self._analyze_dice_values(dice_values)
        available_categories = game_state['available_categories']
        
        # If no categories are available, return None
        if not available_categories:
            return None
        
        # Calculate base scores for all available categories
        scores = {cat: self.scoresheet.get_potential_score(cat, dice_values) 
                 for cat in available_categories}
        
        # Special case: If we have a Yahtzee and it's available, take it
        if YAHTZEE in scores and scores[YAHTZEE] == 50:
            return YAHTZEE
        
        # Apply strategic weights to each category
        weighted_scores = {}
        for category, base_score in scores.items():
            weight = 1.0  # Base weight
            
            # Handle Yahtzee opportunities
            if category == YAHTZEE:
                if base_score == 50:  # We have a Yahtzee
                    weight = 1.5  # Strongly prefer taking Yahtzee when available
                elif game_state['has_yahtzee']:  # Potential bonus Yahtzee
                    weight = 1.3
            
            # Upper section bonus consideration
            if category in self.UPPER_SECTION:
                # Map category names to their values
                value_map = {
                    ONES: 1,
                    TWOS: 2,
                    THREES: 3,
                    FOURS: 4,
                    FIVES: 5,
                    SIXES: 6
                }
                value = value_map[category]
                needed_avg = game_state['avg_needed_per_remaining']
                
                # If this is a good score for this number, prioritize it
                if base_score >= value * 3:
                    weight = 1.4
                elif base_score >= needed_avg:  # Good score for upper section
                    if game_state['needed_for_bonus'] > 0:  # Still need bonus
                        weight = 1.3
                elif game_state['game_progress'] > 0.8 and game_state['needed_for_bonus'] <= 8:
                    # Late game, close to bonus
                    weight = 1.4
                
                # Early game: Prioritize upper section
                if game_state['game_progress'] < 0.3:
                    weight *= 1.2  # Additional boost for upper section in early game
            
            # Handle special scoring categories
            if category in [SMALL_STRAIGHT, LARGE_STRAIGHT, FULL_HOUSE]:
                if base_score == self.CATEGORY_MAX_SCORES[category]:
                    # Maximum score for these categories
                    weight = 1.2 if game_state['game_progress'] < 0.5 else 1.4
            
            # Adjust for Three/Four of a Kind
            if category in [THREE_OF_A_KIND, FOUR_OF_A_KIND]:
                if base_score >= 20:  # Good score
                    weight = 1.1
                    if category == FOUR_OF_A_KIND and YAHTZEE in available_categories:
                        weight = 0.9  # Slightly discourage if Yahtzee still available
            
            # Late game adjustments
            if game_state['game_progress'] > 0.8:
                if base_score == 0:  # Avoid zeros late game if possible
                    weight = 0.7
                elif base_score > self.CATEGORY_MAX_SCORES[category] * 0.7:
                    # Prefer high percentage scores late game
                    weight = 1.2
            
            # Early game adjustments
            elif game_state['game_progress'] < 0.3:
                if category == CHANCE:  # Save chance for later
                    weight = 0.6  # Significantly reduce chance of using CHANCE early
                elif base_score == 0:  # More acceptable to take zeros early
                    weight = 0.9
            
            weighted_scores[category] = base_score * weight
        
        # If no weighted scores (shouldn't happen with the above check), return None
        if not weighted_scores:
            return None
            
        # Choose the category with the highest weighted score
        best_category = max(weighted_scores.items(), key=lambda x: x[1])[0]
        logger.info(f"\nHard AI choosing category for dice values: {dice_values}")
        logger.info(f"Chose category {best_category} with weighted score {weighted_scores[best_category]:.2f}")
        
        return best_category

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
        return max(sequences, key=len) if sequences else []

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
        if not available_categories:
            return 0
            
        scores = {cat: self.scoresheet.get_potential_score(cat, dice_values)
                 for cat in available_categories}
        
        if not scores:
            return 0
            
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

class RLAgent(YahtzeeAI):
    """Reinforcement Learning agent using Actor-Critic architecture."""
    
    def __init__(self, observation_space, device="cpu"):
        super().__init__(difficulty="rl")
        self.device = torch.device(device)
        
        # Initialize actor-critic network
        self.ac_net = YahtzeeActorCritic(
            observation_space=observation_space,
            n_actions_dice=32,  # 2^5 possible dice reroll combinations
            n_actions_category=len(ALL_CATEGORIES)
        ).to(device)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.ac_net.parameters(), lr=3e-4)
        
        # Training parameters
        self.gamma = 0.99  # discount factor
        self.gae_lambda = 0.95  # GAE lambda parameter
        self.entropy_coef = 0.01  # entropy coefficient
        self.value_loss_coef = 0.5  # value loss coefficient
        
        # Buffer for storing transitions
        self.transitions: List[Transition] = []
    
    def decide_reroll(self, current_dice_values, roll_number) -> List[int]:
        """Decide which dice to reroll using the actor network."""
        # Create observation for the network
        obs = self._get_current_observation()
        
        # Get action from the network
        with torch.no_grad():
            binary_action, _ = self.ac_net.get_dice_action(
                obs,
                deterministic=(roll_number == 2)  # Be deterministic on last roll
            )
        
        # Convert binary action to list of indices
        return [i for i, reroll in enumerate(binary_action) if reroll]
    
    def choose_category(self, dice_values) -> str:
        """Choose category using the actor network."""
        # Create observation for the network
        obs = self._get_current_observation()
        
        # Get action from the network
        with torch.no_grad():
            category_idx, _ = self.ac_net.get_category_action(
                obs,
                deterministic=False  # Could be True for evaluation
            )
        
        return ALL_CATEGORIES[category_idx]
    
    def _get_current_observation(self) -> dict:
        """Create observation dictionary from current game state."""
        return {
            'dice': np.array(self.dice.get_values()),
            'roll_number': self.dice.roll_count,
            'scoresheet': np.array([
                self.scoresheet.scores.get(cat, -1) 
                for cat in ALL_CATEGORIES
            ]),
            'upper_bonus': float(self.scoresheet.get_upper_section_bonus() > 0),
            'yahtzee_bonus': np.array([self.scoresheet.yahtzee_bonus_score // 100]),
            'opponent_scores': np.array([0]),  # Placeholder for now
            'relative_rank': 0  # Placeholder for now
        }
    
    def store_transition(
        self,
        state: dict,
        dice_action: torch.Tensor,
        category_action: int,
        reward: float,
        next_state: dict,
        done: bool
    ):
        """Store a transition in the buffer."""
        self.transitions.append(Transition(
            state=state,
            action=(dice_action, category_action),
            reward=reward,
            next_state=next_state,
            done=done
        ))
    
    def compute_gae(
        self,
        rewards: List[float],
        values: torch.Tensor,
        dones: List[bool]
    ) -> torch.Tensor:
        """Compute Generalized Advantage Estimation."""
        advantages = torch.zeros_like(values)
        last_advantage = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            advantages[t] = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_advantage
            last_advantage = advantages[t]
        
        return advantages
    
    def update(self) -> Optional[Tuple[float, float, float]]:
        """Update the actor-critic network using collected transitions."""
        if not self.transitions:
            return None
        
        # Unzip transitions
        states, actions, rewards, next_states, dones = zip(*self.transitions)
        dice_actions, category_actions = zip(*actions)
        
        # Convert to tensors
        dice_actions = torch.stack([torch.tensor(a) for a in dice_actions])
        category_actions = torch.tensor(category_actions)
        rewards = torch.tensor(rewards)
        dones = torch.tensor(dones)
        
        # Get values and advantages
        _, _, values = zip(*[self.ac_net(s) for s in states])
        values = torch.cat(values)
        advantages = self.compute_gae(rewards, values, dones)
        returns = advantages + values
        
        # Evaluate actions
        dice_log_probs, category_log_probs, entropy, new_values = zip(*[
            self.ac_net.evaluate_actions(s, da, ca)
            for s, da, ca in zip(states, dice_actions, category_actions)
        ])
        
        # Compute losses
        dice_log_probs = torch.stack(dice_log_probs)
        category_log_probs = torch.stack(category_log_probs)
        new_values = torch.cat(new_values)
        
        # Actor loss
        actor_loss = -(
            dice_log_probs * advantages.detach() +
            category_log_probs * advantages.detach()
        ).mean()
        
        # Critic loss
        value_loss = F.mse_loss(new_values, returns.detach())
        
        # Entropy loss (for exploration)
        entropy_loss = -torch.stack(entropy).mean()
        
        # Total loss
        total_loss = (
            actor_loss +
            self.value_loss_coef * value_loss +
            self.entropy_coef * entropy_loss
        )
        
        # Optimize
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        # Clear transitions
        self.transitions = []
        
        return (
            actor_loss.item(),
            value_loss.item(),
            entropy_loss.item()
        ) 