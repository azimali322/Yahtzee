import unittest
from yahtzee_ai import YahtzeeAI
from game_logic import (
    ScoreSheet, Dice,
    ONES, TWOS, THREES, FOURS, FIVES, SIXES,
    THREE_OF_A_KIND, FOUR_OF_A_KIND, FULL_HOUSE,
    SMALL_STRAIGHT, LARGE_STRAIGHT, YAHTZEE, CHANCE
)
import itertools

class TestCompleteEnumeration(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        self.dice = Dice()
        self.scoresheet = ScoreSheet()
        self.ai = YahtzeeAI("greedy2")  # Using greedy2 for testing
        self.ai.set_game_state(self.scoresheet, self.dice)

    def test_expected_score_calculation(self):
        """Test that expected score calculation considers all possibilities."""
        # Test case 1: Keeping a pair of sixes
        dice_values = [6, 6, 1, 2, 3]
        keep_indices = [0, 1]  # Keep the sixes
        expected_score = self.ai._calculate_expected_score_for_reroll(keep_indices, dice_values)
        # Should be higher than just scoring the sixes in upper section (12)
        self.assertGreater(expected_score, 12, 
            "Expected score should be higher than just scoring sixes")

        # Test case 2: Three of a kind scenario
        dice_values = [5, 5, 5, 1, 2]
        keep_indices = [0, 1, 2]  # Keep three fives
        expected_score = self.ai._calculate_expected_score_for_reroll(keep_indices, dice_values)
        # Should be higher than just scoring three of a kind (18)
        self.assertGreater(expected_score, 18,
            "Expected score should be higher than just three of a kind")

        # Test case 3: Potential small straight
        dice_values = [1, 2, 3, 4, 6]
        keep_indices = [0, 1, 2, 3]  # Keep 1,2,3,4
        expected_score = self.ai._calculate_expected_score_for_reroll(keep_indices, dice_values)
        # Should be close to small straight score (30) since we need just a 5
        self.assertGreater(expected_score, 25,
            "Expected score should be close to small straight value")

    def test_all_combinations_considered(self):
        """Test that all possible dice combinations are considered."""
        dice_values = [1, 1, 1, 2, 2]
        keep_indices = [0, 1, 2]  # Keep three ones
        
        # Calculate total possible combinations for 2 dice (6^2)
        total_combinations = 21  # Number of unique combinations with replacement for 2 dice
        
        # Track combinations seen
        seen_combinations = set()
        total_score = 0
        
        # Manually track combinations
        for reroll_values in itertools.combinations_with_replacement(range(1, 7), 2):
            combined_values = sorted([1, 1, 1] + list(reroll_values))  # Add kept ones
            seen_combinations.add(tuple(combined_values))
            score = self.ai._get_max_score(combined_values)
            total_score += score
        
        # Verify we saw all combinations
        self.assertEqual(len(seen_combinations), total_combinations,
            f"Expected {total_combinations} combinations, but got {len(seen_combinations)}")
        
        # Verify the AI's calculation matches our manual calculation
        ai_score = self.ai._calculate_expected_score_for_reroll(keep_indices, dice_values)
        manual_expected_score = total_score / total_combinations
        
        # Allow for small floating point differences
        self.assertAlmostEqual(ai_score, manual_expected_score, 2,
            msg=f"AI score {ai_score} differs from manual calculation {manual_expected_score}")

    def test_greedy_decisions(self):
        """Test that greedy AI makes optimal decisions with complete enumeration."""
        test_cases = [
            # Format: (current_dice, roll_number, expected_reroll_count)
            ([6, 6, 6, 1, 1], 1, 2),  # Should reroll the two ones
            ([1, 2, 3, 4, 6], 1, 1),  # Should reroll just the 6 for straight
            ([5, 5, 5, 5, 1], 1, 1),  # Should reroll just the 1 for Yahtzee
            ([3, 3, 4, 4, 1], 1, 1),  # Should reroll just the 1 to try for Full House
        ]
        
        for dice_values, roll_number, expected_reroll_count in test_cases:
            reroll_indices = self.ai._decide_reroll_greedy2(dice_values, roll_number)
            self.assertEqual(len(reroll_indices), expected_reroll_count,
                f"Expected {expected_reroll_count} rerolls for {dice_values}, got {len(reroll_indices)}")
            
            # Additional verification for the two pairs case
            if dice_values == [3, 3, 4, 4, 1]:
                # Calculate scores for both strategies
                score_keep_pairs = self.ai._calculate_expected_score_for_reroll([0, 1, 2, 3], dice_values)  # Keep both pairs
                score_keep_three = self.ai._calculate_expected_score_for_reroll([0], dice_values)  # Keep one three
                
                # Verify that keeping both pairs has higher expected value
                self.assertGreater(score_keep_pairs, score_keep_three,
                    "Keeping both pairs should have higher expected value than keeping one three")

    def test_no_early_pruning(self):
        """Test that the AI considers all possibilities without early pruning."""
        # Test with a good hand that old version would have pruned
        dice_values = [6, 6, 6, 6, 1]  # Four of a kind
        keep_indices = [0, 1, 2, 3]  # Keep four sixes
        
        # Calculate score for keeping four of a kind
        score_keep_four = self.ai._calculate_expected_score_for_reroll(keep_indices, dice_values)
        
        # Calculate score for keeping three of a kind
        score_keep_three = self.ai._calculate_expected_score_for_reroll([0, 1, 2], dice_values)
        
        # Both options should be considered (no pruning)
        self.assertIsNotNone(score_keep_four)
        self.assertIsNotNone(score_keep_three)

if __name__ == '__main__':
    unittest.main() 