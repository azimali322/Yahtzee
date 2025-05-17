import unittest
from yahtzee_ai import YahtzeeAI
from game_logic import (
    ScoreSheet, Dice,
    ONES, TWOS, THREES, FOURS, FIVES, SIXES,
    THREE_OF_A_KIND, FOUR_OF_A_KIND, FULL_HOUSE,
    SMALL_STRAIGHT, LARGE_STRAIGHT, YAHTZEE, CHANCE
)

class TestGreedyAI(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        self.dice = Dice()
        self.scoresheet = ScoreSheet()
        
        # Initialize all three greedy AIs
        self.greedy1 = YahtzeeAI("greedy1")
        self.greedy2 = YahtzeeAI("greedy2")
        self.greedy3 = YahtzeeAI("greedy3")
        
        # Set game state for all AIs
        self.greedy1.set_game_state(self.scoresheet, self.dice)
        self.greedy2.set_game_state(self.scoresheet, self.dice)
        self.greedy3.set_game_state(self.scoresheet, self.dice)

    def test_greedy1_never_rerolls(self):
        """Test that greedy1 AI never rerolls dice."""
        test_cases = [
            [1, 1, 1, 1, 1],  # Yahtzee
            [1, 2, 3, 4, 5],  # Large straight
            [2, 2, 3, 3, 4],  # Two pair
            [1, 2, 2, 3, 4],  # Random dice
        ]
        
        for dice_values in test_cases:
            for roll_number in range(1, 4):
                reroll = self.greedy1.decide_reroll(dice_values, roll_number)
                self.assertEqual(reroll, [], 
                               f"Greedy1 should never reroll, but tried to reroll {reroll}")

    def test_greedy2_respects_roll_limit(self):
        """Test that greedy2 AI only rerolls once."""
        dice_values = [1, 2, 3, 4, 5]
        
        # First roll - should consider rerolling
        reroll1 = self.greedy2.decide_reroll(dice_values, 1)
        self.assertIsInstance(reroll1, list, "Reroll decision should be a list")
        
        # Second roll - should not reroll
        reroll2 = self.greedy2.decide_reroll(dice_values, 2)
        self.assertEqual(reroll2, [], "Greedy2 should not reroll on second roll")
        
        # Third roll - should not reroll
        reroll3 = self.greedy2.decide_reroll(dice_values, 3)
        self.assertEqual(reroll3, [], "Greedy2 should not reroll on third roll")

    def test_greedy3_respects_roll_limit(self):
        """Test that greedy3 AI only rerolls twice."""
        dice_values = [1, 2, 3, 4, 5]
        
        # First roll - should consider rerolling
        reroll1 = self.greedy3.decide_reroll(dice_values, 1)
        self.assertIsInstance(reroll1, list, "Reroll decision should be a list")
        
        # Second roll - should consider rerolling
        reroll2 = self.greedy3.decide_reroll(dice_values, 2)
        self.assertIsInstance(reroll2, list, "Reroll decision should be a list")
        
        # Third roll - should not reroll
        reroll3 = self.greedy3.decide_reroll(dice_values, 3)
        self.assertEqual(reroll3, [], "Greedy3 should not reroll on third roll")

    def test_greedy_category_selection(self):
        """Test that all greedy AIs choose the highest scoring category."""
        test_cases = [
            # Yahtzee of ones
            ([1, 1, 1, 1, 1], YAHTZEE),
            # Large straight
            ([1, 2, 3, 4, 5], LARGE_STRAIGHT),
            # Full house
            ([2, 2, 2, 3, 3], FULL_HOUSE),
            # Four of a kind
            ([4, 4, 4, 4, 2], FOUR_OF_A_KIND),
            # Three of a kind
            ([3, 3, 3, 1, 2], THREE_OF_A_KIND),
            # High straight
            ([2, 3, 4, 5, 6], LARGE_STRAIGHT),
        ]
        
        for dice_values, expected_category in test_cases:
            # All greedy levels should choose the same category for the same dice
            for ai in [self.greedy1, self.greedy2, self.greedy3]:
                category = ai._choose_category_greedy(dice_values)
                self.assertEqual(category, expected_category,
                               f"AI chose {category} instead of {expected_category} for {dice_values}")

    def test_greedy2_keeps_good_combinations(self):
        """Test that greedy2 AI keeps good dice combinations."""
        test_cases = [
            # Keep Yahtzee
            ([5, 5, 5, 5, 5], []),
            # Keep four of a kind
            ([4, 4, 4, 4, 2], [4]),
            # Keep full house
            ([3, 3, 3, 2, 2], []),
            # Keep large straight
            ([1, 2, 3, 4, 5], []),
        ]
        
        for dice_values, expected_reroll in test_cases:
            reroll = self.greedy2.decide_reroll(dice_values, 1)
            self.assertEqual(reroll, expected_reroll,
                           f"Greedy2 should keep good combinations, got {reroll} for {dice_values}")

    def test_expected_score_calculation(self):
        """Test that expected score calculation is working correctly."""
        # Test case: keeping a pair of sixes
        dice_values = [6, 6, 1, 2, 3]
        keep_indices = [0, 1]  # Keeping the sixes
        
        expected_score = self.greedy2._calculate_expected_score_for_reroll(keep_indices, dice_values)
        self.assertGreater(expected_score, 12,  # Should be better than just scoring the sixes
                          "Expected score for keeping pair of sixes should be > 12")

if __name__ == '__main__':
    unittest.main() 