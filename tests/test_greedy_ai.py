import unittest
from yahtzee_ai import YahtzeeAI
from game_logic import (
    ScoreSheet, Dice,
    ONES, TWOS, THREES, FOURS, FIVES, SIXES,
    THREE_OF_A_KIND, FOUR_OF_A_KIND, FULL_HOUSE,
    SMALL_STRAIGHT, LARGE_STRAIGHT, YAHTZEE, CHANCE
)
import random

class TestGreedyAI(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        self.dice = Dice()
        self.scoresheet = ScoreSheet()
        
        # Initialize all AIs
        self.greedy1 = YahtzeeAI("greedy1")
        self.greedy2 = YahtzeeAI("greedy2")
        self.greedy3 = YahtzeeAI("greedy3")
        self.random_ai = YahtzeeAI("random")
        
        # Set game state for all AIs
        self.greedy1.set_game_state(self.scoresheet, self.dice)
        self.greedy2.set_game_state(self.scoresheet, self.dice)
        self.greedy3.set_game_state(self.scoresheet, self.dice)
        self.random_ai.set_game_state(self.scoresheet, self.dice)

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

    def test_random_ai_reroll_distribution(self):
        """Test that random AI's reroll decisions follow expected distribution."""
        dice_values = [1, 2, 3, 4, 5]
        num_trials = 1000
        reroll_counts = [0] * 5  # Count how often each die is rerolled
        
        # Set random seed for reproducibility
        random.seed(42)
        
        # Run multiple trials
        for _ in range(num_trials):
            reroll = self.random_ai.decide_reroll(dice_values, 1)
            for i in reroll:
                reroll_counts[i] += 1
        
        # Check that each die is rerolled roughly 50% of the time
        for i, count in enumerate(reroll_counts):
            proportion = count / num_trials
            self.assertGreater(proportion, 0.45,  # Allow some variance
                             f"Die {i} was rerolled too infrequently: {proportion:.2f}")
            self.assertLess(proportion, 0.55,  # Allow some variance
                          f"Die {i} was rerolled too frequently: {proportion:.2f}")

    def test_random_ai_respects_roll_limit(self):
        """Test that random AI respects the roll limit."""
        dice_values = [1, 2, 3, 4, 5]
        
        # First two rolls should potentially reroll some dice
        reroll1 = self.random_ai.decide_reroll(dice_values, 1)
        self.assertIsInstance(reroll1, list, "Reroll decision should be a list")
        
        reroll2 = self.random_ai.decide_reroll(dice_values, 2)
        self.assertIsInstance(reroll2, list, "Reroll decision should be a list")
        
        # Third roll should return empty list
        reroll3 = self.random_ai.decide_reroll(dice_values, 3)
        self.assertEqual(reroll3, [], "Random AI should not reroll on third roll")

    def test_random_ai_category_selection(self):
        """Test that random AI chooses from available categories."""
        dice_values = [1, 1, 1, 1, 1]  # Yahtzee of ones
        
        # Run multiple trials
        num_trials = 100
        chosen_categories = set()
        
        # Set random seed for reproducibility
        random.seed(42)
        
        for _ in range(num_trials):
            category = self.random_ai.choose_category(dice_values)
            chosen_categories.add(category)
            
            # Reset scoresheet for next trial
            self.random_ai.scoresheet = ScoreSheet()
        
        # Check that multiple different categories were chosen
        self.assertGreater(len(chosen_categories), 1,
                          "Random AI should choose different categories over multiple trials")
        
        # Verify all chosen categories were valid
        available_categories = self.scoresheet.get_available_categories()
        for category in chosen_categories:
            self.assertIn(category, available_categories,
                         f"Chosen category {category} was not in available categories")

if __name__ == '__main__':
    unittest.main() 