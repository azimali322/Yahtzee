import unittest
from yahtzee_ai import Greedy4AI
from game_logic import Dice, ALL_CATEGORIES
import numpy as np

class TestGreedy4AI(unittest.TestCase):
    def setUp(self):
        self.ai = Greedy4AI()
        
    def test_category_averages_initialization(self):
        """Test that category averages are properly computed."""
        # Check that all categories have an average score
        for category in ALL_CATEGORIES:
            self.assertIn(category, self.ai.category_averages)
            self.assertIsInstance(self.ai.category_averages[category], float)
            
        # Check some known average scores
        # Yahtzee should be relatively rare, so average should be low
        self.assertLess(self.ai.category_averages['YAHTZEE'], 10)
        
        # Chance should average around (1+2+3+4+5+6)/6 * 5 = 17.5
        self.assertGreater(self.ai.category_averages['CHANCE'], 15)
        self.assertLess(self.ai.category_averages['CHANCE'], 20)
        
        # Sixes should average around 10 (probability of rolling a 6 is 1/6)
        self.assertGreater(self.ai.category_averages['SIXES'], 8)
        self.assertLess(self.ai.category_averages['SIXES'], 12)
    
    def test_calculate_category_score(self):
        """Test score calculation for different categories."""
        test_cases = [
            # (dice_values, category, expected_score)
            ([1, 1, 1, 1, 1], 'YAHTZEE', 50),
            ([6, 6, 6, 6, 6], 'SIXES', 30),
            ([1, 2, 3, 4, 5], 'LARGE_STRAIGHT', 40),
            ([1, 2, 3, 4, 6], 'SMALL_STRAIGHT', 30),
            ([2, 2, 2, 3, 3], 'FULL_HOUSE', 25),
            ([4, 4, 4, 4, 1], 'FOUR_OF_A_KIND', 17),
            ([3, 3, 3, 1, 2], 'THREE_OF_A_KIND', 12),
            ([1, 2, 3, 4, 5], 'CHANCE', 15)
        ]
        
        for dice_values, category, expected in test_cases:
            score = self.ai._calculate_category_score(category, dice_values)
            self.assertEqual(score, expected, 
                f"Failed for {category} with dice {dice_values}. "
                f"Expected {expected}, got {score}")
    
    def test_expected_score_calculation(self):
        """Test expected score calculation for different scenarios."""
        # Test keeping three of a kind
        kept_dice = [6, 6, 6]
        num_remaining = 2
        expected_score = self.ai._calculate_expected_score('THREE_OF_A_KIND', kept_dice, num_remaining)
        self.assertGreater(expected_score, 15)  # Should be guaranteed three of a kind
        
        # Test potential Yahtzee
        kept_dice = [6, 6, 6, 6]
        num_remaining = 1
        expected_score = self.ai._calculate_expected_score('YAHTZEE', kept_dice, num_remaining)
        self.assertGreater(expected_score, 8)  # 1/6 chance of Yahtzee
        
        # Test potential straight
        kept_dice = [1, 2, 3, 4]
        num_remaining = 1
        expected_score = self.ai._calculate_expected_score('LARGE_STRAIGHT', kept_dice, num_remaining)
        self.assertGreater(expected_score, 6)  # Good chance of straight
    
    def test_get_action_first_roll(self):
        """Test action selection on first roll."""
        dice = Dice()
        dice.values = [6, 6, 6, 1, 1]  # Three sixes and two ones
        dice.roll_count = 1
        
        keep_action, category = self.ai.get_action(dice)
        
        # Should keep the three sixes
        self.assertEqual(len(keep_action), 5)
        self.assertTrue(all(keep_action[i] for i in range(3)))  # Keep first three dice
        self.assertIsNone(category)  # Shouldn't choose category on first roll
    
    def test_get_action_last_roll(self):
        """Test action selection on last roll."""
        dice = Dice()
        dice.values = [6, 6, 6, 1, 1]
        dice.roll_count = 3
        
        keep_action, category = self.ai.get_action(dice)
        
        # Should keep all dice and choose a category
        self.assertEqual(keep_action, [True] * 5)
        self.assertIsNotNone(category)
        self.assertIn(category, ALL_CATEGORIES)
    
    def test_advantage_calculation(self):
        """Test that the AI correctly calculates advantages."""
        dice = Dice()
        dice.values = [6, 6, 6, 6, 1]  # Four sixes
        dice.roll_count = 1
        
        # Get action to trigger advantage calculation
        keep_action, _ = self.ai.get_action(dice)
        
        # Should keep the four sixes
        expected_keep = [True, True, True, True, False]
        self.assertEqual(keep_action, expected_keep)
    
    def test_update_score(self):
        """Test that scores are properly updated."""
        category = 'SIXES'
        score = 24
        
        self.ai.update_score(category, score)
        self.assertEqual(self.ai.scoresheet.scores[category], score)
        
    def test_complete_game_simulation(self):
        """Test that the AI can complete a full game."""
        dice = Dice()
        completed_categories = set()
        
        # Simulate 13 turns (complete game)
        for _ in range(13):
            dice.reset_roll_count()
            
            # First roll
            dice.roll_all()
            keep_action, category = self.ai.get_action(dice)
            
            if category is None:
                # Second roll
                reroll_indices = [i for i, keep in enumerate(keep_action) if not keep]
                if reroll_indices:
                    dice.roll_specific(reroll_indices)
                keep_action, category = self.ai.get_action(dice)
                
                if category is None:
                    # Third roll
                    reroll_indices = [i for i, keep in enumerate(keep_action) if not keep]
                    if reroll_indices:
                        dice.roll_specific(reroll_indices)
                    keep_action, category = self.ai.get_action(dice)
            
            # Verify category selection
            self.assertIsNotNone(category)
            self.assertNotIn(category, completed_categories)
            
            # Update score
            score = self.ai._calculate_category_score(category, dice.get_values())
            self.ai.update_score(category, score)
            completed_categories.add(category)
        
        # Verify all categories were used
        self.assertEqual(len(completed_categories), 13)
        self.assertEqual(completed_categories, set(ALL_CATEGORIES))

if __name__ == '__main__':
    unittest.main() 