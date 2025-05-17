import unittest
from game_logic import ScoreSheet, Dice # Assuming Dice might be needed for integration tests later, or for getting values
from game_logic import ONES, TWOS, THREES, FOURS, FIVES, SIXES, THREE_OF_A_KIND, FOUR_OF_A_KIND, FULL_HOUSE, SMALL_STRAIGHT, LARGE_STRAIGHT, YAHTZEE, CHANCE

class TestScoreSheetCalculations(unittest.TestCase):

    def setUp(self):
        """Set up a new ScoreSheet before each test."""
        self.sheet = ScoreSheet()

    # --- Test Upper Section --- 
    def test_score_ones_basic(self):
        self.assertEqual(self.sheet.get_potential_score(ONES, [1, 1, 2, 3, 1]), 3, "Score for Ones should be sum of ones")
        self.assertEqual(self.sheet.get_potential_score(ONES, [2, 3, 4, 5, 6]), 0, "Score for Ones should be 0 if no ones")

    def test_score_sixes_basic(self):
        self.assertEqual(self.sheet.get_potential_score(SIXES, [6, 6, 2, 3, 6]), 18, "Score for Sixes should be sum of sixes")
        self.assertEqual(self.sheet.get_potential_score(SIXES, [1, 2, 3, 4, 5]), 0, "Score for Sixes should be 0 if no sixes")

    # --- Test Lower Section --- 
    def test_score_chance(self):
        self.assertEqual(self.sheet.get_potential_score(CHANCE, [1, 2, 3, 4, 5]), 15, "Chance should be sum of all dice")
        self.assertEqual(self.sheet.get_potential_score(CHANCE, [6, 6, 6, 6, 6]), 30, "Chance should be sum of all dice for a Yahtzee")

    def test_score_yahtzee_positive(self):
        self.assertEqual(self.sheet.get_potential_score(YAHTZEE, [5, 5, 5, 5, 5]), 50, "Five of a kind should score 50 for Yahtzee")
        self.assertEqual(self.sheet.get_potential_score(YAHTZEE, [1, 1, 1, 1, 1]), 50, "Five of a kind should score 50 for Yahtzee")

    def test_score_yahtzee_negative(self):
        self.assertEqual(self.sheet.get_potential_score(YAHTZEE, [1, 1, 1, 1, 2]), 0, "Not five of a kind should score 0 for Yahtzee")
        self.assertEqual(self.sheet.get_potential_score(YAHTZEE, [1, 2, 3, 4, 5]), 0, "A straight should score 0 for Yahtzee")

    def test_score_three_of_a_kind(self):
        self.assertEqual(self.sheet.get_potential_score(THREE_OF_A_KIND, [3, 3, 3, 4, 5]), 18, "3 of a kind should sum all dice") # 3+3+3+4+5 = 18
        self.assertEqual(self.sheet.get_potential_score(THREE_OF_A_KIND, [2, 2, 4, 4, 5]), 0, "Less than 3 of a kind should score 0")
        self.assertEqual(self.sheet.get_potential_score(THREE_OF_A_KIND, [4, 4, 4, 4, 5]), 21, "4 of a kind also counts for 3 of a kind") # 4+4+4+4+5 = 21
        self.assertEqual(self.sheet.get_potential_score(THREE_OF_A_KIND, [6, 6, 6, 6, 6]), 30, "Yahtzee also counts for 3 of a kind")

    def test_score_four_of_a_kind(self):
        self.assertEqual(self.sheet.get_potential_score(FOUR_OF_A_KIND, [4, 4, 4, 4, 5]), 21, "4 of a kind should sum all dice") # 4+4+4+4+5 = 21
        self.assertEqual(self.sheet.get_potential_score(FOUR_OF_A_KIND, [3, 3, 3, 4, 5]), 0, "Less than 4 of a kind should score 0")
        self.assertEqual(self.sheet.get_potential_score(FOUR_OF_A_KIND, [6, 6, 6, 6, 6]), 30, "Yahtzee also counts for 4 of a kind")

    def test_score_full_house(self):
        self.assertEqual(self.sheet.get_potential_score(FULL_HOUSE, [2, 2, 3, 3, 3]), 25, "Full House should score 25")
        self.assertEqual(self.sheet.get_potential_score(FULL_HOUSE, [5, 5, 5, 2, 2]), 25, "Full House (reversed) should score 25")
        self.assertEqual(self.sheet.get_potential_score(FULL_HOUSE, [2, 2, 3, 4, 5]), 0, "Not a Full House should score 0")
        self.assertEqual(self.sheet.get_potential_score(FULL_HOUSE, [2, 2, 2, 2, 3]), 0, "Four of a kind is not a Full House by this rule")
        self.assertEqual(self.sheet.get_potential_score(FULL_HOUSE, [3, 3, 3, 3, 3]), 0, "Yahtzee is not a Full House by this specific rule (player can choose though)")
        # Note: A Yahtzee can be *scored* as a Full House if player chooses, but _calculate_full_house is specific.

    def test_score_small_straight(self):
        self.assertEqual(self.sheet.get_potential_score(SMALL_STRAIGHT, [1, 2, 3, 4, 6]), 30, "Sequence of 4 (1234) is Small Straight")
        self.assertEqual(self.sheet.get_potential_score(SMALL_STRAIGHT, [6, 2, 3, 4, 5]), 30, "Sequence of 4 (2345) is Small Straight (unordered)")
        self.assertEqual(self.sheet.get_potential_score(SMALL_STRAIGHT, [1, 1, 2, 3, 4]), 30, "Sequence of 4 (1234) with duplicate is Small Straight")
        self.assertEqual(self.sheet.get_potential_score(SMALL_STRAIGHT, [1, 2, 2, 3, 4]), 30, "Sequence of 4 (1234) with internal duplicate is Small Straight")
        self.assertEqual(self.sheet.get_potential_score(SMALL_STRAIGHT, [1, 2, 3, 5, 6]), 0, "Not a sequence of 4")
        self.assertEqual(self.sheet.get_potential_score(SMALL_STRAIGHT, [1, 2, 3, 4, 5]), 30, "Large Straight also contains a Small Straight") # Our specific rule gives 30

    def test_score_large_straight(self):
        self.assertEqual(self.sheet.get_potential_score(LARGE_STRAIGHT, [1, 2, 3, 4, 5]), 40, "Sequence of 5 (12345) is Large Straight")
        self.assertEqual(self.sheet.get_potential_score(LARGE_STRAIGHT, [6, 2, 3, 4, 5]), 40, "Sequence of 5 (23456) is Large Straight (unordered)")
        self.assertEqual(self.sheet.get_potential_score(LARGE_STRAIGHT, [1, 1, 2, 3, 4]), 0, "Not a sequence of 5 (Small Straight)")
        self.assertEqual(self.sheet.get_potential_score(LARGE_STRAIGHT, [1, 2, 3, 4, 6]), 0, "Not a sequence of 5")

    # --- Test Bonuses --- 
    def test_upper_section_bonus(self):
        self.sheet.record_score(ONES, [1, 1, 1, 1, 1])    # 5
        self.sheet.record_score(TWOS, [2, 2, 2, 1, 1])    # 6
        self.sheet.record_score(THREES, [3, 3, 3, 1, 1])  # 9
        self.sheet.record_score(FOURS, [4, 4, 4, 1, 1])   # 12
        self.sheet.record_score(FIVES, [5, 5, 5, 1, 1])   # 15
        # Subtotal = 5+6+9+12+15 = 47. No bonus yet.
        self.assertEqual(self.sheet.get_upper_section_bonus(), 0, "Bonus should be 0 if subtotal < 63")
        
        self.sheet.record_score(SIXES, [6, 6, 6, 1, 1])   # 18. Subtotal = 47 + 18 = 65.
        self.assertEqual(self.sheet.get_upper_section_bonus(), 35, "Bonus should be 35 if subtotal >= 63")

    def test_yahtzee_bonus(self):
        # First Yahtzee scored in Yahtzee box
        self.sheet.record_score(YAHTZEE, [5, 5, 5, 5, 5]) # Scores 50, no bonus yet from this action itself
        self.assertEqual(self.sheet.scores[YAHTZEE], 50)
        self.assertEqual(self.sheet.yahtzee_bonus_score, 0, "No Yahtzee bonus on first Yahtzee score")

        # Second Yahtzee, scored elsewhere (e.g., Fives)
        self.sheet.record_score(FIVES, [5, 5, 5, 5, 5]) # Scores 25 for Fives, should add 100 bonus
        self.assertEqual(self.sheet.scores[FIVES], 25)
        self.assertEqual(self.sheet.yahtzee_bonus_score, 100, "100 point Yahtzee bonus for second Yahtzee scored elsewhere")

        # Third Yahtzee, scored as Chance
        self.sheet.record_score(CHANCE, [2, 2, 2, 2, 2]) # Scores 10 for Chance, should add another 100 bonus
        self.assertEqual(self.sheet.scores[CHANCE], 10)
        self.assertEqual(self.sheet.yahtzee_bonus_score, 200, "Additional 100 point Yahtzee bonus for third Yahtzee")

    def test_yahtzee_bonus_not_awarded_if_yahtzee_box_is_zero(self):
        self.sheet.record_score(YAHTZEE, [1, 2, 3, 4, 5]) # Scored 0 in Yahtzee
        self.assertEqual(self.sheet.scores[YAHTZEE], 0)
        
        # Roll another Yahtzee
        self.sheet.record_score(FIVES, [5, 5, 5, 5, 5]) # Score 25 for Fives
        self.assertEqual(self.sheet.yahtzee_bonus_score, 0, "No Yahtzee bonus if Yahtzee box scored 0")

    def test_yahtzee_bonus_not_awarded_if_yahtzee_box_not_scored_first(self):
        # Score a Yahtzee (e.g. [5,5,5,5,5]) in FIVES first
        self.sheet.record_score(FIVES, [5,5,5,5,5]) # Score 25
        self.assertEqual(self.sheet.yahtzee_bonus_score, 0, "No bonus if Yahtzee box isn't 50 yet")

        # Then score the Yahtzee box with something else (or even a Yahtzee)
        self.sheet.record_score(YAHTZEE, [1,1,1,1,1]) # Score 50
        self.assertEqual(self.sheet.yahtzee_bonus_score, 0, "Still no bonus from past events, only future ones")
        
        # Now a third Yahtzee roll, scored elsewhere
        self.sheet.record_score(SIXES, [6,6,6,6,6]) # Score 30
        self.assertEqual(self.sheet.yahtzee_bonus_score, 100, "Bonus applies for Yahtzees after Yahtzee box is 50")

    # Example of a test that might fail or need rule clarification for _calculate_small_straight
    def test_small_straight_edge_case_large_straight_is_also_small(self):
        # Standard rules often state a Large Straight also fulfills Small Straight if chosen.
        # My _calculate_small_straight specifically looks for sequences of 4. A LS has two such sequences.
        # My current _calculate_small_straight returns 30 if any sequence of 4 is found.
        self.assertEqual(self.sheet.get_potential_score(SMALL_STRAIGHT, [1, 2, 3, 4, 5]), 30, "Large Straight should qualify for Small Straight score")
        self.assertEqual(self.sheet.get_potential_score(SMALL_STRAIGHT, [2, 3, 4, 5, 6]), 30, "Large Straight should qualify for Small Straight score")

if __name__ == '__main__':
    unittest.main() 