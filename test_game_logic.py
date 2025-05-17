import unittest
import io # Added for suppressing print output
from contextlib import redirect_stdout # Added for suppressing print output
from unittest.mock import patch # Added for mocking random.randint
from game_logic import ScoreSheet, Die, Dice # Added Die
from game_logic import ONES, TWOS, THREES, FOURS, FIVES, SIXES, THREE_OF_A_KIND, FOUR_OF_A_KIND, FULL_HOUSE, SMALL_STRAIGHT, LARGE_STRAIGHT, YAHTZEE, CHANCE

class TestScoreSheetCalculations(unittest.TestCase):

    def setUp(self):
        """Set up a new ScoreSheet before each test."""
        self.sheet = ScoreSheet()

    # --- Test Upper Section --- 
    def test_score_ones_basic(self):
        self.assertEqual(self.sheet.get_potential_score(ONES, [1, 1, 2, 3, 1]), 3, "Score for Ones should be sum of ones")
        self.assertEqual(self.sheet.get_potential_score(ONES, [2, 3, 4, 5, 6]), 0, "Score for Ones should be 0 if no ones")

    def test_score_twos_basic(self):
        self.assertEqual(self.sheet.get_potential_score(TWOS, [2, 2, 1, 3, 2]), 6, "Score for Twos should be sum of twos")
        self.assertEqual(self.sheet.get_potential_score(TWOS, [1, 3, 4, 5, 6]), 0, "Score for Twos should be 0 if no twos")

    def test_score_threes_basic(self):
        self.assertEqual(self.sheet.get_potential_score(THREES, [3, 3, 1, 3, 2]), 9, "Score for Threes should be sum of threes")
        self.assertEqual(self.sheet.get_potential_score(THREES, [1, 2, 4, 5, 6]), 0, "Score for Threes should be 0 if no threes")

    def test_score_fours_basic(self):
        self.assertEqual(self.sheet.get_potential_score(FOURS, [4, 4, 1, 3, 4]), 12, "Score for Fours should be sum of fours")
        self.assertEqual(self.sheet.get_potential_score(FOURS, [1, 2, 3, 5, 6]), 0, "Score for Fours should be 0 if no fours")

    def test_score_fives_basic(self):
        self.assertEqual(self.sheet.get_potential_score(FIVES, [5, 5, 1, 3, 5]), 15, "Score for Fives should be sum of fives")
        self.assertEqual(self.sheet.get_potential_score(FIVES, [1, 2, 3, 4, 6]), 0, "Score for Fives should be 0 if no fives")

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
        self.assertEqual(self.sheet.get_potential_score(THREE_OF_A_KIND, [1, 2, 3, 4, 5]), 0, "Straight is not 3 of a kind")

    def test_score_four_of_a_kind(self):
        self.assertEqual(self.sheet.get_potential_score(FOUR_OF_A_KIND, [4, 4, 4, 4, 5]), 21, "4 of a kind should sum all dice") # 4+4+4+4+5 = 21
        self.assertEqual(self.sheet.get_potential_score(FOUR_OF_A_KIND, [3, 3, 3, 4, 5]), 0, "Less than 4 of a kind should score 0")
        self.assertEqual(self.sheet.get_potential_score(FOUR_OF_A_KIND, [6, 6, 6, 6, 6]), 30, "Yahtzee also counts for 4 of a kind")
        self.assertEqual(self.sheet.get_potential_score(FOUR_OF_A_KIND, [1, 2, 3, 4, 5]), 0, "Straight is not 4 of a kind")

    def test_score_full_house(self):
        self.assertEqual(self.sheet.get_potential_score(FULL_HOUSE, [2, 2, 3, 3, 3]), 25, "Full House should score 25")
        self.assertEqual(self.sheet.get_potential_score(FULL_HOUSE, [5, 5, 5, 2, 2]), 25, "Full House (reversed) should score 25")
        self.assertEqual(self.sheet.get_potential_score(FULL_HOUSE, [1, 1, 6, 6, 6]), 25, "Full House with 1s and 6s")
        self.assertEqual(self.sheet.get_potential_score(FULL_HOUSE, [2, 2, 3, 4, 5]), 0, "Not a Full House should score 0")
        self.assertEqual(self.sheet.get_potential_score(FULL_HOUSE, [2, 2, 2, 2, 3]), 0, "Four of a kind is not a Full House by this rule")
        self.assertEqual(self.sheet.get_potential_score(FULL_HOUSE, [3, 3, 3, 3, 3]), 0, "Yahtzee is not a Full House by this specific rule (player can choose though)")
        # Note: A Yahtzee can be *scored* as a Full House if player chooses, but _calculate_full_house is specific.
        self.assertEqual(self.sheet.get_potential_score(FULL_HOUSE, [4, 4, 4, 1, 1]), 25, "Full House different values")

    def test_score_small_straight(self):
        self.assertEqual(self.sheet.get_potential_score(SMALL_STRAIGHT, [1, 2, 3, 4, 6]), 30, "Sequence of 4 (1234) is Small Straight")
        self.assertEqual(self.sheet.get_potential_score(SMALL_STRAIGHT, [6, 2, 3, 4, 5]), 30, "Sequence of 4 (2345) is Small Straight (unordered)")
        self.assertEqual(self.sheet.get_potential_score(SMALL_STRAIGHT, [1, 3, 4, 5, 6]), 30, "Sequence of 4 (3456) is Small Straight (unordered)")
        self.assertEqual(self.sheet.get_potential_score(SMALL_STRAIGHT, [1, 1, 2, 3, 4]), 30, "Sequence of 4 (1234) with duplicate is Small Straight")
        self.assertEqual(self.sheet.get_potential_score(SMALL_STRAIGHT, [1, 2, 2, 3, 4]), 30, "Sequence of 4 (1234) with internal duplicate is Small Straight")
        self.assertEqual(self.sheet.get_potential_score(SMALL_STRAIGHT, [1, 2, 3, 3, 4]), 30, "Sequence of 4 (1234) with duplicate 3")
        self.assertEqual(self.sheet.get_potential_score(SMALL_STRAIGHT, [1, 2, 3, 4, 4]), 30, "Sequence of 4 (1234) with duplicate 4")
        self.assertEqual(self.sheet.get_potential_score(SMALL_STRAIGHT, [1, 2, 3, 5, 6]), 0, "Not a sequence of 4")
        self.assertEqual(self.sheet.get_potential_score(SMALL_STRAIGHT, [1, 2, 4, 5, 6]), 0, "No seq of 4 (x,2,4,5,6)")
        self.assertEqual(self.sheet.get_potential_score(SMALL_STRAIGHT, [1, 2, 3, 3, 3]), 0, "Three of a kind, not straight")
        self.assertEqual(self.sheet.get_potential_score(SMALL_STRAIGHT, [1, 1, 1, 2, 3]), 0, "Not enough unique for small straight")
        self.assertEqual(self.sheet.get_potential_score(SMALL_STRAIGHT, [1, 2, 3, 4, 5]), 30, "Large Straight also contains a Small Straight") # Our specific rule gives 30

    def test_score_large_straight(self):
        self.assertEqual(self.sheet.get_potential_score(LARGE_STRAIGHT, [1, 2, 3, 4, 5]), 40, "Sequence of 5 (12345) is Large Straight")
        self.assertEqual(self.sheet.get_potential_score(LARGE_STRAIGHT, [6, 2, 3, 4, 5]), 40, "Sequence of 5 (23456) is Large Straight (unordered)")
        self.assertEqual(self.sheet.get_potential_score(LARGE_STRAIGHT, [1, 1, 2, 3, 4]), 0, "Not a sequence of 5 (Small Straight)")
        self.assertEqual(self.sheet.get_potential_score(LARGE_STRAIGHT, [1, 2, 3, 4, 6]), 0, "Not a sequence of 5")
        self.assertEqual(self.sheet.get_potential_score(LARGE_STRAIGHT, [1, 2, 3, 4, 4]), 0, "Not a sequence of 5 (Small Straight)")

    # --- Test Bonuses --- 
    def test_upper_section_bonus(self):
        with io.StringIO() as buf, redirect_stdout(buf): # Suppress prints
            self.sheet.record_score(ONES, [1, 1, 1, 1, 1])    # 5
            self.sheet.record_score(TWOS, [2, 2, 2, 1, 1])    # 6
            self.sheet.record_score(THREES, [3, 3, 3, 1, 1])  # 9
            self.sheet.record_score(FOURS, [4, 4, 4, 1, 1])   # 12
            self.sheet.record_score(FIVES, [5, 5, 5, 1, 1])   # 15
        self.assertEqual(self.sheet.get_upper_section_bonus(), 0, "Bonus should be 0 if subtotal < 63")
        
        with io.StringIO() as buf, redirect_stdout(buf): # Suppress prints
            self.sheet.record_score(SIXES, [6, 6, 6, 1, 1])   # 18. Subtotal = 47 + 18 = 65.
        self.assertEqual(self.sheet.get_upper_section_bonus(), 35, "Bonus should be 35 if subtotal >= 63")

    def test_yahtzee_bonus(self):
        with io.StringIO() as buf, redirect_stdout(buf): # Suppress prints
            self.sheet.record_score(YAHTZEE, [5, 5, 5, 5, 5]) 
        self.assertEqual(self.sheet.scores[YAHTZEE], 50)
        self.assertEqual(self.sheet.yahtzee_bonus_score, 0, "No Yahtzee bonus on first Yahtzee score")

        with io.StringIO() as buf, redirect_stdout(buf): # Suppress prints
            self.sheet.record_score(FIVES, [5, 5, 5, 5, 5]) 
        self.assertEqual(self.sheet.scores[FIVES], 25)
        self.assertEqual(self.sheet.yahtzee_bonus_score, 100, "100 point Yahtzee bonus for second Yahtzee scored elsewhere")

        with io.StringIO() as buf, redirect_stdout(buf): # Suppress prints
            self.sheet.record_score(CHANCE, [2, 2, 2, 2, 2]) 
        self.assertEqual(self.sheet.scores[CHANCE], 10)
        self.assertEqual(self.sheet.yahtzee_bonus_score, 200, "Additional 100 point Yahtzee bonus for third Yahtzee")

    def test_yahtzee_bonus_not_awarded_if_yahtzee_box_is_zero(self):
        with io.StringIO() as buf, redirect_stdout(buf): # Suppress prints
            self.sheet.record_score(YAHTZEE, [1, 2, 3, 4, 5]) 
        self.assertEqual(self.sheet.scores[YAHTZEE], 0)
        
        with io.StringIO() as buf, redirect_stdout(buf): # Suppress prints
            self.sheet.record_score(FIVES, [5, 5, 5, 5, 5]) 
        self.assertEqual(self.sheet.yahtzee_bonus_score, 0, "No Yahtzee bonus if Yahtzee box scored 0")

    def test_yahtzee_bonus_not_awarded_if_yahtzee_box_not_scored_first(self):
        with io.StringIO() as buf, redirect_stdout(buf): # Suppress prints
            self.sheet.record_score(FIVES, [5,5,5,5,5]) 
        self.assertEqual(self.sheet.yahtzee_bonus_score, 0, "No bonus if Yahtzee box isn't 50 yet")

        with io.StringIO() as buf, redirect_stdout(buf): # Suppress prints
            self.sheet.record_score(YAHTZEE, [1,1,1,1,1]) 
        self.assertEqual(self.sheet.yahtzee_bonus_score, 0, "Still no bonus from past events, only future ones")
        
        with io.StringIO() as buf, redirect_stdout(buf): # Suppress prints
            self.sheet.record_score(SIXES, [6,6,6,6,6]) 
        self.assertEqual(self.sheet.yahtzee_bonus_score, 100, "Bonus applies for Yahtzees after Yahtzee box is 50")

    def test_small_straight_edge_case_large_straight_is_also_small(self):
        self.assertEqual(self.sheet.get_potential_score(SMALL_STRAIGHT, [1, 2, 3, 4, 5]), 30)
        self.assertEqual(self.sheet.get_potential_score(SMALL_STRAIGHT, [2, 3, 4, 5, 6]), 30)

class TestDie(unittest.TestCase):
    @patch('random.randint')
    def test_die_initialization(self, mock_randint):
        mock_randint.return_value = 4
        die = Die()
        self.assertEqual(die.value, 4)
        self.assertFalse(die.is_held)
        mock_randint.assert_called_once_with(1, 6)

    @patch('random.randint')
    def test_die_roll_not_held(self, mock_randint):
        mock_randint.return_value = 3 # Initial roll
        die = Die()
        self.assertEqual(die.value, 3)

        mock_randint.return_value = 5 # Subsequent roll
        die.roll()
        self.assertEqual(die.value, 5)
        self.assertEqual(mock_randint.call_count, 2) # Called for __init__ and roll()

    @patch('random.randint')
    def test_die_roll_held(self, mock_randint):
        mock_randint.return_value = 2
        die = Die()
        die.is_held = True
        original_value = die.value # Should be 2
        
        mock_randint.return_value = 6 # Attempt to change value
        die.roll()
        self.assertEqual(die.value, original_value, "Held die value should not change on roll")
        mock_randint.assert_called_once_with(1,6) # Only called during __init__

    def test_die_str(self):
        die = Die()
        die.value = 5
        self.assertEqual(str(die), "5")

class TestDice(unittest.TestCase):
    def setUp(self):
        # It's often good to have a fixed starting point for dice values in tests
        # We can patch random.randint for the Dice initialization too
        pass

    @patch('random.randint')
    def test_dice_initialization(self, mock_randint):
        # Make all dice initialize to a specific sequence for predictability if needed
        # e.g. mock_randint.side_effect = [1, 2, 3, 4, 5]
        mock_randint.return_value = 1 # All dice will be 1 initially
        dice_set = Dice(num_dice=5)
        self.assertEqual(len(dice_set.dice), 5)
        self.assertTrue(all(d.value == 1 for d in dice_set.dice))
        self.assertEqual(dice_set.roll_count, 0)

    def test_get_values(self):
        dice_set = Dice()
        # Manually set values for deterministic testing if not using patch in setUp
        for i, val in enumerate([1,2,3,4,5]):
            dice_set.dice[i].value = val
        self.assertEqual(dice_set.get_values(), [1,2,3,4,5])

    @patch.object(Die, 'roll') # Patch Die.roll directly for these tests
    def test_roll_all_not_held(self, mock_die_roll):
        dice_set = Dice(num_dice=3)
        for die in dice_set.dice:
            die.is_held = False
        
        dice_set.roll_all()
        self.assertEqual(mock_die_roll.call_count, 3, "Each unheld die should be rolled once")
        self.assertEqual(dice_set.roll_count, 1)

    @patch.object(Die, 'roll')
    def test_roll_all_some_held(self, mock_die_roll):
        dice_set = Dice(num_dice=5)
        dice_set.dice[0].is_held = True
        dice_set.dice[2].is_held = True
        dice_set.dice[4].is_held = True
        # Dice 1 and 3 (0-indexed) are not held

        dice_set.roll_all()
        self.assertEqual(mock_die_roll.call_count, 2, "Only unheld dice should be rolled")
        self.assertEqual(dice_set.roll_count, 1)

    @patch.object(Die, 'roll')
    def test_roll_all_max_rolls(self, mock_die_roll):
        dice_set = Dice(num_dice=2)
        dice_set.roll_all() # Roll 1
        dice_set.roll_all() # Roll 2
        dice_set.roll_all() # Roll 3
        self.assertEqual(dice_set.roll_count, 3)
        mock_die_roll.reset_mock() # Reset call count for Die.roll
        
        # Attempt 4th roll
        captured_output = ""
        with io.StringIO() as buf, redirect_stdout(buf):
            dice_set.roll_all()
            captured_output = buf.getvalue() # Get value before block ends
        self.assertEqual(mock_die_roll.call_count, 0, "No dice should roll after 3rd roll_all")
        self.assertEqual(dice_set.roll_count, 3, "Roll count should remain 3")
        self.assertIn("already rolled 3 times", captured_output)

    @patch.object(Die, 'roll')
    def test_roll_specific_dice(self, mock_die_roll):
        dice_set = Dice(num_dice=5)
        # Hold some dice to ensure roll_specific unholds them before rolling
        dice_set.dice[0].is_held = True
        dice_set.dice[1].is_held = True

        dice_set.roll_specific([0, 2, 4]) # Roll dice at index 0, 2, 4
        self.assertEqual(mock_die_roll.call_count, 3)
        self.assertFalse(dice_set.dice[0].is_held, "Specified die 0 should be unheld after roll_specific")
        self.assertTrue(dice_set.dice[1].is_held, "Unspecified held die 1 should remain held")
        self.assertFalse(dice_set.dice[2].is_held, "Specified die 2 should be unheld")
        self.assertEqual(dice_set.roll_count, 1)

    @patch.object(Die, 'roll')
    def test_roll_specific_max_rolls(self, mock_die_roll):
        dice_set = Dice(num_dice=5)
        dice_set.roll_specific([0])    # Roll 1
        dice_set.roll_specific([1,2])  # Roll 2
        dice_set.roll_specific([3,4])  # Roll 3
        self.assertEqual(dice_set.roll_count, 3)
        mock_die_roll.reset_mock()

        captured_output = ""
        with io.StringIO() as buf, redirect_stdout(buf):
            dice_set.roll_specific([0]) # Attempt 4th roll
            captured_output = buf.getvalue()
        self.assertEqual(mock_die_roll.call_count, 0)
        self.assertEqual(dice_set.roll_count, 3)
        self.assertIn("already rolled 3 times", captured_output)

    def test_toggle_hold(self):
        dice_set = Dice(num_dice=3)
        self.assertFalse(dice_set.dice[1].is_held)
        dice_set.toggle_hold(1)
        self.assertTrue(dice_set.dice[1].is_held)
        dice_set.toggle_hold(1)
        self.assertFalse(dice_set.dice[1].is_held)

    def test_toggle_hold_invalid_index(self):
        dice_set = Dice(num_dice=1)
        captured_output = ""
        with io.StringIO() as buf, redirect_stdout(buf):
            dice_set.toggle_hold(5) # Invalid index
            captured_output = buf.getvalue()
        self.assertIn("Invalid die index 5 for holding", captured_output)

    def test_reset_roll_count(self):
        dice_set = Dice(num_dice=2)
        dice_set.roll_all() # roll_count = 1
        dice_set.dice[0].is_held = True
        
        dice_set.reset_roll_count()
        self.assertEqual(dice_set.roll_count, 0)
        self.assertFalse(dice_set.dice[0].is_held, "Dice should be unheld after reset")
        self.assertFalse(dice_set.dice[1].is_held, "Dice should be unheld after reset")

class TestJokerRules(unittest.TestCase):
    def setUp(self):
        self.sheet = ScoreSheet()

    def test_joker_rule_upper_section_priority(self):
        """Test that Joker rule forces upper section scoring if available."""
        with io.StringIO() as buf, redirect_stdout(buf):
            self.sheet.record_score(YAHTZEE, [5, 5, 5, 5, 5])  # Score first Yahtzee
        
        # Simulate a Yahtzee of 3s when Threes category is open
        result = self.sheet.determine_joker_action([3, 3, 3, 3, 3])
        self.assertEqual(result['type'], 'force_upper')
        self.assertEqual(result['category'], THREES)
        self.assertEqual(result['score'], 15)

    def test_joker_rule_lower_section_options(self):
        """Test that Joker rule offers lower section options when upper section is used."""
        with io.StringIO() as buf, redirect_stdout(buf):
            self.sheet.record_score(YAHTZEE, [5, 5, 5, 5, 5])  # Score first Yahtzee
            self.sheet.record_score(FIVES, [5, 5, 5, 1, 1])    # Use up corresponding upper section
        
        # Simulate another Yahtzee of 5s
        result = self.sheet.determine_joker_action([5, 5, 5, 5, 5])
        self.assertEqual(result['type'], 'choose_lower')
        self.assertTrue(isinstance(result['options'], dict))
        # Check available scores
        if FULL_HOUSE in result['options']:
            self.assertEqual(result['options'][FULL_HOUSE], 25)
        if SMALL_STRAIGHT in result['options']:
            self.assertEqual(result['options'][SMALL_STRAIGHT], 30)
        if LARGE_STRAIGHT in result['options']:
            self.assertEqual(result['options'][LARGE_STRAIGHT], 40)

    def test_joker_rule_force_zero_upper(self):
        """Test that Joker rule forces zero in upper section when all lower sections are used."""
        with io.StringIO() as buf, redirect_stdout(buf):
            # First score Yahtzee
            self.sheet.record_score(YAHTZEE, [5, 5, 5, 5, 5])
            # Use up corresponding upper section
            self.sheet.record_score(FIVES, [5, 5, 5, 1, 1])
            # Fill all lower section categories
            self.sheet.record_score(THREE_OF_A_KIND, [3, 3, 3, 3, 3])
            self.sheet.record_score(FOUR_OF_A_KIND, [4, 4, 4, 4, 4])
            self.sheet.record_score(FULL_HOUSE, [2, 2, 3, 3, 3])
            self.sheet.record_score(SMALL_STRAIGHT, [1, 2, 3, 4, 5])
            self.sheet.record_score(LARGE_STRAIGHT, [2, 3, 4, 5, 6])
            self.sheet.record_score(CHANCE, [6, 6, 6, 6, 6])

        # Simulate another Yahtzee of 5s
        result = self.sheet.determine_joker_action([5, 5, 5, 5, 5])
        self.assertEqual(result['type'], 'force_zero_upper')
        self.assertTrue(isinstance(result['options'], list))
        self.assertEqual(result['score'], 0)

class TestMaximumScores(unittest.TestCase):
    def setUp(self):
        self.sheet = ScoreSheet()

    def test_maximum_score_with_thirteen_yahtzees(self):
        """Test maximum possible score of 1575 with 13 Yahtzees."""
        with io.StringIO() as buf, redirect_stdout(buf):
            # First Yahtzee in Yahtzee box (50 points)
            self.sheet.record_score(YAHTZEE, [6, 6, 6, 6, 6])    # 50 points
            
            # Score lower section first to maximize Yahtzee bonuses
            # When scoring a Yahtzee in other categories, we get the category's Joker score
            self.sheet.record_score(FOUR_OF_A_KIND, [6, 6, 6, 6, 6])  # 30 points + 100 bonus
            self.sheet.record_score(THREE_OF_A_KIND, [6, 6, 6, 6, 6]) # 30 points + 100 bonus
            self.sheet.record_score(FULL_HOUSE, [6, 6, 6, 6, 6])      # 25 points + 100 bonus
            self.sheet.record_score(SMALL_STRAIGHT, [6, 6, 6, 6, 6])  # 30 points + 100 bonus
            self.sheet.record_score(LARGE_STRAIGHT, [6, 6, 6, 6, 6])  # 40 points + 100 bonus
            self.sheet.record_score(CHANCE, [6, 6, 6, 6, 6])          # 30 points + 100 bonus
            
            # Score upper section last
            self.sheet.record_score(SIXES, [6, 6, 6, 6, 6])      # 30 points + 100 bonus
            self.sheet.record_score(FIVES, [5, 5, 5, 5, 5])      # 25 points + 100 bonus
            self.sheet.record_score(FOURS, [4, 4, 4, 4, 4])      # 20 points + 100 bonus
            self.sheet.record_score(THREES, [3, 3, 3, 3, 3])     # 15 points + 100 bonus
            self.sheet.record_score(TWOS, [2, 2, 2, 2, 2])       # 10 points + 100 bonus
            self.sheet.record_score(ONES, [1, 1, 1, 1, 1])       # 5 points + 100 bonus

        # Verify each component of the score
        self.assertEqual(self.sheet.get_upper_section_subtotal(), 105)  # Sum of upper section
        self.assertEqual(self.sheet.get_upper_section_bonus(), 35)      # Bonus for >=63
        self.assertEqual(self.sheet.scores[YAHTZEE], 50)               # Initial Yahtzee
        self.assertEqual(self.sheet.yahtzee_bonus_score, 1200)         # 12 additional Yahtzees * 100
        
        # Lower section scores with Joker rules
        expected_scores = {
            FOUR_OF_A_KIND: 30,    # Sum of dice for [6,6,6,6,6]
            THREE_OF_A_KIND: 30,   # Sum of dice for [6,6,6,6,6]
            FULL_HOUSE: 25,        # Fixed score for Full House
            SMALL_STRAIGHT: 30,    # Fixed score for Small Straight
            LARGE_STRAIGHT: 40,    # Fixed score for Large Straight
            CHANCE: 30,            # Sum of dice for [6,6,6,6,6]
            YAHTZEE: 50           # Initial Yahtzee score
        }
        for category, expected_score in expected_scores.items():
            self.assertEqual(self.sheet.scores[category], expected_score, 
                           f"Wrong score for {category}. Expected {expected_score}, got {self.sheet.scores[category]}")
        
        # Final total: 105 + 35 + 235 + 1200 = 1575
        self.assertEqual(self.sheet.get_grand_total(), 1575)

    def test_maximum_score_without_yahtzee_bonus(self):
        """Test maximum possible score of 300 with exactly one Yahtzee but no bonus."""
        with io.StringIO() as buf, redirect_stdout(buf):
            # Upper section optimal scoring
            self.sheet.record_score(SIXES, [6, 6, 6, 6, 5])    # 24
            self.sheet.record_score(FIVES, [5, 5, 5, 5, 4])    # 20
            self.sheet.record_score(FOURS, [4, 4, 4, 4, 3])    # 16
            self.sheet.record_score(THREES, [3, 3, 3, 3, 2])   # 12
            self.sheet.record_score(TWOS, [2, 2, 2, 2, 1])     # 8
            self.sheet.record_score(ONES, [1, 1, 1, 1, 6])     # 4
            
            # Lower section optimal scoring - with one Yahtzee but no bonuses
            self.sheet.record_score(YAHTZEE, [6, 6, 6, 6, 6])       # 50 (one Yahtzee)
            self.sheet.record_score(FOUR_OF_A_KIND, [6, 6, 6, 6, 5])  # 29
            self.sheet.record_score(THREE_OF_A_KIND, [6, 6, 6, 6, 5]) # 29 (four of a kind counts for three of a kind)
            self.sheet.record_score(FULL_HOUSE, [5, 5, 6, 6, 6])      # 25
            self.sheet.record_score(SMALL_STRAIGHT, [1, 2, 3, 4, 5])  # 30
            self.sheet.record_score(LARGE_STRAIGHT, [2, 3, 4, 5, 6])  # 40
            self.sheet.record_score(CHANCE, [6, 6, 6, 6, 5])          # 29

        # Verify each component of the score
        self.assertEqual(self.sheet.get_upper_section_subtotal(), 84)  # Sum of upper section
        self.assertEqual(self.sheet.get_upper_section_bonus(), 35)     # Bonus for >=63
        self.assertEqual(self.sheet.yahtzee_bonus_score, 0)           # No Yahtzee bonuses
        
        # Lower section scores
        expected_scores = {
            FOUR_OF_A_KIND: 29,    # Sum of [6,6,6,6,5]
            THREE_OF_A_KIND: 29,   # Sum of [6,6,6,6,5] (four of a kind counts for three of a kind)
            FULL_HOUSE: 25,        # Fixed score
            SMALL_STRAIGHT: 30,    # Fixed score
            LARGE_STRAIGHT: 40,    # Fixed score
            CHANCE: 29,            # Sum of [6,6,6,6,5]
            YAHTZEE: 50           # One Yahtzee
        }
        for category, expected_score in expected_scores.items():
            self.assertEqual(self.sheet.scores[category], expected_score,
                           f"Wrong score for {category}. Expected {expected_score}, got {self.sheet.scores[category]}")
        
        # Final total: 84 + 35 + 232 = 351
        self.assertEqual(self.sheet.get_grand_total(), 351)

    def test_maximum_score_without_five_of_a_kind(self):
        """Test maximum possible score of 301 without any five-of-a-kind."""
        with io.StringIO() as buf, redirect_stdout(buf):
            # Upper section optimal scoring (using four-of-a-kind)
            self.sheet.record_score(SIXES, [6, 6, 6, 6, 5])    # 24
            self.sheet.record_score(FIVES, [5, 5, 5, 5, 4])    # 20
            self.sheet.record_score(FOURS, [4, 4, 4, 4, 3])    # 16
            self.sheet.record_score(THREES, [3, 3, 3, 3, 2])   # 12
            self.sheet.record_score(TWOS, [2, 2, 2, 2, 1])     # 8
            self.sheet.record_score(ONES, [1, 1, 1, 1, 6])     # 4
            
            # Lower section optimal scoring - no five-of-a-kind anywhere
            self.sheet.record_score(YAHTZEE, [1, 2, 3, 4, 6])       # 0 (not a Yahtzee)
            self.sheet.record_score(FOUR_OF_A_KIND, [6, 6, 6, 6, 5])  # 29
            self.sheet.record_score(THREE_OF_A_KIND, [6, 6, 6, 6, 5]) # 29 (four of a kind counts for three of a kind)
            self.sheet.record_score(FULL_HOUSE, [6, 6, 6, 5, 5])      # 25
            self.sheet.record_score(SMALL_STRAIGHT, [1, 2, 3, 4, 5])  # 30
            self.sheet.record_score(LARGE_STRAIGHT, [2, 3, 4, 5, 6])  # 40
            self.sheet.record_score(CHANCE, [6, 6, 6, 6, 5])          # 29

        # Verify each component of the score
        self.assertEqual(self.sheet.get_upper_section_subtotal(), 84)  # Sum of upper section
        self.assertEqual(self.sheet.get_upper_section_bonus(), 35)     # Bonus for >=63
        self.assertEqual(self.sheet.yahtzee_bonus_score, 0)           # No Yahtzee bonuses
        
        # Lower section scores
        expected_scores = {
            FOUR_OF_A_KIND: 29,    # Sum of [6,6,6,6,5]
            THREE_OF_A_KIND: 29,   # Sum of [6,6,6,6,5] (four of a kind counts for three of a kind)
            FULL_HOUSE: 25,        # Fixed score
            SMALL_STRAIGHT: 30,    # Fixed score
            LARGE_STRAIGHT: 40,    # Fixed score
            CHANCE: 29,            # Sum of [6,6,6,6,5]
            YAHTZEE: 0            # Not a Yahtzee
        }
        for category, expected_score in expected_scores.items():
            self.assertEqual(self.sheet.scores[category], expected_score,
                           f"Wrong score for {category}. Expected {expected_score}, got {self.sheet.scores[category]}")
        
        # Final total: 84 + 35 + 182 = 301
        self.assertEqual(self.sheet.get_grand_total(), 301)

if __name__ == '__main__':
    unittest.main() 