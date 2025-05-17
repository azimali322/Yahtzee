import random
from collections import Counter

class Die:
    """Represents a single die."""
    def __init__(self):
        self.value = random.randint(1, 6)
        self.is_held = False

    def roll(self):
        """Rolls the die and updates its value if not held."""
        if not self.is_held:
            self.value = random.randint(1, 6)
        return self.value

    def __str__(self):
        return str(self.value)

class Dice:
    """Represents a collection of dice (typically 5 for Yahtzee)."""
    def __init__(self, num_dice=5):
        self.dice = [Die() for _ in range(num_dice)]
        self.roll_count = 0

    def roll_all(self):
        """Rolls all dice that are not held."""
        if self.roll_count < 3:
            for die in self.dice:
                die.roll()
            self.roll_count += 1
        else:
            print("You have already rolled 3 times this turn.")
        return self.get_values()

    def roll_specific(self, indices_to_roll):
        """
        Rolls specific dice based on their indices.
        Indices are 0-based.
        """
        if self.roll_count < 3:
            for i in indices_to_roll:
                if 0 <= i < len(self.dice):
                    self.dice[i].is_held = False # Unhold before rolling
                    self.dice[i].roll()
                else:
                    print(f"Warning: Invalid die index {i} specified.")
            self.roll_count += 1
        else:
            print("You have already rolled 3 times this turn.")
        return self.get_values()

    def get_values(self):
        """Returns a list of the current values of all dice."""
        return [die.value for die in self.dice]

    def toggle_hold(self, index):
        """Toggles the hold status of a die at the given index."""
        if 0 <= index < len(self.dice):
            self.dice[index].is_held = not self.dice[index].is_held
        else:
            print(f"Warning: Invalid die index {index} for holding.")

    def reset_roll_count(self):
        """Resets the roll count for the start of a new turn."""
        self.roll_count = 0
        for die in self.dice:
            die.is_held = False # Unhold all dice for the new turn

    def __str__(self):
        return " ".join(str(die) for die in self.dice)

# Constants for scoring categories
ONES = "Ones"
TWOS = "Twos"
THREES = "Threes"
FOURS = "Fours"
FIVES = "Fives"
SIXES = "Sixes"
THREE_OF_A_KIND = "Three of a Kind"
FOUR_OF_A_KIND = "Four of a Kind"
FULL_HOUSE = "Full House"
SMALL_STRAIGHT = "Small Straight"
LARGE_STRAIGHT = "Large Straight"
YAHTZEE = "Yahtzee"
CHANCE = "Chance"

UPPER_SECTION_CATEGORIES = [ONES, TWOS, THREES, FOURS, FIVES, SIXES]
LOWER_SECTION_CATEGORIES = [THREE_OF_A_KIND, FOUR_OF_A_KIND, FULL_HOUSE, SMALL_STRAIGHT, LARGE_STRAIGHT, YAHTZEE, CHANCE]
ALL_CATEGORIES = UPPER_SECTION_CATEGORIES + LOWER_SECTION_CATEGORIES

class ScoreSheet:
    """Manages the Yahtzee scoresheet, calculations, and recording scores."""
    def __init__(self):
        self.scores = {category: None for category in ALL_CATEGORIES}
        self.yahtzee_bonus_score = 0
        self._custom_score_calculators = {} # For custom category logic

    def _calculate_sum_for_face_value(self, dice_values, face_value):
        """Helper to calculate sum for upper section categories (Ones, Twos, etc.)."""
        return sum(die_value for die_value in dice_values if die_value == face_value)

    def _calculate_n_of_a_kind(self, dice_values, n):
        """Helper to calculate score for N of a Kind (3 or 4)."""
        counts = Counter(dice_values)
        for val, count in counts.items():
            if count >= n:
                return sum(dice_values) # Standard Yahtzee rule: sum of all dice
        return 0

    def _calculate_full_house(self, dice_values):
        """Calculates score for Full House."""
        counts = Counter(dice_values)
        if len(counts) == 2 and (counts.most_common(1)[0][1] == 3 or counts.most_common(1)[0][1] == 2):
             # Check if one value appears 3 times and another 2 times
            if 3 in counts.values() and 2 in counts.values():
                return 25
        return 0

    def _calculate_straight(self, dice_values, required_length):
        """Helper to calculate score for Small or Large Straight."""
        unique_dice = sorted(list(set(dice_values)))
        if len(unique_dice) < required_length:
            return 0

        # Check all possible sub-sequences of required_length
        for i in range(len(unique_dice) - required_length + 1):
            is_straight = True
            for j in range(required_length - 1):
                if unique_dice[i+j+1] - unique_dice[i+j] != 1:
                    is_straight = False
                    break
            if is_straight:
                if required_length == 4: return 30 # Small Straight
                if required_length == 5: return 40 # Large Straight
        return 0
        
    def _calculate_small_straight(self, dice_values):
        """Calculates score for Small Straight (sequence of 4)."""
        # A small straight can be e.g. 1234, 2345, 3456
        # Remove duplicates and sort
        unique_sorted_dice = sorted(list(set(dice_values)))
        
        if len(unique_sorted_dice) < 4:
            return 0

        # Check for 1234, 2345, 3456 by checking all subsequences of length 4
        for i in range(len(unique_sorted_dice) - 3):
            sub_sequence = unique_sorted_dice[i:i+4]
            is_seq = True
            for j in range(3):
                if sub_sequence[j+1] - sub_sequence[j] != 1:
                    is_seq = False
                    break
            if is_seq:
                return 30
        return 0


    def _calculate_large_straight(self, dice_values):
        """Calculates score for Large Straight (sequence of 5)."""
        # A large straight is 12345 or 23456
        unique_sorted_dice = sorted(list(set(dice_values)))
        if len(unique_sorted_dice) < 5:
            return 0
        
        # Check for 12345 or 23456
        if unique_sorted_dice == [1,2,3,4,5] or unique_sorted_dice == [2,3,4,5,6]:
            return 40
        return 0

    def _calculate_yahtzee(self, dice_values):
        """Calculates score for Yahtzee (5 of a kind)."""
        counts = Counter(dice_values)
        if len(counts) == 1 and len(dice_values) == 5 : # All 5 dice are the same
            return 50
        return 0

    def _calculate_chance(self, dice_values):
        """Calculates score for Chance (sum of all dice)."""
        return sum(dice_values)

    def get_potential_score(self, category_name, dice_values):
        """Calculates the potential score for a given category and dice values."""
        if not dice_values or len(dice_values) != 5:
            # print("Warning: Dice values must be a list of 5 integers.")
            return 0 # Or raise error

        if category_name == ONES: return self._calculate_sum_for_face_value(dice_values, 1)
        if category_name == TWOS: return self._calculate_sum_for_face_value(dice_values, 2)
        if category_name == THREES: return self._calculate_sum_for_face_value(dice_values, 3)
        if category_name == FOURS: return self._calculate_sum_for_face_value(dice_values, 4)
        if category_name == FIVES: return self._calculate_sum_for_face_value(dice_values, 5)
        if category_name == SIXES: return self._calculate_sum_for_face_value(dice_values, 6)
        if category_name == THREE_OF_A_KIND: return self._calculate_n_of_a_kind(dice_values, 3)
        if category_name == FOUR_OF_A_KIND: return self._calculate_n_of_a_kind(dice_values, 4)
        if category_name == FULL_HOUSE: return self._calculate_full_house(dice_values)
        if category_name == SMALL_STRAIGHT: return self._calculate_small_straight(dice_values)
        if category_name == LARGE_STRAIGHT: return self._calculate_large_straight(dice_values)
        if category_name == YAHTZEE: return self._calculate_yahtzee(dice_values)
        if category_name == CHANCE: return self._calculate_chance(dice_values)
        
        if category_name in self._custom_score_calculators:
            return self._custom_score_calculators[category_name](dice_values)
            
        # print(f"Warning: Unknown category '{category_name}' for potential score calculation.")
        return 0

    def record_score(self, category_name, dice_values):
        """Records the score for a category if it's not already scored."""
        if category_name not in self.scores:
            print(f"Error: Category '{category_name}' does not exist.")
            return False
        if self.scores[category_name] is not None:
            print(f"Error: Category '{category_name}' has already been scored ({self.scores[category_name]}).")
            return False
        if not dice_values or len(dice_values) != 5:
            print("Error: Must provide 5 dice values to record a score.")
            return False

        score_to_record = self.get_potential_score(category_name, dice_values)
        self.scores[category_name] = score_to_record

        # Yahtzee Bonus Logic:
        # If a Yahtzee is rolled (all 5 dice same), AND the "Yahtzee" box has already been scored with 50,
        # AND the current category being scored is NOT "Yahtzee" itself (unless it's a 0 score for Yahtzee),
        # then a 100 point bonus is awarded.
        # The player must still place the current roll in an available category.
        is_current_roll_yahtzee = len(set(dice_values)) == 1 and len(dice_values) == 5
        
        if is_current_roll_yahtzee and self.scores.get(YAHTZEE) == 50:
            if category_name != YAHTZEE: # Bonus for subsequent Yahtzees not scored in Yahtzee box
                 self.yahtzee_bonus_score += 100
            # If they are scoring YAHTZEE category again (e.g. for a zero if they must), no extra bonus here.
            # The original 50 in YAHTZEE box is what enables this.

        print(f"Score of {score_to_record} recorded for {category_name}.")
        return True

    def add_custom_category(self, category_name, scoring_function, display_name=None):
        """Adds a new custom scoring category."""
        if category_name in self.scores:
            print(f"Warning: Category '{category_name}' already exists.")
            # Potentially allow overwriting or require unique internal name
            return

        # For simplicity, custom categories are added to a general list for now.
        # Could refine to place in upper/lower if needed.
        ALL_CATEGORIES.append(category_name) # Modify global, or make ALL_CATEGORIES instance member
        self.scores[category_name] = None
        self._custom_score_calculators[category_name] = scoring_function
        print(f"Custom category '{display_name or category_name}' added.")

    def get_upper_section_subtotal(self):
        """Calculates the subtotal for the upper section."""
        return sum(self.scores[cat] or 0 for cat in UPPER_SECTION_CATEGORIES)

    def get_upper_section_bonus(self):
        """Returns 35 if upper section subtotal is 63 or more, else 0."""
        return 35 if self.get_upper_section_subtotal() >= 63 else 0

    def get_total_upper_score(self):
        """Calculates the total score for the upper section including bonus."""
        return self.get_upper_section_subtotal() + self.get_upper_section_bonus()

    def get_lower_section_score(self):
        """Calculates the total score for the lower section."""
        # Includes standard lower categories + any custom ones not specifically assigned to upper
        lower_score = sum(self.scores[cat] or 0 for cat in LOWER_SECTION_CATEGORIES)
        
        # Add scores from custom categories that are not in UPPER_SECTION_CATEGORIES
        for cat_name, score in self.scores.items():
            if cat_name not in ALL_CATEGORIES and cat_name not in UPPER_SECTION_CATEGORIES: # Heuristic for custom
                 if score is not None:
                    lower_score += score
        return lower_score


    def get_grand_total(self):
        """Calculates the grand total score."""
        return self.get_total_upper_score() + self.get_lower_section_score() + self.yahtzee_bonus_score

    def get_available_categories(self):
        """Returns a list of categories that have not yet been scored."""
        return [cat for cat, score in self.scores.items() if score is None]

    def is_complete(self):
        """Checks if all categories have been scored."""
        return all(score is not None for score in self.scores.values())

    def display_scoresheet(self):
        """Prints a formatted representation of the scoresheet."""
        print("\n--- Scoresheet ---")
        print("Upper Section:")
        for cat in UPPER_SECTION_CATEGORIES:
            score_display = str(self.scores[cat]) if self.scores[cat] is not None else "-"
            print(f"  {cat:<15}: {score_display:>3}")
        print(f"  {'Upper Subtotal':<15}: {self.get_upper_section_subtotal():>3}")
        print(f"  {'Bonus (>=63)':<15}: {self.get_upper_section_bonus():>3}")
        print(f"  {'Total Upper':<15}: {self.get_total_upper_score():>3}")
        
        print("\nLower Section:")
        for cat in LOWER_SECTION_CATEGORIES:
            score_display = str(self.scores[cat]) if self.scores[cat] is not None else "-"
            print(f"  {cat:<15}: {score_display:>3}")

        # Display custom categories if any
        custom_cats_displayed = False
        for cat_name in self.scores.keys():
            if cat_name not in ALL_CATEGORIES: # A bit of a hack, assumes ALL_CATEGORIES is static std
                if not custom_cats_displayed:
                    print("\nCustom Categories:")
                    custom_cats_displayed = True
                score_display = str(self.scores[cat_name]) if self.scores[cat_name] is not None else "-"
                print(f"  {cat_name:<15}: {score_display:>3}")


        print(f"\n  {'Yahtzee Bonus':<15}: {self.yahtzee_bonus_score:>3}")
        print(f"  {'Total Lower':<15}: {self.get_lower_section_score():>3}") # This total needs to be accurate with custom
        print(f"\n  {'GRAND TOTAL':<15}: {self.get_grand_total():>3}")
        print("------------------\n")

# Example usage (can be removed later or moved to main.py)
if __name__ == "__main__":
    my_dice = Dice()
    print("Initial roll:", my_dice)
    print("Values:", my_dice.get_values())

    my_dice.toggle_hold(0)
    my_dice.toggle_hold(2)
    print(f"Die 0 held: {my_dice.dice[0].is_held}, Die 2 held: {my_dice.dice[2].is_held}")

    print("Rolling unheld dice (1st re-roll):", my_dice.roll_all())
    print("Dice after 1st re-roll:", my_dice)

    # Example of rolling specific dice that are currently held
    # We might want to unhold them first if that's the game logic
    # Or the roll_specific could automatically unhold them
    my_dice.toggle_hold(0) # Unhold die 0
    print("Rolling specific (0, 4) (2nd re-roll):", my_dice.roll_specific([0, 4]))
    print("Dice after 2nd re-roll:", my_dice)


    print("Rolling all (3rd re-roll):", my_dice.roll_all())
    print("Dice after 3rd re-roll:", my_dice)

    # Attempt another roll
    my_dice.roll_all()

    my_dice.reset_roll_count()
    print("After reset:", my_dice, "Roll count:", my_dice.roll_count)
    print("Rolling after reset:", my_dice.roll_all())
    print("Dice:", my_dice)

    print("\n--- ScoreSheet Example ---")
    sheet = ScoreSheet()
    
    # Simulate some dice rolls and scoring
    dice_roll_1 = [1, 1, 2, 3, 4] # Potential Small Straight, Ones, Twos, etc.
    dice_roll_2 = [5, 5, 5, 5, 5] # Yahtzee
    dice_roll_3 = [2, 2, 3, 3, 3] # Full House, Threes, Twos
    dice_roll_4 = [6, 6, 6, 1, 2] # Three of a Kind (Sixes)
    dice_roll_5 = [1, 2, 3, 4, 5] # Large Straight

    print("Initial sheet:")
    sheet.display_scoresheet()

    print(f"Potential for Ones with {dice_roll_1}: {sheet.get_potential_score(ONES, dice_roll_1)}")
    sheet.record_score(ONES, dice_roll_1) # Score 2 for Ones
    
    print(f"Potential for Small Straight with {dice_roll_1}: {sheet.get_potential_score(SMALL_STRAIGHT, dice_roll_1)}")
    # sheet.record_score(SMALL_STRAIGHT, dice_roll_1) # This would fail as ONES is scored with it

    print(f"Potential for Yahtzee with {dice_roll_2}: {sheet.get_potential_score(YAHTZEE, dice_roll_2)}")
    sheet.record_score(YAHTZEE, dice_roll_2) # Score 50 for Yahtzee

    sheet.display_scoresheet()

    # Simulate another Yahtzee roll after Yahtzee is scored with 50
    dice_roll_yahtzee_again = [4, 4, 4, 4, 4]
    print(f"Dice: {dice_roll_yahtzee_again}. Yahtzee box scored: {sheet.scores[YAHTZEE]}")
    # Try to score it as Fours (should get bonus)
    print(f"Potential for Fours with {dice_roll_yahtzee_again}: {sheet.get_potential_score(FOURS, dice_roll_yahtzee_again)}")
    sheet.record_score(FOURS, dice_roll_yahtzee_again) # Score 20 for Fours, +100 Yahtzee Bonus
    
    print(f"Potential for Full House with {dice_roll_3}: {sheet.get_potential_score(FULL_HOUSE, dice_roll_3)}")
    sheet.record_score(FULL_HOUSE, dice_roll_3)

    sheet.display_scoresheet()
    print(f"Available categories: {sheet.get_available_categories()}")

    # Test custom category
    def score_all_even(dice_values):
        return sum(d for d in dice_values if d % 2 == 0)
    
    sheet.add_custom_category("AllEvens", score_all_even, display_name="Sum of Evens")
    custom_roll = [2, 2, 4, 1, 6]
    print(f"Potential for Sum of Evens with {custom_roll}: {sheet.get_potential_score('AllEvens', custom_roll)}")
    sheet.record_score("AllEvens", custom_roll)

    sheet.display_scoresheet()

    # Fill up upper section to test bonus
    sheet.record_score(TWOS, [2,2,2,1,1]) # 6
    sheet.record_score(THREES, [3,3,3,1,1]) # 9
    # sheet.record_score(FOURS, [4,4,4,1,1]) # 12 - Fours already scored by Yahtzee bonus example
    sheet.record_score(FIVES, [5,5,5,1,1]) # 15
    sheet.record_score(SIXES, [6,6,6,1,1]) # 18
    # Current upper: Ones (2) + Fours (20) + Twos (6) + Threes (9) + Fives (15) + Sixes (18) = 70. Bonus should apply.
    
    sheet.display_scoresheet()
    print(f"Is sheet complete? {sheet.is_complete()}")

    # Try scoring an already scored category
    sheet.record_score(ONES, [1,1,1,1,1]) 