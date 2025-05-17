import random

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