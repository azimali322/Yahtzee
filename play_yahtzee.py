from game_manager import GameManager
from game_logic import UPPER_SECTION_CATEGORIES, LOWER_SECTION_CATEGORIES, ALL_CATEGORIES

def print_available_categories(scoresheet):
    """Prints all available scoring categories."""
    print("\nAvailable categories:")
    for category in ALL_CATEGORIES:
        if scoresheet.scores[category] is None:
            print(f"- {category}")

def get_category_choice(scoresheet):
    """Gets the player's category choice."""
    available = [cat for cat in ALL_CATEGORIES if scoresheet.scores[cat] is None]
    while True:
        choice = input("\nEnter category to score: ").strip()
        if choice in available:
            return choice
        print("Invalid category. Please choose from the available categories.")

def play_turn(game_manager):
    """Handles a single player's turn."""
    name, scoresheet = game_manager.get_current_player()
    dice = game_manager.dice
    
    print(f"\n=== {name}'s Turn ===")
    scoresheet.display_scoresheet()
    
    # Allow up to 3 rolls
    for roll_num in range(3):
        if roll_num == 0:
            dice.roll_all()
        else:
            print("\nCurrent dice:", dice)
            if input("Would you like to roll again? (y/n): ").lower() != 'y':
                break
            
            # Get which dice to reroll
            while True:
                try:
                    to_reroll = input("Enter dice positions to reroll (1-5, space-separated) or press Enter for all: ").strip()
                    if not to_reroll:  # Roll all unheld dice
                        dice.roll_all()
                        break
                    indices = [int(x) - 1 for x in to_reroll.split()]
                    if all(0 <= i < 5 for i in indices):
                        dice.roll_specific(indices)
                        break
                    print("Invalid dice positions. Please use numbers 1-5.")
                except ValueError:
                    print("Invalid input. Please use numbers 1-5, space-separated.")
    
    # Final dice state
    print("\nFinal dice:", dice)
    
    # Score the turn
    print_available_categories(scoresheet)
    category = get_category_choice(scoresheet)
    score = scoresheet.get_potential_score(category, dice.get_values())
    scoresheet.record_score(category, dice.get_values())
    print(f"\nScored {score} points in {category}")
    scoresheet.display_scoresheet()

def main():
    """Main game loop."""
    print("Welcome to Multiplayer Yahtzee!")
    
    # Setup game
    game = GameManager()
    game.setup_game()
    
    # Main game loop
    while not game.is_game_over():
        play_turn(game)
        game.next_turn()
    
    # Display final results
    print("\nGame Over!")
    game.display_rankings()

if __name__ == "__main__":
    main() 