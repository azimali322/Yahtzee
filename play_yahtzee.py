from game_manager import GameManager
from game_logic import UPPER_SECTION_CATEGORIES, LOWER_SECTION_CATEGORIES, ALL_CATEGORIES

def print_available_categories(scoresheet, dice_values):
    """Prints all available scoring categories with their potential scores."""
    print("\nAvailable categories (with potential scores):")
    print("-" * 50)
    print(f"{'#':<3} {'Category':<20} {'Potential Score':>10}")
    print("-" * 50)
    
    # Keep track of available categories and their numbers
    available_categories = []
    option_num = 1
    
    # First show upper section categories
    if any(scoresheet.scores[cat] is None for cat in UPPER_SECTION_CATEGORIES):
        print("Upper Section:")
        for category in UPPER_SECTION_CATEGORIES:
            if scoresheet.scores[category] is None:
                potential_score = scoresheet.get_potential_score(category, dice_values)
                print(f"{option_num:<3} {category:<20} {potential_score:>10}")
                available_categories.append(category)
                option_num += 1
    
    # Then show lower section categories
    if any(scoresheet.scores[cat] is None for cat in LOWER_SECTION_CATEGORIES):
        print("\nLower Section:")
        for category in LOWER_SECTION_CATEGORIES:
            if scoresheet.scores[category] is None:
                potential_score = scoresheet.get_potential_score(category, dice_values)
                print(f"{option_num:<3} {category:<20} {potential_score:>10}")
                available_categories.append(category)
                option_num += 1
    print("-" * 50)
    return available_categories

def get_category_choice(scoresheet, available_categories):
    """Gets the player's category choice using numbered options."""
    while True:
        choice = input("\nEnter category number to score: ").strip()
        try:
            choice_num = int(choice)
            if 1 <= choice_num <= len(available_categories):
                return available_categories[choice_num - 1]
            print(f"Please enter a number between 1 and {len(available_categories)}.")
        except ValueError:
            # Also allow typing the category name for backward compatibility
            if choice in available_categories:
                return choice
            print(f"Please enter a valid category number (1-{len(available_categories)}).")

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
    
    # Final dice state and scoring options
    print("\nFinal dice:", dice)
    print("\nScoring options for:", " ".join(str(v) for v in dice.get_values()))
    available_categories = print_available_categories(scoresheet, dice.get_values())
    
    # Get category choice and score it
    category = get_category_choice(scoresheet, available_categories)
    score = scoresheet.get_potential_score(category, dice.get_values())
    scoresheet.record_score(category, dice.get_values())
    print(f"\nScored {score} points in {category}")
    scoresheet.display_scoresheet()

    # Ask if player wants to quit after completing their turn
    if input("\nWould you like to quit the game? (y/n): ").lower() == 'y':
        print(f"\n{name} has left the game.")
        return game_manager.remove_player(name)
    
    return True  # Continue game

def main():
    """Main game loop."""
    print("Welcome to Yahtzee!")
    print("You can play alone or with up to 9 other players.")
    print("Players can choose to quit after each round.")
    
    # Setup game
    game = GameManager()
    game.setup_game()
    
    # Main game loop
    while not game.is_game_over():
        if not play_turn(game):
            if len(game.players) == 0:
                print("\nAll players have quit.")
                game.display_all_scores()
                break
        game.next_turn()
    
    # Display final results
    if len(game.players) > 0:
        print("\nGame Over!")
        if len(game.players) == 1:
            name, scoresheet = game.players[0]
            print(f"\nFinal Score for {name}: {scoresheet.get_grand_total()}")
        game.display_rankings()
        # Also show scores for players who quit
        if game.quit_players:
            print("\nPlayers who quit earlier:")
            game.display_all_scores()

if __name__ == "__main__":
    main() 