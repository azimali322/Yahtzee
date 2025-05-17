from game_logic import Dice, ScoreSheet, ALL_CATEGORIES, UPPER_SECTION_CATEGORIES, LOWER_SECTION_CATEGORIES, YAHTZEE
from game_manager import GameManager
import sys

def get_player_input(prompt, allowed_type=str, allowed_values=None, allow_empty=False):
    """Generic function to get and validate player input."""
    while True:
        user_input = input(prompt).strip()
        if not user_input:
            if allow_empty:
                return [] if allowed_type == list else None
            else:
                print("Input cannot be empty.")
                continue
        
        try:
            if allowed_type == int:
                value = int(user_input)
            elif allowed_type == list:
                try:
                    # Ensure items are numbers, handles cases like "1,foo,3"
                    value = [int(x.strip()) for x in user_input.split(',') if x.strip()]
                    if not value and user_input: # e.g. input was just "," or ", ,"
                        print("Invalid format. Please enter comma-separated numbers (e.g., 1,2,3).")
                        continue
                except ValueError:
                    print("Invalid input. Ensure all items are comma-separated numbers (e.g., 1,2,3).")
                    continue
            else:
                value = user_input
            
            if allowed_values is not None and value not in allowed_values:
                print(f"Input must be one of: {', '.join(map(str, allowed_values))}")
                continue
            
            return value
        except ValueError:
            print(f"Invalid input. Expected {allowed_type.__name__}.")

def play_turn(name, dice, scoresheet, is_ai=False, ai_agent=None):
    """Manages a single turn for a player."""
    print(f"\n--- {name}'s Turn ---")
    dice.reset_roll_count() # Resets roll count and unholds all dice
    proceed_to_score_flag = False

    for roll_num in range(1, 4): # Max 3 rolls
        print(f"\nRoll {roll_num} of 3")
        if roll_num == 1:
            dice.roll_all()
        else: # Subsequent rolls
            if is_ai:
                # Get AI's reroll decision
                indices_to_reroll = ai_agent.decide_reroll(dice.get_values(), roll_num)
                if indices_to_reroll:
                    dice.roll_specific(indices_to_reroll)
                else:
                    proceed_to_score_flag = True
                    break
            else:
                # Human player's turn
                print("\nCurrent dice:", dice)
                print(f"Values: {dice.get_values()}")
                if input("Would you like to roll again? (y/n): ").lower() != 'y':
                    break
                
                # Get which dice to reroll
                while True: 
                    print("\nCurrent hold status:")
                    for i, die_obj in enumerate(dice.dice):
                        print(f"  Die {i+1} ({die_obj.value}): {'Held' if die_obj.is_held else 'Not Held'}")

                    hold_input_list = get_player_input(
                        "Enter dice numbers (1-5, comma-separated) to toggle hold status (e.g., 1,3,5), or press Enter to keep current holds: ", 
                        allowed_type=list,
                        allow_empty=True
                    )

                    indices_to_toggle = []
                    valid_numbers_for_toggle = True
                    if hold_input_list:
                        for die_num in hold_input_list:
                            if 1 <= die_num <= len(dice.dice):
                                indices_to_toggle.append(die_num - 1)
                            else:
                                print(f"Invalid die number: {die_num}. Must be between 1 and {len(dice.dice)}.")
                                valid_numbers_for_toggle = False
                                break
                        if not valid_numbers_for_toggle:
                            continue

                        # Apply hold toggles
                        for index in indices_to_toggle:
                            dice.toggle_hold(index)
                    
                    # Roll the dice
                    dice.roll_all()
                    break

        print(f"\nDice after roll {roll_num}: {dice}")
        print(f"Values: {dice.get_values()}")

    # Scoring phase
    print("\n--- Choose a Category to Score ---")
    current_dice_values = dice.get_values()
    print(f"Final Dice: {dice} (Values: {current_dice_values})")
    
    available_categories = scoresheet.get_available_categories()
    if not available_categories:
        print("No categories left to score! This shouldn't happen in a standard game.")
        return

    if is_ai:
        # AI chooses category
        chosen_category = ai_agent.choose_category(current_dice_values)
        score = scoresheet.get_potential_score(chosen_category, current_dice_values)
        print(f"\nAI chooses to score {score} points in {chosen_category}")
    else:
        # Human chooses category
        print("Available Categories and Potential Scores:")
        potential_scores_display_map = {}
        for i, category_name in enumerate(available_categories):
            score = scoresheet.get_potential_score(category_name, current_dice_values)
            display_text = f"  {i+1}. {category_name:<20}: {score}"
            potential_scores_display_map[str(i+1)] = category_name
            print(display_text)

        chosen_category_num = get_player_input(
            "Select a category number to score: ", 
            allowed_type=str,
            allowed_values=list(potential_scores_display_map.keys())
        )
        chosen_category = potential_scores_display_map[chosen_category_num]
    
    scoresheet.record_score(chosen_category, current_dice_values)
    scoresheet.display_scoresheet()

def main():
    """Main game loop."""
    print("Welcome to Yahtzee!")
    print("==================")
    print("\nYou can play against AI players with different difficulty levels:")
    print("- Type 'ai:easy' for an easy AI opponent")
    print("- Type 'ai:medium' for a medium AI opponent")
    print("- Type 'ai:hard' for a hard AI opponent")
    print("- Press Enter or type a name for a human player")
    
    # Create game manager and set up players
    game = GameManager()
    game.setup_game()
    
    # Main game loop
    while True:
        name, scoresheet, ai_agent = game.get_current_player()
        is_ai = ai_agent is not None
        
        print(f"\n=== {name}'s Turn ===")
        if is_ai:
            print("(AI Player)")
        
        # Display scoresheet
        scoresheet.display_scoresheet()
        
        # Handle player's turn
        play_turn(name, game.dice, scoresheet, is_ai, ai_agent)
        
        # Check if current player wants to quit (only for human players)
        if not is_ai and not scoresheet.is_complete():
            if get_player_input("Would you like to quit? (y/n): ", str, ['y', 'n']) == 'y':
                if not game.remove_player(name):
                    print("\nGame Over - All players have quit!")
                    break
                continue
        
        # Move to next player
        game.next_turn()
        
        # Check if game is complete
        if all(sheet.is_complete() for _, sheet, _ in game.players):
            print("\nGame Over!")
            game.display_rankings()
            break

if __name__ == "__main__":
    main() 