from game_logic import Dice, ScoreSheet, ALL_CATEGORIES, UPPER_SECTION_CATEGORIES, LOWER_SECTION_CATEGORIES, YAHTZEE
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

            if allowed_values:
                if isinstance(value, list):
                    # This check might be too restrictive if allowed_values is for each item.
                    # For now, assuming allowed_values is for the whole list if it's a list.
                    # Or, if it's a list of choices for string inputs like ['r', 's'].
                    # The current usage is for string inputs.
                    if not all(item in allowed_values for item in value) and value:
                        print(f"Invalid input. Please choose from {allowed_values}.")
                        continue
                elif value not in allowed_values:
                    print(f"Invalid input. Please choose from {allowed_values}.")
                    continue
            return value
        except ValueError:
            # This primarily catches int(user_input) failure for allowed_type == int
            print(f"Invalid input. Please enter a valid {allowed_type.__name__}.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

def play_turn(player_name, dice, scoresheet):
    """Manages a single turn for a player."""
    print(f"\n--- {player_name}'s Turn ---")
    dice.reset_roll_count() # Resets roll count and unholds all dice
    proceed_to_score_flag = False

    for roll_num in range(1, 4): # Max 3 rolls
        print(f"\nRoll {roll_num} of 3")
        if roll_num == 1:
            dice.roll_all()
        else: # Subsequent rolls
            # roll_all() in Dice class only rolls unheld dice, which is what we want
            dice.roll_all()

        print(f"Dice: {dice}")
        print(f"Values: {dice.get_values()}")

        if roll_num < 3:
            # Loop for getting hold input and deciding next action (roll again / score now)
            while True: 
                print("\nCurrent hold status:")
                for i, die_obj in enumerate(dice.dice):
                    print(f"  Die {i+1} ({die_obj.value}): {'Held' if die_obj.is_held else 'Not Held'}")

                hold_input_list = get_player_input(
                    "Enter dice numbers (1-5, comma-separated) to toggle hold status (e.g., 1,3,5), or press Enter to keep current holds: ", 
                    allowed_type=list, # Expects list of ints or empty list
                    allow_empty=True
                )

                indices_to_toggle = []
                valid_numbers_for_toggle = True
                if hold_input_list: # If list is not empty, player entered numbers to toggle
                    for die_num_from_list in hold_input_list:
                        if 1 <= die_num_from_list <= len(dice.dice):
                            indices_to_toggle.append(die_num_from_list - 1) # 0-indexed
                        else:
                            print(f"Invalid die number: {die_num_from_list}. Must be between 1 and {len(dice.dice)}.")
                            valid_numbers_for_toggle = False
                            break 
                    if not valid_numbers_for_toggle:
                        continue # Re-ask for dice to hold/unhold

                chosen_action_post_holds = '' # r or s

                if indices_to_toggle: # Player specified dice to toggle, now confirm
                    # --- Confirmation Sub-Loop --- 
                    while True:
                        print("\nPreview of changes to held dice:")
                        current_holds_status = [d.is_held for d in dice.dice]
                        proposed_holds_status = list(current_holds_status) # Create a mutable copy
                        
                        for index_to_toggle_preview in indices_to_toggle:
                            # Toggle the state in the preview
                            proposed_holds_status[index_to_toggle_preview] = not proposed_holds_status[index_to_toggle_preview]

                        for i in range(len(dice.dice)):
                            original_status_text = "Held" if current_holds_status[i] else "Not Held"
                            proposed_status_text = "Held" if proposed_holds_status[i] else "Not Held"
                            change_indicator = ""
                            if current_holds_status[i] != proposed_holds_status[i]:
                                change_indicator = f" (will change from {original_status_text} to {proposed_status_text})"
                            print(f"  Die {i+1} ({dice.dice[i].value}): Current: {original_status_text}{change_indicator}")
                        
                        confirm_choice = get_player_input("Confirm these changes? (y/n): ", allowed_values=['y', 'n'])
                        if confirm_choice == 'y':
                            for index_to_apply_toggle in indices_to_toggle: # Apply confirmed changes
                                dice.toggle_hold(index_to_apply_toggle)
                            
                            print("\nUpdated hold status:")
                            for i, die_obj in enumerate(dice.dice):
                                print(f"  Die {i+1} ({die_obj.value}): {'Held' if die_obj.is_held else 'Not Held'}")
                            
                            chosen_action_post_holds = get_player_input("Roll again (r) or score this turn (s)? ", allowed_values=['r', 's'])
                            break # Break from confirmation sub-loop
                        else: # confirm_choice == 'n'
                            print("Hold selection cancelled. Please re-enter dice to hold/unhold.")
                            chosen_action_post_holds = None # Signal to re-prompt for holds
                            break # Break from confirmation sub-loop
                    # --- End Confirmation Sub-Loop ---

                    if chosen_action_post_holds is None: # Means confirmation was 'n', re-prompt for hold_input_list
                        continue 
                    # If 'y' was confirmed, chosen_action_post_holds is 'r' or 's'
                
                else: # No dice specified to toggle (hold_input_list was empty)
                    print("Keeping current hold status.")
                    chosen_action_post_holds = get_player_input("Roll unheld dice (r) or score this turn (s)? ", allowed_values=['r', 's'])

                # At this point, chosen_action_post_holds is 'r' or 's'
                if chosen_action_post_holds == 'r':
                    break # Break from the input `while True` loop, proceed to next roll in `for roll_num`
                elif chosen_action_post_holds == 's':
                    proceed_to_score_flag = True
                    break # Break from the input `while True` loop
            
            if proceed_to_score_flag:
                print("Proceeding to scoring phase early.")
                break # Break from the `for roll_num` loop (main rolling loop)
        else: # This is for roll_num == 3 (final roll of the turn)
            print("Final roll for this turn. Proceeding to scoring.")
            # Loop will naturally end, and execution will fall through to the scoring phase

    # Scoring phase
    print("\n--- Choose a Category to Score ---")
    current_dice_values = dice.get_values()
    print(f"Final Dice: {dice} (Values: {current_dice_values})")
    
    available_categories = scoresheet.get_available_categories()
    if not available_categories:
        print("No categories left to score! This shouldn't happen in a standard game.")
        return # Should ideally not happen if game loop runs for num_turns

    print("Available Categories and Potential Scores:")
    potential_scores_display_map = {}
    for i, category_name_iter in enumerate(available_categories):
        score = scoresheet.get_potential_score(category_name_iter, current_dice_values)
        display_text = f"  {i+1}. {category_name_iter:<20}: {score}"
        potential_scores_display_map[str(i+1)] = category_name_iter
        print(display_text)

    # Ensure there are categories to choose from before prompting
    if not potential_scores_display_map:
        print("Error: No available categories to display for scoring.")
        # This might indicate an issue if available_categories was populated but map isn't
        return

    chosen_category_num_str = get_player_input(
        "Select a category number to score: ", 
        allowed_type=str, # Input will be a string number like "1", "2"
        allowed_values=list(potential_scores_display_map.keys())
    )
    chosen_category_name = potential_scores_display_map[chosen_category_num_str]
    
    scoresheet.record_score(chosen_category_name, current_dice_values)
    scoresheet.display_scoresheet()

def game_loop():
    print("Welcome to Command-Line Yahtzee!")
    
    player_name = "Player 1" # For now, single player
    player_dice = Dice()
    player_scoresheet = ScoreSheet()

    num_turns = len(ALL_CATEGORIES) # Standard Yahtzee has 13 scoring categories for a full game

    for turn_num in range(1, num_turns + 1):
        print(f"\n==================== Turn {turn_num}/{num_turns} ====================")
        play_turn(player_name, player_dice, player_scoresheet)
        if player_scoresheet.is_complete():
            print("All categories scored! Game finished early.")
            break
    
    print("\n==================== Game Over ====================")
    print(f"Final Scores for {player_name}:")
    player_scoresheet.display_scoresheet()
    print(f"GRAND TOTAL for {player_name}: {player_scoresheet.get_grand_total()}")
    print("Thanks for playing!")

if __name__ == "__main__":
    game_loop() 