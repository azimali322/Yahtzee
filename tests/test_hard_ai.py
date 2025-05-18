import pytest
from yahtzee_ai import YahtzeeAI
from game_logic import (
    ScoreSheet, Dice,
    ONES, TWOS, THREES, FOURS, FIVES, SIXES,
    THREE_OF_A_KIND, FOUR_OF_A_KIND, FULL_HOUSE,
    SMALL_STRAIGHT, LARGE_STRAIGHT, YAHTZEE, CHANCE
)

@pytest.fixture
def hard_ai_setup():
    """Setup fixture for hard AI tests."""
    dice = Dice()
    scoresheet = ScoreSheet()
    ai = YahtzeeAI("hard")
    ai.set_game_state(scoresheet, dice)
    return ai, dice, scoresheet

def test_yahtzee_priority(hard_ai_setup):
    """Test that AI prioritizes Yahtzee when available."""
    ai, _, _ = hard_ai_setup
    dice_values = [6, 6, 6, 6, 6]  # Yahtzee of sixes
    
    # Make sure YAHTZEE is available
    assert YAHTZEE in ai.scoresheet.get_available_categories()
    
    # AI should choose Yahtzee category
    chosen_category = ai.choose_category(dice_values)
    assert chosen_category == YAHTZEE
    
    # Verify score
    score = ai.scoresheet.get_potential_score(chosen_category, dice_values)
    assert score == 50

def test_upper_section_bonus_strategy(hard_ai_setup):
    """Test AI's strategy for achieving upper section bonus."""
    ai, _, _ = hard_ai_setup
    
    # Test with a roll that could contribute to bonus
    dice_values = [5, 5, 5, 2, 1]  # Three fives
    chosen_category = ai.choose_category(dice_values)
    
    # Should choose FIVES to maximize bonus potential (three 5s = 15 points)
    assert chosen_category == FIVES
    score = ai.scoresheet.get_potential_score(chosen_category, dice_values)
    assert score == 15  # 3 fives = 15 points

def test_reroll_strategy_for_straight(hard_ai_setup):
    """Test AI's reroll strategy when pursuing a straight."""
    ai, _, _ = hard_ai_setup
    dice_values = [1, 2, 3, 4, 6]  # One away from large straight
    
    # First roll
    reroll_indices = ai._decide_reroll_hard(dice_values, 1)
    assert len(reroll_indices) == 1  # Should only reroll the 6
    assert 4 in reroll_indices  # Index of the 6

def test_full_house_strategy(hard_ai_setup):
    """Test AI's strategy for Full House opportunities."""
    ai, _, _ = hard_ai_setup
    dice_values = [3, 3, 3, 4, 4]  # Perfect Full House
    
    # Make sure FULL_HOUSE is available
    assert FULL_HOUSE in ai.scoresheet.get_available_categories()
    
    chosen_category = ai.choose_category(dice_values)
    assert chosen_category == FULL_HOUSE
    score = ai.scoresheet.get_potential_score(chosen_category, dice_values)
    assert score == 25

def test_late_game_strategy(hard_ai_setup):
    """Test AI's late game strategy."""
    ai, _, _ = hard_ai_setup
    
    # Simulate late game by filling most categories
    # Fill upper section except SIXES
    for category in [ONES, TWOS, THREES, FOURS, FIVES]:
        ai.scoresheet.record_score(category, [1, 1, 1, 1, 1])
    
    # Fill some lower section
    ai.scoresheet.record_score(SMALL_STRAIGHT, [1, 2, 3, 4, 5])
    ai.scoresheet.record_score(LARGE_STRAIGHT, [2, 3, 4, 5, 6])
    ai.scoresheet.record_score(FULL_HOUSE, [2, 2, 2, 3, 3])
    
    # Test with a good roll for THREE_OF_A_KIND
    dice_values = [6, 6, 6, 4, 5]  # Three of a kind
    
    # Make sure THREE_OF_A_KIND is available
    assert THREE_OF_A_KIND in ai.scoresheet.get_available_categories()
    
    # AI should be conservative in late game and prefer keeping three of a kind
    chosen_category = ai.choose_category(dice_values)
    assert chosen_category == THREE_OF_A_KIND  # Should take the guaranteed three of a kind
    score = ai.scoresheet.get_potential_score(chosen_category, dice_values)
    assert score == 27  # Sum of all dice (6+6+6+4+5 = 27)

def test_chance_optimization(hard_ai_setup):
    """Test AI's strategy for optimizing Chance category."""
    ai, _, _ = hard_ai_setup
    dice_values = [6, 6, 5, 5, 4]  # High value roll (26 points)
    
    # Early game - should prefer pairs of high numbers over Chance
    # Initialize scoresheet with some scores to make it early game
    ai.scoresheet.record_score(ONES, [1, 1, 1, 1, 1])
    ai.scoresheet.record_score(TWOS, [2, 2, 2, 2, 2])
    
    first_choice = ai.choose_category(dice_values)
    assert first_choice == SIXES  # Should take the pair of sixes
    score = ai.scoresheet.get_potential_score(first_choice, dice_values)
    assert score == 12  # Two sixes = 12
    
    # Late game - should use Chance for high rolls
    ai, _, _ = hard_ai_setup  # Get fresh setup
    # Fill all categories except CHANCE and SIXES
    for category in [cat for cat in ai.UPPER_SECTION if cat != SIXES]:
        ai.scoresheet.record_score(category, [1, 1, 1, 1, 1])
    # Fill some lower section to make it truly late game
    ai.scoresheet.record_score(SMALL_STRAIGHT, [1, 2, 3, 4, 5])
    ai.scoresheet.record_score(LARGE_STRAIGHT, [2, 3, 4, 5, 6])
    ai.scoresheet.record_score(FULL_HOUSE, [2, 2, 2, 3, 3])
    ai.scoresheet.record_score(THREE_OF_A_KIND, [2, 2, 2, 3, 4])
    ai.scoresheet.record_score(FOUR_OF_A_KIND, [2, 2, 2, 2, 4])
    
    # Verify available categories
    available = ai.scoresheet.get_available_categories()
    assert CHANCE in available, "CHANCE should be available"
    assert SIXES in available, "SIXES should be available"
    
    late_choice = ai.choose_category(dice_values)
    score = ai.scoresheet.get_potential_score(late_choice, dice_values)
    
    # Should choose either CHANCE (26 points) or SIXES (12 points)
    if late_choice == CHANCE:
        assert score == 26  # 6+6+5+5+4 = 26
    else:
        assert late_choice == SIXES
        assert score == 12  # Two sixes = 12

def test_strategic_weights(hard_ai_setup):
    """Test that strategic weights are properly applied."""
    ai, _, _ = hard_ai_setup
    dice_values = [4, 4, 4, 4, 1]
    
    # Early game weights
    game_state = ai._analyze_game_state()  # Use actual game state analysis
    dice_analysis = ai._analyze_dice_values(dice_values)
    
    # Keep the four 4s
    kept_values = [dice_values[i] for i in [0, 1, 2, 3]]
    kept_analysis = ai._analyze_dice_values(kept_values)
    
    # Calculate weighted score
    base_score = ai.scoresheet.get_potential_score(FOUR_OF_A_KIND, dice_values)
    weighted_score = ai._apply_strategic_weights(
        base_score,
        [0, 1, 2, 3],  # Keep the four 4s
        dice_values,
        game_state,
        kept_analysis
    )
    
    assert weighted_score > base_score  # Strategic weight should increase the score

def test_complete_enumeration(hard_ai_setup):
    """Test the complete enumeration strategy."""
    ai, _, _ = hard_ai_setup
    dice_values = [6, 6, 6, 1, 2]
    
    # Should identify keeping three 6s as optimal
    best_reroll = ai._complete_enumeration_strategy(
        dice_values,
        1,  # First roll
        ai._analyze_game_state(),
        ai._analyze_dice_values(dice_values)
    )
    
    # Should keep the three 6s and reroll the others
    assert len(best_reroll) == 2
    assert all(dice_values[i] != 6 for i in best_reroll) 