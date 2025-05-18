import pytest
import numpy as np
import pandas as pd
from yahtzee_ai import YahtzeeAI
from game_logic import (
    ScoreSheet, Dice,
    ONES, TWOS, THREES, FOURS, FIVES, SIXES,
    THREE_OF_A_KIND, FOUR_OF_A_KIND, FULL_HOUSE,
    SMALL_STRAIGHT, LARGE_STRAIGHT, YAHTZEE, CHANCE
)
import itertools

@pytest.fixture
def ai_setup():
    """Setup fixture for AI tests."""
    dice = Dice()
    scoresheet = ScoreSheet()
    ai = YahtzeeAI("greedy2")
    ai.set_game_state(scoresheet, dice)
    return ai, dice, scoresheet

# Test data for expected score calculations
EXPECTED_SCORE_TEST_CASES = [
    {
        'dice_values': [6, 6, 1, 2, 3],
        'keep_indices': [0, 1],
        'min_expected': 12,
        'description': 'pair of sixes'
    },
    {
        'dice_values': [5, 5, 5, 1, 2],
        'keep_indices': [0, 1, 2],
        'min_expected': 18,
        'description': 'three of a kind'
    },
    {
        'dice_values': [1, 2, 3, 4, 6],
        'keep_indices': [0, 1, 2, 3],
        'min_expected': 25,
        'description': 'potential small straight'
    }
]

# Test data for greedy decisions
GREEDY_DECISION_TEST_CASES = pd.DataFrame([
    {
        'dice_values': [6, 6, 6, 1, 1],
        'roll_number': 1,
        'expected_reroll_count': 2,
        'description': 'reroll two ones with three of a kind'
    },
    {
        'dice_values': [1, 2, 3, 4, 6],
        'roll_number': 1,
        'expected_reroll_count': 1,
        'description': 'reroll one for straight'
    },
    {
        'dice_values': [5, 5, 5, 5, 1],
        'roll_number': 1,
        'expected_reroll_count': 1,
        'description': 'reroll one for yahtzee'
    },
    {
        'dice_values': [3, 3, 4, 4, 1],
        'roll_number': 1,
        'expected_reroll_count': 1,
        'description': 'reroll one for full house'
    }
])

@pytest.mark.parametrize("test_case", EXPECTED_SCORE_TEST_CASES)
def test_expected_score_calculation(ai_setup, test_case):
    """Test that expected score calculation considers all possibilities."""
    ai, _, _ = ai_setup
    expected_score = ai._calculate_expected_score_for_reroll(
        test_case['keep_indices'], 
        test_case['dice_values']
    )
    assert expected_score > test_case['min_expected'], \
        f"Expected score for {test_case['description']} should be higher than {test_case['min_expected']}"

def test_all_combinations_considered(ai_setup):
    """Test that all possible dice combinations are considered."""
    ai, _, _ = ai_setup
    dice_values = [1, 1, 1, 2, 2]
    keep_indices = [0, 1, 2]  # Keep three ones
    
    # Calculate total possible combinations for 2 dice
    total_combinations = 21  # Number of unique combinations with replacement for 2 dice
    
    # Create numpy array to store all combinations and scores
    combinations = []
    scores = []
    
    # Generate all combinations using numpy
    for reroll_values in itertools.combinations_with_replacement(range(1, 7), 2):
        combined_values = [1, 1, 1] + list(reroll_values)  # Add kept ones
        combinations.append(combined_values)
        scores.append(ai._get_max_score(combined_values))
    
    combinations_array = np.array(combinations)
    scores_array = np.array(scores)
    
    # Calculate statistics
    stats = {
        'mean_score': np.mean(scores_array),
        'std_score': np.std(scores_array),
        'min_score': np.min(scores_array),
        'max_score': np.max(scores_array),
        'unique_combinations': len(np.unique(combinations_array, axis=0))
    }
    
    # Verify we saw all combinations
    assert stats['unique_combinations'] == total_combinations, \
        f"Expected {total_combinations} combinations, got {stats['unique_combinations']}"
    
    # Verify AI calculation matches manual calculation
    ai_score = ai._calculate_expected_score_for_reroll(keep_indices, dice_values)
    manual_expected_score = stats['mean_score']
    
    np.testing.assert_almost_equal(ai_score, manual_expected_score, decimal=2,
        err_msg=f"AI score {ai_score} differs from manual calculation {manual_expected_score}")

@pytest.mark.parametrize(
    "dice_values,roll_number,expected_reroll_count,description",
    GREEDY_DECISION_TEST_CASES.values
)
def test_greedy_decisions(ai_setup, dice_values, roll_number, expected_reroll_count, description):
    """Test that greedy AI makes optimal decisions with complete enumeration."""
    ai, _, _ = ai_setup
    reroll_indices = ai._decide_reroll_greedy2(list(dice_values), roll_number)
    
    assert len(reroll_indices) == expected_reroll_count, \
        f"{description}: Expected {expected_reroll_count} rerolls, got {len(reroll_indices)}"
    
    # Additional verification for the two pairs case
    if list(dice_values) == [3, 3, 4, 4, 1]:
        score_keep_pairs = ai._calculate_expected_score_for_reroll([0, 1, 2, 3], dice_values)
        score_keep_three = ai._calculate_expected_score_for_reroll([0], dice_values)
        
        assert score_keep_pairs > score_keep_three, \
            "Keeping both pairs should have higher expected value than keeping one three"

def test_no_early_pruning(ai_setup):
    """Test that the AI considers all possibilities without early pruning."""
    ai, _, _ = ai_setup
    dice_values = [6, 6, 6, 6, 1]  # Four of a kind
    
    # Create a DataFrame to store and analyze all possible keep combinations
    keep_combinations = []
    scores = []
    
    # Generate all possible keep combinations
    for keep_count in range(5):
        for indices in itertools.combinations(range(5), keep_count):
            score = ai._calculate_expected_score_for_reroll(list(indices), dice_values)
            keep_combinations.append(list(indices))
            scores.append(score)
    
    results_df = pd.DataFrame({
        'keep_indices': keep_combinations,
        'expected_score': scores
    })
    
    # Sort by expected score to find best strategies
    results_df = results_df.sort_values('expected_score', ascending=False)
    
    # Verify that keeping four of a kind and three of a kind are both considered
    four_of_a_kind_score = results_df[
        results_df['keep_indices'].apply(lambda x: len(x) == 4)
    ]['expected_score'].iloc[0]
    
    three_of_a_kind_score = results_df[
        results_df['keep_indices'].apply(lambda x: len(x) == 3)
    ]['expected_score'].iloc[0]
    
    assert pd.notna(four_of_a_kind_score), "Four of a kind strategy should be considered"
    assert pd.notna(three_of_a_kind_score), "Three of a kind strategy should be considered" 