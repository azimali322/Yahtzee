# Yahtzee AI Implementation

A Python implementation of Yahtzee featuring multiple AI agents with different strategies and a complete test suite. The project focuses on implementing and benchmarking various AI decision-making approaches for playing Yahtzee.

## Features

### Game Implementation
- Complete Yahtzee game logic implementation
- Support for multiple players (human and AI)
- Full scoring system including bonus points
- Interactive command-line interface

### AI Agents
The project implements several AI agents with different strategies:

1. **Greedy AI Levels**
   - `greedy1`: Never rerolls, takes best immediate score
   - `greedy2`: One reroll allowed, uses complete enumeration
   - `greedy3`: Two rerolls allowed, uses complete enumeration

2. **Difficulty-based AI**
   - `easy`: Simple strategy focusing on most frequent values
   - `medium`: Balanced strategy considering straights and high-value combinations
   - `hard`: Advanced strategy using expected value calculations
   - `random`: Random decision making for baseline comparison

### Decision Making Strategies
- Complete enumeration of all possible outcomes
- Optimal reroll decision making
- Category selection optimization
- Special case handling for common scenarios (pairs, straights, etc.)

### Benchmarking and Analysis
- Comprehensive test suite using pytest
- Performance metrics and statistics using numpy
- Data analysis using pandas
- Visualization capabilities for strategy comparison

## Requirements

```python
# Core dependencies
python >= 3.8
numpy
pandas
pytest

# Optional dependencies for visualization
matplotlib
seaborn
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Yahtzee
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Playing the Game

```python
from game_logic import ScoreSheet, Dice
from yahtzee_ai import YahtzeeAI

# Create a new game
scoresheet = ScoreSheet()
dice = Dice()

# Create an AI player
ai = YahtzeeAI(difficulty="greedy2")
ai.set_game_state(scoresheet, dice)

# Play a turn
dice.roll_all()
reroll_indices = ai.decide_reroll(dice.get_values(), 1)
dice.roll_specific(reroll_indices)
category = ai.choose_category(dice.get_values())
score = scoresheet.record_score(category, dice.get_values())
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest test_complete_enumeration.py

# Run with verbose output
pytest -v

# Run with coverage report
pytest --cov=.
```

## Code Structure

```
.
├── game_logic.py          # Core game mechanics and rules
├── yahtzee_ai.py         # AI agent implementations
├── play_yahtzee.py       # Game runner and UI
├── benchmark_ai.py       # AI performance testing
└── tests/
    ├── test_complete_enumeration.py  # AI strategy tests
    └── test_game_logic.py           # Game mechanics tests
```

## AI Strategy Details

### Complete Enumeration Strategy
The greedy AI agents use complete enumeration to evaluate all possible outcomes:

1. **Decision Making Process**:
   - Evaluates all possible keep/reroll combinations
   - Calculates exact expected values for each choice
   - Considers special cases (e.g., two pairs)
   - No probability approximations used

2. **Scoring Optimization**:
   - Maintains optimal choices for common patterns
   - Balances immediate vs. potential scores
   - Special handling for Yahtzee bonus opportunities

3. **Performance Considerations**:
   - Uses efficient numpy operations for calculations
   - Implements smart pruning for obvious decisions
   - Balances accuracy vs. computation time

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## Testing

The project includes a comprehensive test suite:

```bash
# Run all tests with coverage
pytest --cov=. --cov-report=term-missing

# Run specific test categories
pytest test_complete_enumeration.py -v -k "test_greedy"
```

## License

[MIT License](LICENSE)

## Acknowledgments

- Built with Python 3.8
- Uses numpy for efficient computations
- Uses pandas for data analysis
- Uses pytest for testing framework
