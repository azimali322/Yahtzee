import unittest
from unittest.mock import patch
from io import StringIO
from game_manager import GameManager
from game_logic import YAHTZEE, CHANCE

class TestGameManager(unittest.TestCase):
    def setUp(self):
        self.game = GameManager()
    
    @patch('builtins.input')
    def test_setup_game(self, mock_input):
        # Mock inputs: 3 players with names "Alice", "Bob", and "Charlie"
        mock_input.side_effect = ["3", "Alice", "Bob", "Charlie"]
        
        self.game.setup_game()
        
        self.assertEqual(len(self.game.players), 3)
        self.assertEqual([name for name, _ in self.game.players], ["Alice", "Bob", "Charlie"])
        self.assertEqual(self.game.current_player_index, 0)
    
    @patch('builtins.input')
    def test_setup_game_invalid_inputs(self, mock_input):
        # Test invalid number of players followed by valid input
        mock_input.side_effect = ["0", "11", "abc", "2", "Alice", "Bob"]
        
        self.game.setup_game()
        
        self.assertEqual(len(self.game.players), 2)
        self.assertEqual([name for name, _ in self.game.players], ["Alice", "Bob"])
    
    @patch('builtins.input')
    def test_setup_game_duplicate_names(self, mock_input):
        # Test duplicate names followed by unique name
        mock_input.side_effect = ["2", "Alice", "Alice", "Bob"]
        
        self.game.setup_game()
        
        self.assertEqual(len(self.game.players), 2)
        self.assertEqual([name for name, _ in self.game.players], ["Alice", "Bob"])
    
    def test_next_turn(self):
        # Setup a game with 3 players
        self.game.players = [
            ("Alice", None),
            ("Bob", None),
            ("Charlie", None)
        ]
        
        self.assertEqual(self.game.current_player_index, 0)
        
        name, _ = self.game.next_turn()
        self.assertEqual(name, "Bob")
        self.assertEqual(self.game.current_player_index, 1)
        
        name, _ = self.game.next_turn()
        self.assertEqual(name, "Charlie")
        self.assertEqual(self.game.current_player_index, 2)
        
        name, _ = self.game.next_turn()
        self.assertEqual(name, "Alice")
        self.assertEqual(self.game.current_player_index, 0)
    
    def test_get_rankings_no_ties(self):
        # Setup game with different scores
        self.game.players = [
            ("Alice", self._create_mock_scoresheet(100)),
            ("Bob", self._create_mock_scoresheet(200)),
            ("Charlie", self._create_mock_scoresheet(150))
        ]
        
        rankings = self.game.get_rankings()
        
        expected = [
            (1, "Bob", 200),
            (2, "Charlie", 150),
            (3, "Alice", 100)
        ]
        self.assertEqual(rankings, expected)
    
    def test_get_rankings_with_ties(self):
        # Setup game with tied scores
        self.game.players = [
            ("Alice", self._create_mock_scoresheet(150)),
            ("Bob", self._create_mock_scoresheet(200)),
            ("Charlie", self._create_mock_scoresheet(150)),
            ("David", self._create_mock_scoresheet(100))
        ]
        
        rankings = self.game.get_rankings()
        
        expected = [
            (1, "Bob", 200),
            (2, "Alice", 150),
            (2, "Charlie", 150),
            (4, "David", 100)
        ]
        self.assertEqual(rankings, expected)
    
    def test_get_rankings_with_multiple_ties_ten_players(self):
        """Test rankings with 10 players where 7 players tie for second place."""
        # Setup game with 10 players:
        # - 1 player with 300 points (1st place)
        # - 7 players with 200 points (tied for 2nd place)
        # - 1 player with 150 points (9th place)
        # - 1 player with 100 points (10th place)
        self.game.players = [
            ("Alice", self._create_mock_scoresheet(200)),    # Tied 2nd
            ("Bob", self._create_mock_scoresheet(300)),      # 1st
            ("Charlie", self._create_mock_scoresheet(200)),  # Tied 2nd
            ("David", self._create_mock_scoresheet(200)),    # Tied 2nd
            ("Eve", self._create_mock_scoresheet(200)),      # Tied 2nd
            ("Frank", self._create_mock_scoresheet(200)),    # Tied 2nd
            ("Grace", self._create_mock_scoresheet(200)),    # Tied 2nd
            ("Henry", self._create_mock_scoresheet(200)),    # Tied 2nd
            ("Ivy", self._create_mock_scoresheet(150)),      # 9th
            ("Jack", self._create_mock_scoresheet(100))      # 10th
        ]
        
        rankings = self.game.get_rankings()
        
        expected = [
            (1, "Bob", 300),
            (2, "Alice", 200),
            (2, "Charlie", 200),
            (2, "David", 200),
            (2, "Eve", 200),
            (2, "Frank", 200),
            (2, "Grace", 200),
            (2, "Henry", 200),
            (9, "Ivy", 150),
            (10, "Jack", 100)
        ]
        self.assertEqual(rankings, expected)
        
        # Test the display output
        with patch('sys.stdout', new=StringIO()) as fake_output:
            self.game.display_rankings()
            output = fake_output.getvalue()
        
        # Verify the output contains the correct information
        self.assertIn("=== Final Rankings ===", output)
        self.assertIn("Bob", output)
        self.assertIn("300", output)
        self.assertIn("Tie for 2nd place between: Alice, Charlie, David, Eve, Frank, Grace, Henry", output)
        self.assertIn("   9  Ivy", output)  # Match exact formatting
        self.assertIn("  10  Jack", output)  # Match exact formatting
    
    def _create_mock_scoresheet(self, total_score):
        """Creates a mock scoresheet that returns the specified total score."""
        class MockScoreSheet:
            def __init__(self, score):
                self.total = score
                self.scores = {YAHTZEE: 50, CHANCE: 30}  # Example scores
            
            def get_grand_total(self):
                return self.total
            
            def is_complete(self):
                return True
        
        return MockScoreSheet(total_score)
    
    def test_display_rankings(self):
        # Setup game with tied scores
        self.game.players = [
            ("Alice", self._create_mock_scoresheet(150)),
            ("Bob", self._create_mock_scoresheet(200)),
            ("Charlie", self._create_mock_scoresheet(150)),
            ("David", self._create_mock_scoresheet(100))
        ]
        
        # Capture printed output
        with patch('sys.stdout', new=StringIO()) as fake_output:
            self.game.display_rankings()
            output = fake_output.getvalue()
        
        # Verify output contains expected elements
        self.assertIn("=== Final Rankings ===", output)
        self.assertIn("Bob", output)
        self.assertIn("200", output)
        self.assertIn("Tie for 2nd place between: Alice, Charlie", output)

if __name__ == '__main__':
    unittest.main() 