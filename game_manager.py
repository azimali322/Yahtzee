from game_logic import ScoreSheet, Dice

class GameManager:
    """Manages a multiplayer Yahtzee game."""
    
    def __init__(self):
        self.players = []  # List of (name, scoresheet) tuples for active players
        self.quit_players = []  # List of (name, scoresheet) tuples for players who quit
        self.current_player_index = 0
        self.dice = Dice()
    
    def setup_game(self):
        """Sets up a new game by getting the number of players and their names."""
        while True:
            try:
                num_players = int(input("Enter number of players (1-10): "))
                if 1 <= num_players <= 10:
                    break
                print("Please enter a number between 1 and 10.")
            except ValueError:
                print("Please enter a valid number.")
        
        # Get player names and create scoresheets
        for i in range(num_players):
            while True:
                name = input(f"Enter name for Player {i + 1}: ").strip()
                if name and not any(player[0] == name for player in self.players):
                    break
                print("Please enter a unique, non-empty name.")
            self.players.append((name, ScoreSheet()))
    
    def next_turn(self):
        """Advances to the next player's turn."""
        self.current_player_index = (self.current_player_index + 1) % len(self.players)
        self.dice.reset_roll_count()
        return self.get_current_player()
    
    def get_current_player(self):
        """Returns the current player's name and scoresheet."""
        return self.players[self.current_player_index]
    
    def remove_player(self, player_name):
        """Removes a player from the game and adds them to quit_players list."""
        # Find the player to remove
        player_to_remove = None
        for player in self.players:
            if player[0] == player_name:
                player_to_remove = player
                break
        
        if player_to_remove:
            self.quit_players.append(player_to_remove)
            self.players = [(name, sheet) for name, sheet in self.players if name != player_name]
            if self.current_player_index >= len(self.players):
                self.current_player_index = 0
        return len(self.players) > 0  # Return True if there are still players in the game

    def is_game_over(self):
        """Checks if all players have completed their scoresheets or if no players remain."""
        return len(self.players) == 0 or all(scoresheet.is_complete() for _, scoresheet in self.players)
    
    def get_rankings(self):
        """
        Returns the final rankings of all players.
        Handles ties by giving tied players the same rank and skipping subsequent ranks.
        """
        # Create list of (name, score) tuples
        scores = [(name, sheet.get_grand_total()) for name, sheet in self.players]
        
        # Sort by score in descending order
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # Create rankings with tie handling
        rankings = []
        current_rank = 1
        i = 0
        while i < len(scores):
            current_score = scores[i][1]
            # Find all players with the same score
            tied_players = []
            while i < len(scores) and scores[i][1] == current_score:
                tied_players.append(scores[i][0])
                i += 1
            
            # Add all tied players with the same rank
            for player in tied_players:
                rankings.append((current_rank, player, current_score))
            
            # Skip ranks for tied players
            current_rank += len(tied_players)
        
        return rankings
    
    def display_rankings(self):
        """Displays the final rankings in a formatted way."""
        rankings = self.get_rankings()
        
        print("\n=== Final Rankings ===")
        print("Rank  Player          Score")
        print("-" * 30)
        
        for rank, name, score in rankings:
            print(f"{rank:4d}  {name:<15} {score:5d}")
        print("=" * 30)
        
        # Handle ties in the display
        prev_rank = 1
        for rank, _, _ in rankings:
            if rank != prev_rank + 1 and rank != prev_rank:
                tied_rank = prev_rank
                tied_players = [name for r, name, _ in rankings if r == tied_rank]
                print(f"\nTie for {tied_rank}{self._get_rank_suffix(tied_rank)} place between: {', '.join(tied_players)}")
            prev_rank = rank
    
    def _get_rank_suffix(self, rank):
        """Returns the appropriate suffix for a ranking (1st, 2nd, 3rd, etc.)."""
        if 10 <= rank % 100 <= 20:
            suffix = 'th'
        else:
            suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(rank % 10, 'th')
        return suffix

    def display_all_scores(self):
        """Displays scores for all players, including those who quit."""
        all_players = self.players + self.quit_players
        if not all_players:
            return
        
        # Sort all players by score
        sorted_players = sorted(all_players, key=lambda x: x[1].get_grand_total(), reverse=True)
        
        print("\n=== Final Scores (Including Players Who Quit) ===")
        print(f"{'Player':<20} {'Score':>10} {'Status':>10}")
        print("-" * 42)
        
        for name, scoresheet in sorted_players:
            status = "Active" if (name, scoresheet) in self.players else "Quit"
            score = scoresheet.get_grand_total()
            print(f"{name:<20} {score:>10} {status:>10}")
        print("-" * 42) 