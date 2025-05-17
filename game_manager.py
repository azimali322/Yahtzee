import random
from game_logic import ScoreSheet, Dice
from yahtzee_ai import YahtzeeAI

# List of 100 random English names
RANDOM_NAMES = [
    "Oliver", "Emma", "Liam", "Ava", "Noah", "Isabella", "William", "Sophia", "James", "Mia",
    "Benjamin", "Charlotte", "Lucas", "Amelia", "Henry", "Harper", "Theodore", "Evelyn", "Jack",
    "Abigail", "Alexander", "Emily", "Sebastian", "Elizabeth", "Michael", "Sofia", "Daniel",
    "Avery", "Matthew", "Ella", "Samuel", "Scarlett", "Joseph", "Victoria", "David", "Grace",
    "John", "Chloe", "Owen", "Camila", "Dylan", "Penelope", "Luke", "Riley", "Isaac", "Layla",
    "Gabriel", "Zoey", "Julian", "Nora", "Christopher", "Lily", "Joshua", "Eleanor", "Andrew",
    "Hannah", "Lincoln", "Lillian", "Ryan", "Addison", "Nathan", "Aubrey", "Adrian", "Ellie",
    "Christian", "Stella", "Maverick", "Natalie", "Caleb", "Zoe", "Adam", "Leah", "Thomas",
    "Hazel", "Robert", "Violet", "Elijah", "Aurora", "Nicholas", "Savannah", "Charles", "Audrey",
    "Ezra", "Brooklyn", "Austin", "Bella", "Hudson", "Claire", "Cooper", "Lucy", "Xavier",
    "Anna", "Marcus", "Caroline", "Parker", "Sarah", "Roman", "Alice", "Leo", "Eva", "Max",
    "Madeline", "Eric", "Ruby", "Kevin", "Autumn"
]

def get_random_name():
    """Returns a random name from the list of names, ensuring it hasn't been used yet."""
    available_names = [name for name in RANDOM_NAMES if name not in [p[0] for p in GameManager.active_players]]
    if not available_names:
        # If all names are used, append a number to a random name
        base_name = random.choice(RANDOM_NAMES)
        counter = 1
        while f"{base_name}_{counter}" in [p[0] for p in GameManager.active_players]:
            counter += 1
        return f"{base_name}_{counter}"
    return random.choice(available_names)

class GameManager:
    """Manages a multiplayer Yahtzee game."""
    
    active_players = []  # Class variable to track all active players
    
    def __init__(self, num_players=1):
        self.num_players = max(1, min(10, num_players))  # Limit to 1-10 players
        self.dice = Dice()
        self.current_player_idx = 0
        self.players = []  # List of (name, scoresheet, ai_agent) tuples, ai_agent is None for human players
        self.quit_players = []  # List of (name, scoresheet, ai_agent) tuples for players who quit
        self.human_players_count = 0  # Track number of human players
    
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
        
        print("\nEnter player names (press Enter for random name, 'ai:easy', 'ai:medium', or 'ai:hard' for AI players):")
        for i in range(num_players):
            name = input(f"Player {i + 1}: ").strip().lower()
            
            # Check for AI player
            ai_agent = None
            if name.startswith('ai:'):
                difficulty = name.split(':')[1] if len(name.split(':')) > 1 else 'medium'
                if difficulty not in ['easy', 'medium', 'hard']:
                    difficulty = 'medium'
                ai_agent = YahtzeeAI(difficulty)
                name = None  # Will be assigned a random name
            else:
                self.human_players_count += 1
            
            # Get or assign name
            assigned_name = self.add_player(name, ai_agent)
            if not name or name.startswith('ai:'):
                print(f"Assigned name: {assigned_name}")
            
        return self.players
    
    def next_turn(self):
        """Advances to the next player's turn."""
        self.current_player_idx = (self.current_player_idx + 1) % len(self.players)
        self.dice.reset_roll_count()
        return self.get_current_player()
    
    def get_current_player(self):
        """Returns the current player's name, scoresheet, and AI agent (if any)."""
        return self.players[self.current_player_idx]
    
    def remove_player(self, player_name):
        """
        Removes a human player from the game and adds them to quit_players list.
        Returns True if game should continue, False if game should end.
        """
        # Find the player to remove
        player_to_remove = None
        for player in self.players:
            if player[0] == player_name:
                player_to_remove = player
                break
        
        if player_to_remove:
            # Only remove if it's a human player
            if player_to_remove[2] is None:  # Check if not an AI player
                self.quit_players.append(player_to_remove)
                self.players = [(name, sheet, ai) for name, sheet, ai in self.players if name != player_name]
                if self.current_player_idx >= len(self.players):
                    self.current_player_idx = 0
                self.human_players_count -= 1
        
        # Return True if there are still human players or if game should continue
        return self.human_players_count > 0
    
    def has_human_players(self):
        """Returns True if there are still human players in the game."""
        return self.human_players_count > 0
    
    def is_game_over(self, allow_final_round=False):
        """
        Checks if the game should end:
        - All human players have quit AND final round is complete (if allow_final_round is True), OR
        - All remaining players (human and AI) have completed their scoresheets
        """
        if self.human_players_count == 0 and not allow_final_round:
            return True
        return all(scoresheet.is_complete() for _, scoresheet, _ in self.players)
    
    def get_rankings(self):
        """
        Returns the final rankings of all players.
        Handles ties by giving tied players the same rank and skipping subsequent ranks.
        """
        # Create list of (name, score) tuples
        scores = [(name, sheet.get_grand_total()) for name, sheet, _ in self.players]
        
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
        
        for name, scoresheet, ai_agent in sorted_players:
            status = "Active" if (name, scoresheet, ai_agent) in self.players else "Quit"
            score = scoresheet.get_grand_total()
            print(f"{name:<20} {score:>10} {status:>10}")
        print("-" * 42)

    def add_player(self, name=None, ai_agent=None):
        """Add a new player to the game. If no name provided, assign a random name."""
        if not name:
            name = get_random_name()
        
        # Ensure unique name
        counter = 1
        original_name = name
        while name in [p[0] for p in self.players]:
            name = f"{original_name}_{counter}"
            counter += 1
        
        scoresheet = ScoreSheet()
        if ai_agent:
            ai_agent.set_game_state(scoresheet, self.dice)
        
        player = (name, scoresheet, ai_agent)
        self.players.append(player)
        GameManager.active_players.append(player)
        return name
    
    def is_current_player_ai(self):
        """Returns True if the current player is an AI agent."""
        return self.players[self.current_player_idx][2] is not None
    
    def get_ai_reroll_decision(self):
        """Gets the AI's decision about which dice to reroll."""
        _, _, ai_agent = self.players[self.current_player_idx]
        if ai_agent:
            return ai_agent.decide_reroll(self.dice.get_values(), self.dice.roll_count + 1)
        return []
    
    def get_ai_category_choice(self):
        """Gets the AI's decision about which category to score."""
        _, _, ai_agent = self.players[self.current_player_idx]
        if ai_agent:
            return ai_agent.choose_category(self.dice.get_values())
        return None 