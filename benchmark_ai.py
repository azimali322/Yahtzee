from game_logic import ScoreSheet, Dice
from yahtzee_ai import YahtzeeAI, RLAgent
from pretrain_rl_agent import PreTrainer
import logging
import random
import time
import statistics
from collections import defaultdict
from typing import Dict, List, Tuple
import os
import pandas as pd
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import torch
from pathlib import Path

# Configure logging for benchmarking
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_pretrained_agent(checkpoint_path: str = "checkpoints/yahtzee_pretrain_faster/best_model.pt") -> RLAgent:
    """Load the pretrained RL agent."""
    # Create observation space
    observation_space = {
        "dice": np.zeros(5, dtype=np.float32),
        "roll_number": np.array([0], dtype=np.float32),
        "scoresheet": np.zeros(13, dtype=np.float32),
        "upper_bonus": np.array([0], dtype=np.float32),
        "yahtzee_bonus": np.array([0], dtype=np.float32),
        "opponent_scores": np.array([0], dtype=np.float32),
        "relative_rank": np.array([0], dtype=np.float32)
    }
    
    # Initialize agent
    device = "cuda" if torch.cuda.is_available() else "cpu"
    agent = RLAgent(observation_space=observation_space, device=device)
    
    # Load checkpoint with weights_only=False for PyTorch 2.6 compatibility
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        agent.ac_net.load_state_dict(checkpoint['model_state_dict'])
        agent.ac_net.eval()  # Set to evaluation mode
        logger.info(f"Successfully loaded checkpoint from {checkpoint_path}")
        return agent
    except Exception as e:
        logger.error(f"Error loading checkpoint: {str(e)}")
        raise

class YahtzeeBenchmark:
    """Benchmark Yahtzee AI agents."""
    
    def __init__(self):
        """Initialize the benchmark system."""
        self.results = defaultdict(lambda: defaultdict(list))
        self.scores = defaultdict(list)
        self.turn_times = defaultdict(list)
        
        # Create output directory
        self.output_dir = "benchmark_results"
        os.makedirs(self.output_dir, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def play_game(self, ai1: YahtzeeAI, ai2: YahtzeeAI) -> Tuple[YahtzeeAI, int, int]:
        """Play a single game between two AI agents."""
        dice = Dice()
        scoresheet1 = ScoreSheet()
        scoresheet2 = ScoreSheet()
        ai1.set_game_state(scoresheet1, dice)
        ai2.set_game_state(scoresheet2, dice)
        
        # Play 13 rounds
        for _ in range(13):
            # Player 1's turn
            self._play_turn(ai1)
            # Player 2's turn
            self._play_turn(ai2)
        
        # Calculate final scores
        score1 = scoresheet1.get_grand_total()
        score2 = scoresheet2.get_grand_total()
        
        # Record scores
        self.scores[ai1.difficulty].append(score1)
        self.scores[ai2.difficulty].append(score2)
        
        return (ai1 if score1 > score2 else ai2, score1, score2)
    
    def _play_turn(self, ai: YahtzeeAI):
        """Play a single turn for an AI agent."""
        start_time = time.time()
        
        # Reset dice for new turn
        ai.dice.reset_roll_count()
        ai.dice.roll_all()
        
        # Make decisions until turn is complete
        while ai.dice.roll_count < 3:
            current_values = ai.dice.get_values()
            reroll_indices = ai.decide_reroll(current_values, ai.dice.roll_count)
            if not reroll_indices:
                break
            ai.dice.roll_specific(reroll_indices)
        
        # Choose category
        current_values = ai.dice.get_values()
        category = ai.choose_category(current_values)
        if category:
            ai.scoresheet.record_score(category, current_values)
        
        # Record turn time
        turn_time = time.time() - start_time
        self.turn_times[ai.difficulty].append(turn_time)
    
    def run_benchmark(self, num_games: int = 100) -> Dict[str, Dict[str, float]]:
        """Run benchmark comparing Greedy3 vs Pretrained RL agent."""
        logger.info(f"\nStarting benchmark with {num_games} games...")
        start_time = time.time()
        
        # Load agents
        pretrained_agent = load_pretrained_agent()
        greedy3_agent = YahtzeeAI("greedy3")
        
        # Play games
        for game in range(num_games):
            # Play one game with each agent going first
            for first_agent, second_agent in [(greedy3_agent, pretrained_agent), 
                                            (pretrained_agent, greedy3_agent)]:
                winner, score1, score2 = self.play_game(first_agent, second_agent)
                
                # Record results
                if winner == first_agent:
                    self.results[first_agent.difficulty][second_agent.difficulty].append(True)
                    self.results[second_agent.difficulty][first_agent.difficulty].append(False)
                else:
                    self.results[first_agent.difficulty][second_agent.difficulty].append(False)
                    self.results[second_agent.difficulty][first_agent.difficulty].append(True)
            
            # Print progress
            games_played = (game + 1) * 2
            total_games = num_games * 2
            elapsed_time = time.time() - start_time
            games_per_second = games_played / elapsed_time
            estimated_total_time = total_games / games_per_second
            remaining_time = estimated_total_time - elapsed_time
            
            print(f"\rProgress: {games_played}/{total_games} games "
                  f"({(games_played/total_games)*100:.1f}%) - "
                  f"Elapsed: {elapsed_time:.1f}s - "
                  f"Remaining: {remaining_time:.1f}s", end="")
        
        print("\nBenchmark completed!")
        
        # Calculate statistics
        stats = self.calculate_statistics()
        self.print_statistics(stats)
        self.save_statistics(stats)
        
        return stats
    
    def calculate_statistics(self) -> Dict:
        """Calculate benchmark statistics."""
        stats = {}
        
        # Calculate win rates
        for agent1 in self.results:
            stats[agent1] = {}
            for agent2 in self.results[agent1]:
                wins = sum(self.results[agent1][agent2])
                total = len(self.results[agent1][agent2])
                stats[agent1][agent2] = {
                    'win_rate': wins / total if total > 0 else 0,
                    'games_played': total
                }
        
        # Calculate score statistics
        for agent in self.scores:
            if self.scores[agent]:
                stats[agent]['score_stats'] = {
                    'mean': statistics.mean(self.scores[agent]),
                    'std': statistics.stdev(self.scores[agent]) if len(self.scores[agent]) > 1 else 0,
                    'min': min(self.scores[agent]),
                    'max': max(self.scores[agent])
                }
        
        # Calculate timing statistics
        for agent in self.turn_times:
            if self.turn_times[agent]:
                stats[agent]['time_stats'] = {
                    'mean_turn_time': statistics.mean(self.turn_times[agent]),
                    'std_turn_time': statistics.stdev(self.turn_times[agent]) if len(self.turn_times[agent]) > 1 else 0
                }
        
        return stats
    
    def print_statistics(self, stats: Dict):
        """Print benchmark statistics."""
        print("\nBenchmark Results:")
        print("=" * 50)
        
        # Print win rates
        print("\nWin Rates:")
        for agent1 in stats:
            for agent2 in stats[agent1]:
                if 'win_rate' in stats[agent1][agent2]:
                    win_rate = stats[agent1][agent2]['win_rate']
                    games = stats[agent1][agent2]['games_played']
                    print(f"{agent1} vs {agent2}: {win_rate:.1%} ({games} games)")
        
        # Print score statistics
        print("\nScore Statistics:")
        for agent in stats:
            if 'score_stats' in stats[agent]:
                ss = stats[agent]['score_stats']
                print(f"\n{agent}:")
                print(f"  Mean Score: {ss['mean']:.1f} ± {ss['std']:.1f}")
                print(f"  Range: {ss['min']} - {ss['max']}")
        
        # Print timing statistics
        print("\nTiming Statistics:")
        for agent in stats:
            if 'time_stats' in stats[agent]:
                ts = stats[agent]['time_stats']
                print(f"\n{agent}:")
                print(f"  Mean Turn Time: {ts['mean_turn_time']*1000:.1f} ms ± {ts['std_turn_time']*1000:.1f} ms")
    
    def save_statistics(self, stats: Dict):
        """Save benchmark statistics to file."""
        output_file = os.path.join(self.output_dir, f"benchmark_results_{self.timestamp}.txt")
        with open(output_file, 'w') as f:
            f.write("Yahtzee AI Benchmark Results\n")
            f.write("=" * 50 + "\n\n")
            
            # Write win rates
            f.write("Win Rates:\n")
            for agent1 in stats:
                for agent2 in stats[agent1]:
                    if 'win_rate' in stats[agent1][agent2]:
                        win_rate = stats[agent1][agent2]['win_rate']
                        games = stats[agent1][agent2]['games_played']
                        f.write(f"{agent1} vs {agent2}: {win_rate:.1%} ({games} games)\n")
            
            # Write score statistics
            f.write("\nScore Statistics:\n")
            for agent in stats:
                if 'score_stats' in stats[agent]:
                    ss = stats[agent]['score_stats']
                    f.write(f"\n{agent}:\n")
                    f.write(f"  Mean Score: {ss['mean']:.1f} ± {ss['std']:.1f}\n")
                    f.write(f"  Range: {ss['min']} - {ss['max']}\n")
            
            # Write timing statistics
            f.write("\nTiming Statistics:\n")
            for agent in stats:
                if 'time_stats' in stats[agent]:
                    ts = stats[agent]['time_stats']
                    f.write(f"\n{agent}:\n")
                    f.write(f"  Mean Turn Time: {ts['mean_turn_time']*1000:.1f} ms ± {ts['std_turn_time']*1000:.1f} ms\n")
        
        logger.info(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    # Run benchmark
    benchmark = YahtzeeBenchmark()
    benchmark.run_benchmark() 