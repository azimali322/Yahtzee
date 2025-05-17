from game_logic import ScoreSheet, Dice
from yahtzee_ai import YahtzeeAI
import logging
import random
import time
import statistics
import math
from collections import defaultdict
from typing import Dict, List, Tuple, DefaultDict
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import os

# Configure logging for benchmarking
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class YahtzeeBenchmark:
    """Benchmark different Yahtzee AI agents against each other."""
    
    def __init__(self):
        """Initialize the benchmark system."""
        self.results: Dict[str, Dict[str, List[bool]]] = defaultdict(lambda: defaultdict(list))
        self.scores: Dict[str, List[int]] = defaultdict(list)
        self.turn_times: Dict[str, List[float]] = defaultdict(list)
        self.benchmark_agents = ["random", "greedy1", "greedy2"]
        
        # Create output directory if it doesn't exist
        self.output_dir = "benchmark_results"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Generate timestamp for this benchmark run
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def play_game(self, ai1: YahtzeeAI, ai2: YahtzeeAI) -> Tuple[YahtzeeAI, int, int]:
        """Play a single game between two AI agents and return the winner."""
        # Initialize game state
        dice = Dice()
        scoresheet1 = ScoreSheet()
        scoresheet2 = ScoreSheet()
        ai1.set_game_state(scoresheet1, dice)
        ai2.set_game_state(scoresheet2, dice)
        
        # Play 13 rounds (one for each category)
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
        
        # Return winner and scores
        return (ai1 if score1 > score2 else ai2, score1, score2)
    
    def _play_turn(self, ai: YahtzeeAI):
        """Play a single turn for an AI agent."""
        start_time = time.time()
        
        # Reset dice for new turn
        ai.dice.reset_roll_count()
        ai.dice.roll_all()
        current_values = ai.dice.get_values()
        
        # Allow up to 3 rolls
        for roll_num in range(1, 4):
            # Get reroll decision
            reroll_indices = ai.decide_reroll(current_values, roll_num)
            if not reroll_indices:
                break
                
            # Perform reroll
            ai.dice.roll_specific(reroll_indices)
            current_values = ai.dice.get_values()
        
        # Choose category and score
        final_category = ai.choose_category(current_values)
        if final_category:
            ai.scoresheet.record_score(final_category, current_values)
        
        # Record turn time
        turn_time = time.time() - start_time
        self.turn_times[ai.difficulty].append(turn_time)
    
    def run_benchmark(self, num_games: int = 1000) -> Dict[str, Dict[str, float]]:
        """Run benchmark matches between AI agents."""
        logger.info(f"Starting benchmark with {num_games} games per matchup...")
        
        # Test each agent against benchmark agents
        test_agents = ["random", "greedy1", "greedy2", "greedy3", "medium", "easy"]
        
        for agent1 in test_agents:
            for agent2 in self.benchmark_agents:
                if agent1 == agent2:
                    continue
                    
                logger.info(f"\nTesting {agent1} vs {agent2}...")
                for game in range(num_games):
                    # Create fresh AI instances for each game
                    ai1 = YahtzeeAI(agent1)
                    ai2 = YahtzeeAI(agent2)
                    
                    # Play game and record result
                    winner, score1, score2 = self.play_game(ai1, ai2)
                    won = winner == ai1
                    self.results[agent1][agent2].append(won)
                    
                    if (game + 1) % 50 == 0:
                        win_rate = sum(self.results[agent1][agent2]) / len(self.results[agent1][agent2])
                        logger.info(f"Progress: {game + 1}/{num_games} games. Current win rate: {win_rate:.2%}")
        
        return self.get_win_rates()
    
    def get_win_rates(self) -> Dict[str, Dict[str, float]]:
        """Calculate win rates for all matchups."""
        win_rates = {}
        for agent1 in self.results:
            win_rates[agent1] = {}
            for agent2 in self.results[agent1]:
                games = self.results[agent1][agent2]
                win_rate = sum(games) / len(games)
                win_rates[agent1][agent2] = win_rate
        return win_rates
    
    def calculate_confidence_interval(self, data: List[float], confidence: float = 0.95) -> float:
        """Calculate confidence interval for a list of values."""
        if not data:
            return 0.0
        
        n = len(data)
        mean = statistics.mean(data)
        if n < 2:
            return 0.0
            
        std_dev = statistics.stdev(data)
        # Using t-distribution for small sample sizes
        t_value = 1.96  # Approximation for 95% confidence
        return t_value * (std_dev / math.sqrt(n))
    
    def _create_win_rate_heatmap(self, win_rates: Dict[str, Dict[str, float]]):
        """Create a heatmap visualization of win rates."""
        # Convert win rates to DataFrame
        agents = sorted(set(agent for d in win_rates.values() for agent in d.keys()) | set(win_rates.keys()))
        data = []
        for agent1 in agents:
            row = []
            for agent2 in agents:
                if agent1 in win_rates and agent2 in win_rates[agent1]:
                    row.append(win_rates[agent1][agent2] * 100)
                else:
                    row.append(float('nan'))
            data.append(row)
        
        df = pd.DataFrame(data, index=agents, columns=agents)
        
        # Create heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(df, annot=True, fmt='.1f', cmap='RdYlGn',
                    vmin=0, vmax=100, center=50,
                    xticklabels=agents, yticklabels=agents)
        plt.title('Win Rates (%)')
        plt.xlabel('Opponent')
        plt.ylabel('Agent')
        
        # Save plot
        plt.savefig(f"{self.output_dir}/win_rates_heatmap_{self.timestamp}.png")
        plt.close()

    def _create_score_distribution_plot(self):
        """Create a box plot of score distributions."""
        # Convert scores to DataFrame
        data = []
        for agent, scores in self.scores.items():
            data.extend([(agent, score) for score in scores])
        
        df = pd.DataFrame(data, columns=['Agent', 'Score'])
        
        # Create box plot
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='Agent', y='Score', data=df)
        plt.title('Score Distribution by Agent')
        plt.xticks(rotation=45)
        
        # Save plot
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/score_distribution_{self.timestamp}.png")
        plt.close()

    def _create_performance_plot(self):
        """Create a bar plot of average turn times."""
        # Calculate average turn times
        avg_times = {agent: sum(times) / len(times) * 1000 for agent, times in self.turn_times.items()}
        
        # Convert to DataFrame
        df = pd.DataFrame(list(avg_times.items()), columns=['Agent', 'Average Time (ms)'])
        
        # Create bar plot
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Agent', y='Average Time (ms)', data=df)
        plt.title('Average Turn Time by Agent')
        plt.xticks(rotation=45)
        
        # Save plot
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/turn_times_{self.timestamp}.png")
        plt.close()

    def save_results_to_csv(self, win_rates: Dict[str, Dict[str, float]]):
        """Save benchmark results to CSV files."""
        # Save win rates
        win_rate_data = []
        for agent1 in win_rates:
            for agent2 in win_rates[agent1]:
                win_rate_data.append({
                    'Agent': agent1,
                    'Opponent': agent2,
                    'Win Rate': win_rates[agent1][agent2]
                })
        pd.DataFrame(win_rate_data).to_csv(f"{self.output_dir}/win_rates_{self.timestamp}.csv", index=False)
        
        # Save scores
        score_data = []
        for agent in self.scores:
            for score in self.scores[agent]:
                score_data.append({
                    'Agent': agent,
                    'Score': score
                })
        pd.DataFrame(score_data).to_csv(f"{self.output_dir}/scores_{self.timestamp}.csv", index=False)
        
        # Save turn times
        time_data = []
        for agent in self.turn_times:
            for time_ms in self.turn_times[agent]:
                time_data.append({
                    'Agent': agent,
                    'Time (ms)': time_ms * 1000
                })
        pd.DataFrame(time_data).to_csv(f"{self.output_dir}/turn_times_{self.timestamp}.csv", index=False)

    def print_results(self):
        """Print benchmark results in formatted tables and create visualizations."""
        win_rates = self.get_win_rates()
        
        # Print win rates table
        print("\nBenchmark Results - Win Rates")
        print("-" * 60)
        print(f"{'Agent':15} | {'Opponent':10} | {'Win Rate':10} | {'Games':8}")
        print("-" * 60)
        
        for agent in win_rates:
            for opponent in win_rates[agent]:
                games = len(self.results[agent][opponent])
                win_rate = win_rates[agent][opponent]
                print(f"{agent:15} | {opponent:10} | {win_rate:9.2%} | {games:8}")
        
        print("-" * 60)
        
        # Print performance statistics table
        print("\nPerformance Statistics")
        print("-" * 100)
        print(f"{'Agent':15} | {'Avg Score':10} | {'95% CI':15} | {'Avg Time/Turn':12} | {'Games Played':12}")
        print("-" * 100)
        
        all_agents = set(self.scores.keys())
        for agent in sorted(all_agents):
            scores = self.scores[agent]
            times = self.turn_times[agent]
            
            if not scores or not times:
                continue
                
            avg_score = statistics.mean(scores)
            score_ci = self.calculate_confidence_interval(scores)
            avg_time = statistics.mean(times) * 1000  # Convert to milliseconds
            games_played = len(scores)
            
            print(f"{agent:15} | {avg_score:10.1f} | ±{score_ci:13.1f} | {avg_time:10.1f}ms | {games_played:12}")
        
        print("-" * 100)
        
        # Create and save visualizations
        self._create_win_rate_heatmap(win_rates)
        self._create_score_distribution_plot()
        self._create_performance_plot()
        
        # Save results to CSV
        self.save_results_to_csv(win_rates)
        
        # Print information about saved files
        print(f"\nResults have been saved to the '{self.output_dir}' directory with timestamp {self.timestamp}")
        print("Generated files:")
        print(f"- win_rates_heatmap_{self.timestamp}.png")
        print(f"- score_distribution_{self.timestamp}.png")
        print(f"- turn_times_{self.timestamp}.png")
        print(f"- win_rates_{self.timestamp}.csv")
        print(f"- scores_{self.timestamp}.csv")
        print(f"- turn_times_{self.timestamp}.csv")

if __name__ == "__main__":
    # Run benchmark
    benchmark = YahtzeeBenchmark()
    benchmark.run_benchmark(num_games=1000)
    benchmark.print_results() 