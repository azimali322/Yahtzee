from game_logic import ScoreSheet, Dice
from yahtzee_ai import YahtzeeAI
import logging
import random
import time
import statistics
import math
from collections import defaultdict
from typing import Dict, List, Tuple, DefaultDict
import os
import pandas as pd
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import select
import termios
import tty
import numpy as np

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
        self.benchmark_agents = ["random", "greedy1", "greedy2", "greedy3", "medium", "easy", "hard"]
        self.stop_requested = False
        self.show_live_plots = False  # Default to no live plots
        
        # Create output directory if it doesn't exist
        self.output_dir = "benchmark_results"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Generate timestamp for this benchmark run
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Store original terminal settings
        self.old_settings = termios.tcgetattr(sys.stdin)
        
        # Initialize plot figures
        plt.ion()  # Turn on interactive mode
        
        # Create initial figures with a non-blocking backend
        plt.switch_backend('TkAgg')  # Use TkAgg backend for better performance
        
        # Initialize heatmap with dummy data
        self.fig_heatmap = plt.figure(figsize=(10, 8))
        self.ax_heatmap = self.fig_heatmap.add_subplot(111)
        dummy_data = pd.DataFrame(np.zeros((len(self.benchmark_agents), len(self.benchmark_agents))),
                                index=self.benchmark_agents,
                                columns=self.benchmark_agents)
        self.heatmap = sns.heatmap(dummy_data, annot=True, fmt='.1f', cmap='RdYlGn',
                                  vmin=0, vmax=100, center=50,
                                  xticklabels=self.benchmark_agents,
                                  yticklabels=self.benchmark_agents,
                                  ax=self.ax_heatmap, cbar=False)
        self.ax_heatmap.set_title('Win Rates (%)')
        self.ax_heatmap.set_xlabel('Opponent')
        self.ax_heatmap.set_ylabel('Agent')
        
        self.fig_scores = plt.figure(figsize=(12, 6))
        self.ax_scores = self.fig_scores.add_subplot(111)
        self.ax_scores.set_title('Score Distribution by Agent')
        
        self.fig_times = plt.figure(figsize=(10, 6))
        self.ax_times = self.fig_times.add_subplot(111)
        self.ax_times.set_title('Average Turn Time by Agent')
        
        # Show all figures
        for fig in [self.fig_heatmap, self.fig_scores, self.fig_times]:
            fig.show()
            
    def get_num_games(self) -> int:
        """Get the number of games to run from user input."""
        while True:
            try:
                num_games = input("Enter the number of games to run for each matchup (default is 10): ").strip()
                if not num_games:  # If user just hits enter, use default
                    return 10
                num_games = int(num_games)
                if num_games <= 0:
                    print("Please enter a positive number.")
                    continue
                return num_games
            except ValueError:
                print("Please enter a valid number.")
    
    def ask_print_after_benchmark(self) -> bool:
        """Ask user if they want to print results after benchmark completion."""
        print("\nWould you like to see the results now? (y/n, default is y): ", end='', flush=True)
        
        # Store original terminal settings
        old_settings = termios.tcgetattr(sys.stdin)
        try:
            # Set terminal to raw mode
            tty.setraw(sys.stdin.fileno())
            
            # Wait for input with timeout
            if select.select([sys.stdin], [], [], 30.0)[0]:  # 30 second timeout
                # Read the input
                choice = sys.stdin.read(1).strip().lower()
                # Print newline after input
                print()
                if choice == 'n':
                    return False
            else:
                # If no input received within timeout
                print("\nNo input received within 30 seconds. Showing results...")
                return True
                
            return True
            
        except Exception as e:
            logger.error(f"Error getting user input: {str(e)}")
            return True
            
        finally:
            # Always restore terminal settings
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
    
    def ask_keep_partial_results(self) -> bool:
        """Ask user if they want to keep partial results after early stopping."""
        while True:
            choice = input("\nWould you like to keep the partial benchmark results? (y/n, default is y): ").strip().lower()
            if not choice or choice == 'y':
                return True
            if choice == 'n':
                return False
            print("Please enter 'y' or 'n'")
    
    def ask_live_plots(self) -> bool:
        """Ask user if they want live plot updates during the benchmark."""
        while True:
            choice = input("\nWould you like to see live plot updates during the benchmark? (y/n, default is n): ").strip().lower()
            if not choice or choice == 'n':
                return False
            if choice == 'y':
                return True
            print("Please enter 'y' or 'n'")
    
    def format_time(self, seconds: float) -> str:
        """Format time in seconds to minutes, seconds, and milliseconds."""
        minutes = int(seconds // 60)
        seconds_remaining = seconds % 60
        whole_seconds = int(seconds_remaining)
        milliseconds = int((seconds_remaining - whole_seconds) * 1000)
        
        time_parts = []
        if minutes > 0:
            time_parts.append(f"{minutes} minute{'s' if minutes != 1 else ''}")
        if whole_seconds > 0 or (minutes == 0 and milliseconds == 0):
            time_parts.append(f"{whole_seconds} second{'s' if whole_seconds != 1 else ''}")
        if milliseconds > 0:
            time_parts.append(f"{milliseconds} millisecond{'s' if milliseconds != 1 else ''}")
        
        return " and ".join(time_parts)
    
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
    
    def update_plots(self):
        """Update all plots with current data."""
        try:
            # Update win rates heatmap
            win_rates = self.get_win_rates()
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
            
            df_heatmap = pd.DataFrame(data, index=agents, columns=agents)
            
            # Update existing heatmap data
            self.ax_heatmap.clear()
            self.heatmap = sns.heatmap(df_heatmap, annot=True, fmt='.1f', cmap='RdYlGn',
                                     vmin=0, vmax=100, center=50,
                                     xticklabels=agents, yticklabels=agents,
                                     ax=self.ax_heatmap, cbar=False)  # Disable colorbar
            self.ax_heatmap.set_title('Win Rates (%)')
            self.ax_heatmap.set_xlabel('Opponent')
            self.ax_heatmap.set_ylabel('Agent')
            
            # Update score distribution
            self.ax_scores.clear()
            if self.scores:
                data = []
                medians = {}  # Store median scores for each agent
                for agent, scores in self.scores.items():
                    data.extend([(agent, score) for score in scores])
                    medians[agent] = statistics.median(scores)
                
                df_scores = pd.DataFrame(data, columns=['Agent', 'Score'])
                sns.boxplot(x='Agent', y='Score', data=df_scores, ax=self.ax_scores)
                
                # Add median values above each box
                for i, agent in enumerate(sorted(medians.keys())):
                    median = medians[agent]
                    self.ax_scores.text(i, median, f'Median: {median:.1f}',
                                      horizontalalignment='center',
                                      verticalalignment='bottom')
                
                self.ax_scores.set_title('Score Distribution by Agent')
                self.ax_scores.tick_params(axis='x', rotation=45)
            
            # Update turn times
            self.ax_times.clear()
            if self.turn_times:
                avg_times = {agent: sum(times) / len(times) * 1000 
                            for agent, times in self.turn_times.items()}
                df_times = pd.DataFrame(list(avg_times.items()), 
                                      columns=['Agent', 'Average Time (ms)'])
                
                # Create bar plot
                bars = sns.barplot(x='Agent', y='Average Time (ms)', data=df_times, 
                                 ax=self.ax_times)
                
                # Add average time values above each bar
                for i, bar in enumerate(bars.patches):
                    height = bar.get_height()
                    self.ax_times.text(bar.get_x() + bar.get_width()/2., height,
                                     f'{height:.1f}ms',
                                     ha='center', va='bottom')
                
                self.ax_times.set_title('Average Turn Time by Agent')
                self.ax_times.tick_params(axis='x', rotation=45)
            
            # Adjust layouts and draw
            for fig in [self.fig_heatmap, self.fig_scores, self.fig_times]:
                fig.tight_layout()
                fig.canvas.draw_idle()
                fig.canvas.flush_events()
            
            # Add a small pause to allow GUI to update
            plt.pause(0.01)
                
        except Exception as e:
            logger.error(f"Error updating plots: {str(e)}")
            
    def check_for_stop(self) -> bool:
        """Check if user has pressed 'Q' to stop the benchmark."""
        try:
            # Check for input with a very short timeout
            if select.select([sys.stdin], [], [], 0.1)[0]:  # 100ms timeout
                # Temporarily restore normal terminal mode
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)
                
                # Read the input
                key = sys.stdin.read(1)
                
                # Set back to raw mode
                tty.setraw(sys.stdin.fileno())
                
                if key.lower() == 'q':
                    # Print a newline to avoid terminal corruption
                    print("\n")
                    return True
                    
            return False
            
        except Exception as e:
            logger.error(f"Error checking for stop: {str(e)}")
            return False
            
    def run_benchmark(self, num_games: int = None, print_results: bool = None) -> Dict[str, Dict[str, float]]:
        """Run benchmark matches between AI agents."""
        if num_games is None:
            num_games = self.get_num_games()
            
        # Ask about live plots
        self.show_live_plots = self.ask_live_plots()
        
        print("\nBenchmark starting...")
        print("To stop the benchmark early, press 'Q' at any time.")
        print("The benchmark will complete the current game and ask if you want to keep the partial results.\n")
            
        logger.info(f"Starting benchmark with {num_games} games per matchup...")
        start_time = time.time()
        
        total_games = 0
        total_matchups = len(self.benchmark_agents) * len(self.benchmark_agents)
        # Update every 20% of total games or at least every 50 games
        update_frequency = max(50, min(100, (num_games * total_matchups) // 5))
        
        # Set terminal to raw mode
        tty.setraw(sys.stdin.fileno())
        
        try:
            # Use the same list of agents for both players
            for agent1 in self.benchmark_agents:
                for agent2 in self.benchmark_agents:
                    # Include self-play matches
                    logger.info(f"\nTesting {agent1} vs {agent2}...")
                    for game in range(num_games):
                        # Check for stop request
                        if self.check_for_stop():
                            # Restore terminal settings before printing
                            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)
                            print("\nBenchmark stopped by user.")
                            raise StopIteration
                            
                        # Create fresh AI instances for each game
                        ai1 = YahtzeeAI(agent1)
                        ai2 = YahtzeeAI(agent2)
                        
                        # Play game and record result
                        winner, score1, score2 = self.play_game(ai1, ai2)
                        won = winner == ai1
                        self.results[agent1][agent2].append(won)
                        
                        total_games += 1
                        if total_games % update_frequency == 0:
                            # Restore terminal settings before printing
                            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)
                            win_rate = sum(self.results[agent1][agent2]) / len(self.results[agent1][agent2])
                            progress = (total_games / (num_games * total_matchups)) * 100
                            logger.info(f"Overall Progress: {progress:.1f}% ({total_games}/{num_games * total_matchups} games)")
                            logger.info(f"Current matchup ({agent1} vs {agent2}): {game + 1}/{num_games} games. Win rate: {win_rate:.2%}")
                            
                            # Update plots if requested
                            if self.show_live_plots:
                                self.update_plots()
                            
                            # Set back to raw mode
                            tty.setraw(sys.stdin.fileno())
                            
        except StopIteration:
            end_time = time.time()
            total_time = end_time - start_time
            logger.info(f"Stopped after {self.format_time(total_time)}")
            
            # Final plot update
            if self.show_live_plots:
                self.update_plots()
            
            # Ask if user wants to keep partial results
            if not self.ask_keep_partial_results():
                # Clear all data and close plots
                self.results.clear()
                self.scores.clear()
                self.turn_times.clear()
                plt.close('all')
                return {}
                
        except Exception as e:
            logger.error(f"Error during benchmark: {str(e)}")
            plt.close('all')  # Ensure plots are closed on error
            raise
            
        finally:
            # Always restore terminal settings
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)
            
        # Normal completion
        end_time = time.time()
        total_time = end_time - start_time
        logger.info(f"\nBenchmark completed in {self.format_time(total_time)}")
        
        # Final plot update
        self.update_plots()
        
        win_rates = self.get_win_rates()
        
        # If print_results wasn't explicitly set, ask user after benchmark
        if print_results is None:
            if self.ask_print_after_benchmark():
                self.save_final_plots()
                self.print_results()
        elif print_results:
            self.save_final_plots()
            self.print_results()
        
        # Close all plot windows
        plt.close('all')
        
        return win_rates
    
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
    
    def save_final_plots(self):
        """Save the final versions of all plots."""
        self.fig_heatmap.savefig(f"{self.output_dir}/win_rates_heatmap_{self.timestamp}.png")
        self.fig_scores.savefig(f"{self.output_dir}/score_distribution_{self.timestamp}.png")
        self.fig_times.savefig(f"{self.output_dir}/turn_times_{self.timestamp}.png")

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
        self.update_plots()
        
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

if __name__ == "__main__":
    # Run benchmark
    benchmark = YahtzeeBenchmark()
    benchmark.run_benchmark()  # Will only prompt for number of games and ask about printing after completion
    
    # If results weren't printed during benchmark, give another chance to print them
    if not any(benchmark.results.values()):  # Check if results exist but weren't printed
        if benchmark.ask_print_after_benchmark():
            benchmark.print_results() 