import gymnasium as gym
import numpy as np
from yahtzee_env import YahtzeeEnv
from yahtzee_gym_wrapper import YahtzeeAIGymWrapper

def run_episode(env, agent, render=True):
    """Run a single episode with the given agent"""
    observation, info = env.reset()
    total_reward = 0
    game_score = 0
    done = False
    
    while not done:
        # Get action from agent
        dice_to_reroll, category = agent.act(observation)
        action = (dice_to_reroll, category)
        
        # Take step in environment
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        game_score = info["game_score"]  # Get the actual game score
        done = terminated or truncated
        
        if render:
            env.render()
    
    return game_score, total_reward

def main():
    # Create environment
    env = YahtzeeEnv(render_mode="human")
    
    # Create agents with all available difficulties
    agents = {
        "easy": YahtzeeAIGymWrapper("easy"),
        "medium": YahtzeeAIGymWrapper("medium"),
        "hard": YahtzeeAIGymWrapper("hard"),
        "greedy1": YahtzeeAIGymWrapper("greedy1"),
        "greedy2": YahtzeeAIGymWrapper("greedy2"),
        "greedy3": YahtzeeAIGymWrapper("greedy3"),
        "random": YahtzeeAIGymWrapper("random")
    }
    
    # Run episodes with each agent
    n_episodes = 5
    results = {difficulty: {"game_scores": [], "reward_scores": []} for difficulty in agents.keys()}
    
    # Print header
    print("\n" + "="*60)
    print("Testing Yahtzee AI Agents")
    print("="*60)
    
    for difficulty, agent in agents.items():
        print(f"\nTesting {difficulty.upper()} agent:")
        print("-" * 30)
        for episode in range(n_episodes):
            print(f"Episode {episode + 1}:")
            game_score, reward = run_episode(env, agent)
            results[difficulty]["game_scores"].append(game_score)
            results[difficulty]["reward_scores"].append(reward)
            print(f"Game score: {game_score}")
            print(f"Reward score: {reward:.1f}")
            print("-" * 20)
    
    # Print summary with more detailed statistics
    print("\n" + "="*60)
    print("Results Summary")
    print("="*60)
    
    # Calculate best performing agent based on game scores
    mean_game_scores = {diff: np.mean(scores["game_scores"]) for diff, scores in results.items()}
    best_agent = max(mean_game_scores.items(), key=lambda x: x[1])[0]
    
    for difficulty, scores in results.items():
        game_scores = scores["game_scores"]
        reward_scores = scores["reward_scores"]
        
        # Calculate statistics for both types of scores
        mean_game = np.mean(game_scores)
        std_game = np.std(game_scores)
        min_game = np.min(game_scores)
        max_game = np.max(game_scores)
        
        mean_reward = np.mean(reward_scores)
        std_reward = np.std(reward_scores)
        min_reward = np.min(reward_scores)
        max_reward = np.max(reward_scores)
        
        print(f"\n{difficulty.upper()}:")
        print("-" * 30)
        print("Game Scores:")
        print(f"  Mean score:     {mean_game:.1f}")
        print(f"  Std deviation:  {std_game:.1f}")
        print(f"  Minimum score:  {min_game:.1f}")
        print(f"  Maximum score:  {max_game:.1f}")
        print(f"  All scores:     {game_scores}")
        
        print("\nReward Scores:")
        print(f"  Mean score:     {mean_reward:.1f}")
        print(f"  Std deviation:  {std_reward:.1f}")
        print(f"  Minimum score:  {min_reward:.1f}")
        print(f"  Maximum score:  {max_reward:.1f}")
        print(f"  All scores:     {[f'{x:.1f}' for x in reward_scores]}")
        
        # Highlight if this is the best performing agent
        if difficulty == best_agent:
            print("\n🏆 Best performing agent!")
    
    print("\n" + "="*60)
    print(f"Best performing agent: {best_agent.upper()}")
    print(f"Best mean game score: {mean_game_scores[best_agent]:.1f}")
    print("="*60 + "\n")

if __name__ == "__main__":
    main() 