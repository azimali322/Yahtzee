import gymnasium as gym
import numpy as np
from yahtzee_env import YahtzeeEnv
from yahtzee_gym_wrapper import YahtzeeAIGymWrapper

def run_episode(env, agent, render=True):
    """Run a single episode with the given agent"""
    observation, info = env.reset()
    total_reward = 0
    done = False
    
    while not done:
        # Get action from agent
        dice_to_reroll, category = agent.act(observation)
        action = (dice_to_reroll, category)
        
        # Take step in environment
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated
        
        if render:
            env.render()
    
    return total_reward

def main():
    # Create environment
    env = YahtzeeEnv(render_mode="human")
    
    # Create agents with different difficulties
    agents = {
        "easy": YahtzeeAIGymWrapper("easy"),
        "medium": YahtzeeAIGymWrapper("medium"),
        "hard": YahtzeeAIGymWrapper("hard")
    }
    
    # Run episodes with each agent
    n_episodes = 5
    results = {difficulty: [] for difficulty in agents.keys()}
    
    for difficulty, agent in agents.items():
        print(f"\nTesting {difficulty.upper()} agent:")
        for episode in range(n_episodes):
            print(f"\nEpisode {episode + 1}:")
            reward = run_episode(env, agent)
            results[difficulty].append(reward)
            print(f"Total score: {reward}")
    
    # Print summary
    print("\nResults Summary:")
    print("-" * 40)
    for difficulty, scores in results.items():
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        print(f"{difficulty.upper()}:")
        print(f"  Mean score: {mean_score:.1f}")
        print(f"  Std dev: {std_score:.1f}")
        print(f"  Scores: {scores}")
        print("-" * 40)

if __name__ == "__main__":
    main() 