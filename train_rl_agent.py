import torch
import numpy as np
from yahtzee_ai import RLAgent
from game_logic import ScoreSheet, Dice, ALL_CATEGORIES
from collections import deque
import logging
import time
from typing import List, Tuple, Dict
import json
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import datetime
import shutil
import os
import torch.optim as optim

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RewardScaler:
    """Implements reward scaling using running statistics."""
    def __init__(self, gamma: float = 0.99, epsilon: float = 1e-8):
        self.gamma = gamma
        self.epsilon = epsilon
        self.running_mean = 0
        self.running_variance = 0
        self.count = 0
    
    def update(self, reward: float) -> None:
        """Update running statistics with new reward."""
        self.count += 1
        delta = reward - self.running_mean
        self.running_mean += delta / self.count
        delta2 = reward - self.running_mean
        self.running_variance += delta * delta2
    
    def scale(self, reward: float) -> float:
        """Scale the reward using running statistics."""
        if self.count < 2:
            return reward
        std = np.sqrt(self.running_variance / (self.count - 1) + self.epsilon)
        return (reward - self.running_mean) / std

    def get_stats(self) -> Dict[str, float]:
        """Get current statistics."""
        return {
            'mean': self.running_mean,
            'std': np.sqrt(self.running_variance / max(1, self.count - 1) + self.epsilon),
            'count': self.count
        }

class YahtzeeTrainer:
    def __init__(
        self,
        n_episodes: int = 10000,
        batch_size: int = 32,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        entropy_coef: float = 0.01,
        value_loss_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        eval_frequency: int = 100,
        eval_episodes: int = 10,
        checkpoint_frequency: int = 1000,  # Save checkpoint every N episodes
        max_checkpoints: int = 5,          # Maximum number of checkpoints to keep
        experiment_name: str = None,       # For organizing runs in tensorboard
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        pretrained_model_path: str = None  # Add parameter for pre-trained model
    ):
        self.n_episodes = n_episodes
        self.batch_size = batch_size
        self.device = device
        self.max_grad_norm = max_grad_norm
        self.gamma = gamma
        self.eval_frequency = eval_frequency
        self.eval_episodes = eval_episodes
        self.checkpoint_frequency = checkpoint_frequency
        self.max_checkpoints = max_checkpoints
        self.learning_rate = learning_rate  # Store learning rate
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        
        # Setup experiment name and directories
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_name = experiment_name or f"yahtzee_run_{timestamp}"
        self.run_dir = Path("runs") / self.experiment_name
        self.checkpoint_dir = self.run_dir / "checkpoints"
        self.metrics_dir = self.run_dir / "metrics"
        
        # Create directories
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.metrics_dir.mkdir(exist_ok=True)
        
        # Initialize tensorboard writer
        self.writer = SummaryWriter(log_dir=str(self.run_dir / "tensorboard"))
        
        # Save hyperparameters
        self.save_hyperparameters()
        
        # Create observation space for the agent
        self.observation_space = {
            "dice": np.zeros(5),
            "roll_number": 0,
            "scoresheet": np.zeros(13),
            "upper_bonus": 0,
            "yahtzee_bonus": np.array([0]),
            "opponent_scores": np.array([0]),  # Single opponent for now
            "relative_rank": 0
        }
        
        # Initialize agent
        self.agent = RLAgent(
            observation_space=self.observation_space,
            device=device
        )
        
        # Load pre-trained model if provided
        if pretrained_model_path:
            self.load_pretrained_model(pretrained_model_path)
            logger.info(f"Loaded pre-trained model from {pretrained_model_path}")
        
        # Training metrics
        self.episode_rewards = []
        self.episode_lengths = []
        self.value_losses = []
        self.policy_losses = []
        self.entropy_losses = []
        self.eval_scores = []  # Track evaluation performance
        
        # Experience buffer
        self.buffer_size = 1000
        self.experience_buffer = deque(maxlen=self.buffer_size)
        
        # Initialize reward scaler
        self.reward_scaler = RewardScaler(gamma=self.gamma)
        
        # Track best scores for reward context
        self.best_episode_score = float('-inf')
        self.worst_episode_score = float('inf')
        
        # Track best evaluation performance
        self.best_eval_score = float('-inf')
    
    def save_hyperparameters(self):
        """Save hyperparameters to a file."""
        hyperparams = {
            'n_episodes': self.n_episodes,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'gamma': self.gamma,
            'gae_lambda': self.gae_lambda,
            'entropy_coef': self.entropy_coef,
            'value_loss_coef': self.value_loss_coef,
            'max_grad_norm': self.max_grad_norm,
            'eval_frequency': self.eval_frequency,
            'eval_episodes': self.eval_episodes,
            'device': str(self.device)
        }
        
        with open(self.run_dir / "hyperparameters.json", 'w') as f:
            json.dump(hyperparams, f, indent=4)
    
    def normalize_score(self, score: float) -> float:
        """Normalize score based on game context."""
        # Update best/worst scores
        self.best_episode_score = max(self.best_episode_score, score)
        self.worst_episode_score = min(self.worst_episode_score, score)
        
        # If we don't have enough context, use basic scaling
        if self.best_episode_score == self.worst_episode_score:
            return score / 50.0  # fallback to basic scaling
        
        # Normalize score to [0, 1] range based on historical context
        normalized = (score - self.worst_episode_score) / (self.best_episode_score - self.worst_episode_score)
        return normalized
        
    def collect_episode(self) -> Tuple[List[Dict], float]:
        """Collect a single episode of experience."""
        # Reset game state
        dice = Dice()
        scoresheet = ScoreSheet()
        episode_reward = 0
        observations = []
        turn_rewards = []  # Track rewards for each turn
        
        # Play until scoresheet is complete
        while not scoresheet.is_complete():
            # Get current observation
            obs = {
                'dice': np.array(dice.get_values()),
                'roll_number': dice.roll_count,
                'scoresheet': np.array([scoresheet.scores.get(cat, -1) for cat in ALL_CATEGORIES]),
                'upper_bonus': float(scoresheet.get_upper_section_bonus() > 0),
                'yahtzee_bonus': np.array([scoresheet.yahtzee_bonus_score // 100]),
                'opponent_scores': np.array([0]),  # Placeholder for now
                'relative_rank': 0  # Placeholder for now
            }
            
            # First decision: which dice to reroll
            dice_action, dice_log_prob = self.agent.ac_net.get_dice_action(obs)
            if dice.roll_count < 3:  # Can only reroll if not used all rolls
                reroll_indices = [i for i, should_reroll in enumerate(dice_action) if should_reroll]
                dice.roll_specific(reroll_indices)
            
            # Second decision: which category to score in
            category_idx, category_log_prob = self.agent.ac_net.get_category_action(obs)
            category = ALL_CATEGORIES[category_idx]
            
            # Score the dice and calculate reward
            score = scoresheet.score_category(category, dice.get_values())
            
            # Calculate immediate reward
            immediate_reward = self.normalize_score(score)
            
            # Add bonus rewards for strategic plays
            if category in ['YAHTZEE', 'LARGE_STRAIGHT']:
                immediate_reward *= 1.2  # Bonus for difficult combinations
            elif scoresheet.get_upper_section_bonus() > 0:
                immediate_reward *= 1.1  # Bonus for achieving upper section bonus
            
            # Scale the reward using running statistics
            scaled_reward = self.reward_scaler.scale(immediate_reward)
            turn_rewards.append(scaled_reward)
            episode_reward += scaled_reward
            
            # Store transition
            next_obs = {
                'dice': np.array(dice.get_values()),
                'roll_number': dice.roll_count,
                'scoresheet': np.array([scoresheet.scores.get(cat, -1) for cat in ALL_CATEGORIES]),
                'upper_bonus': float(scoresheet.get_upper_section_bonus() > 0),
                'yahtzee_bonus': np.array([scoresheet.yahtzee_bonus_score // 100]),
                'opponent_scores': np.array([0]),
                'relative_rank': 0
            }
            
            done = scoresheet.is_complete()
            
            # Add final game reward if done
            if done:
                final_score = scoresheet.get_grand_total()
                final_reward = self.normalize_score(final_score) * 0.5  # Final game reward
                scaled_final_reward = self.reward_scaler.scale(final_reward)
                episode_reward += scaled_final_reward
            
            # Store experience
            self.agent.store_transition(
                state=obs,
                dice_action=dice_action,
                category_action=category_idx,
                reward=scaled_reward,
                next_state=next_obs,
                done=done
            )
            
            observations.append(obs)
            
            # Reset dice for next turn if not done
            if not done:
                dice.reset_roll_count()
            
            # Update reward statistics
            self.reward_scaler.update(immediate_reward)
        
        return observations, episode_reward
    
    def evaluate(self, n_episodes: int = None) -> Dict[str, float]:
        """Evaluate the current policy without exploration."""
        if n_episodes is None:
            n_episodes = self.eval_episodes
            
        total_score = 0
        total_reward = 0
        episode_lengths = []
        upper_bonus_count = 0
        yahtzee_count = 0
        
        # Store original exploration state
        original_deterministic = self.agent.deterministic
        self.agent.deterministic = True
        
        try:
            for _ in range(n_episodes):
                dice = Dice()
                scoresheet = ScoreSheet()
                episode_reward = 0
                turn_count = 0
                
                while not scoresheet.is_complete():
                    obs = {
                        'dice': np.array(dice.get_values()),
                        'roll_number': dice.roll_count,
                        'scoresheet': np.array([scoresheet.scores.get(cat, -1) for cat in ALL_CATEGORIES]),
                        'upper_bonus': float(scoresheet.get_upper_section_bonus() > 0),
                        'yahtzee_bonus': np.array([scoresheet.yahtzee_bonus_score // 100]),
                        'opponent_scores': np.array([0]),
                        'relative_rank': 0
                    }
                    
                    # Get dice action
                    dice_action, _ = self.agent.ac_net.get_dice_action(obs)
                    if dice.roll_count < 3:
                        reroll_indices = [i for i, should_reroll in enumerate(dice_action) if should_reroll]
                        dice.roll_specific(reroll_indices)
                    
                    # Get category action
                    category_idx, _ = self.agent.ac_net.get_category_action(obs)
                    category = ALL_CATEGORIES[category_idx]
                    
                    # Score and get reward
                    score = scoresheet.score_category(category, dice.get_values())
                    reward = self.normalize_score(score)
                    episode_reward += reward
                    
                    turn_count += 1
                    
                    if not scoresheet.is_complete():
                        dice.reset_roll_count()
                
                # Collect statistics
                final_score = scoresheet.get_grand_total()
                total_score += final_score
                total_reward += episode_reward
                episode_lengths.append(turn_count)
                
                if scoresheet.get_upper_section_bonus() > 0:
                    upper_bonus_count += 1
                
                if scoresheet.scores.get('YAHTZEE', 0) > 0:
                    yahtzee_count += 1
        
        finally:
            # Restore original exploration state
            self.agent.deterministic = original_deterministic
        
        # Calculate statistics
        avg_score = total_score / n_episodes
        avg_reward = total_reward / n_episodes
        avg_length = sum(episode_lengths) / n_episodes
        upper_bonus_rate = upper_bonus_count / n_episodes
        yahtzee_rate = yahtzee_count / n_episodes
        
        return {
            'avg_score': avg_score,
            'avg_reward': avg_reward,
            'avg_length': avg_length,
            'upper_bonus_rate': upper_bonus_rate,
            'yahtzee_rate': yahtzee_rate,
            'min_length': min(episode_lengths),
            'max_length': max(episode_lengths)
        }
    
    def save_checkpoint(self, episode: int, is_best: bool = False):
        """Save a training checkpoint."""
        checkpoint = {
            'episode': episode,
            'model_state_dict': self.agent.ac_net.state_dict(),
            'optimizer_state_dict': self.agent.optimizer.state_dict(),
            'reward_scaler_state': {
                'mean': self.reward_scaler.running_mean,
                'variance': self.reward_scaler.running_variance,
                'count': self.reward_scaler.count
            },
            'best_eval_score': self.best_eval_score,
            'episode_rewards': self.episode_rewards,
            'eval_scores': self.eval_scores,
            'metrics': {
                'value_losses': self.value_losses,
                'policy_losses': self.policy_losses,
                'entropy_losses': self.entropy_losses
            }
        }
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_episode_{episode}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # If this is the best model, save a copy
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            shutil.copy(checkpoint_path, best_path)
        
        # Maintain only max_checkpoints most recent checkpoints
        checkpoints = sorted(
            [f for f in self.checkpoint_dir.glob("checkpoint_episode_*.pt")],
            key=lambda x: int(x.stem.split('_')[-1])
        )
        
        if len(checkpoints) > self.max_checkpoints:
            for checkpoint_to_remove in checkpoints[:-self.max_checkpoints]:
                checkpoint_to_remove.unlink()
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load a training checkpoint."""
        checkpoint = torch.load(checkpoint_path)
        
        # Load model and optimizer states
        self.agent.ac_net.load_state_dict(checkpoint['model_state_dict'])
        self.agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load reward scaler state
        self.reward_scaler.running_mean = checkpoint['reward_scaler_state']['mean']
        self.reward_scaler.running_variance = checkpoint['reward_scaler_state']['variance']
        self.reward_scaler.count = checkpoint['reward_scaler_state']['count']
        
        # Load training state
        self.best_eval_score = checkpoint['best_eval_score']
        self.episode_rewards = checkpoint['episode_rewards']
        self.eval_scores = checkpoint['eval_scores']
        self.value_losses = checkpoint['metrics']['value_losses']
        self.policy_losses = checkpoint['metrics']['policy_losses']
        self.entropy_losses = checkpoint['metrics']['entropy_losses']
        
        return checkpoint['episode']
    
    def log_metrics(self, metrics: Dict[str, float], step: int, prefix: str = ""):
        """Log metrics to tensorboard."""
        for name, value in metrics.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(f"{prefix}/{name}", value, step)
    
    def train(self):
        """Main training loop."""
        logger.info(f"Starting training on device: {self.device}")
        logger.info(f"Experiment name: {self.experiment_name}")
        logger.info(f"Tensorboard logs: {self.run_dir / 'tensorboard'}")
        
        start_time = time.time()
        running_reward = 0
        
        for episode in range(self.n_episodes):
            # Collect experience
            observations, episode_reward = self.collect_episode()
            
            # Update running reward
            running_reward = 0.05 * episode_reward + (1 - 0.95) * running_reward
            
            # Store metrics
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(len(observations))
            
            # Log episode metrics
            self.log_metrics(
                {
                    'reward': episode_reward,
                    'running_reward': running_reward,
                    'episode_length': len(observations)
                },
                episode,
                prefix='train'
            )
            
            # Update policy if we have enough experience
            if len(self.agent.transitions) >= self.batch_size:
                losses = self.agent.update()
                if losses is not None:
                    policy_loss, value_loss, entropy_loss = losses
                    self.policy_losses.append(policy_loss)
                    self.value_losses.append(value_loss)
                    self.entropy_losses.append(entropy_loss)
                    
                    # Log loss metrics
                    self.log_metrics(
                        {
                            'policy_loss': policy_loss,
                            'value_loss': value_loss,
                            'entropy_loss': entropy_loss
                        },
                        episode,
                        prefix='losses'
                    )
                    
                    # Log hyperparameter schedules
                    hyperparams = self.agent.update_hyperparameters()
                    self.log_metrics(
                        {
                            'gae_lambda': hyperparams['gae_lambda'],
                            'entropy_coef': hyperparams['entropy_coef'],
                            'learning_rate': hyperparams['learning_rate']
                        },
                        episode,
                        prefix='hyperparameters'
                    )
            
            # Evaluation phase
            if (episode + 1) % self.eval_frequency == 0:
                eval_stats = self.evaluate()
                self.eval_scores.append(eval_stats['avg_score'])
                
                # Log evaluation metrics
                self.log_metrics(eval_stats, episode, prefix='eval')
                
                # Save best model if we have a new best score
                is_best = eval_stats['avg_score'] > self.best_eval_score
                if is_best:
                    self.best_eval_score = eval_stats['avg_score']
                    logger.info(f"New best model with eval score: {self.best_eval_score:.2f}")
                
                # Log evaluation results and current hyperparameters
                logger.info(
                    f"\nEvaluation after episode {episode + 1}:\n"
                    f"Average Score: {eval_stats['avg_score']:.2f}\n"
                    f"Average Episode Length: {eval_stats['avg_length']:.2f}\n"
                    f"Upper Bonus Rate: {eval_stats['upper_bonus_rate']:.2%}\n"
                    f"Yahtzee Rate: {eval_stats['yahtzee_rate']:.2%}\n"
                    f"Current GAE Lambda: {hyperparams['gae_lambda']:.3f}\n"
                    f"Current Entropy Coef: {hyperparams['entropy_coef']:.2e}\n"
                    f"Current Learning Rate: {hyperparams['learning_rate']:.2e}\n"
                )
            
            # Save checkpoint
            if (episode + 1) % self.checkpoint_frequency == 0:
                self.save_checkpoint(episode + 1, is_best=is_best)
                logger.info(f"Saved checkpoint at episode {episode + 1}")
            
            # Log training progress
            if (episode + 1) % 100 == 0:
                elapsed_time = time.time() - start_time
                reward_stats = self.reward_scaler.get_stats()
                
                # Log reward scaler stats
                self.log_metrics(reward_stats, episode, prefix='reward_scaler')
                
                logger.info(
                    f"Episode {episode + 1}/{self.n_episodes} | "
                    f"Running reward: {running_reward:.2f} | "
                    f"Episode reward: {episode_reward:.2f} | "
                    f"Reward mean: {reward_stats['mean']:.2f} | "
                    f"Reward std: {reward_stats['std']:.2f} | "
                    f"Time: {elapsed_time:.2f}s"
                )
        
        logger.info("Training completed!")
        # Save final checkpoint
        self.save_checkpoint(self.n_episodes, is_best=False)
        self.writer.close()

    def load_pretrained_model(self, model_path: str):
        """Load a pre-trained model."""
        checkpoint = torch.load(model_path)
        self.agent.ac_net.load_state_dict(checkpoint['model_state_dict'])
        
        # Don't load optimizer state from pre-training
        # Instead, initialize with RL training parameters
        self.agent.optimizer = optim.Adam(
            self.agent.ac_net.parameters(),
            lr=self.learning_rate
        )
        
        logger.info("Loaded pre-trained model weights")

if __name__ == "__main__":
    # Create trainer with default hyperparameters
    trainer = YahtzeeTrainer(
        experiment_name="yahtzee_ac_v1",
        checkpoint_frequency=1000,
        eval_frequency=100
    )
    
    # Start training
    trainer.train() 