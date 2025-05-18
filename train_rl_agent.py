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
            
            # Get action from unified action space
            action, dice_action, category_idx, action_logits = self.agent.ac_net.get_action(obs)
            
            # Handle dice reroll if applicable
            if dice_action is not None and dice.roll_count < 3:  # Can only reroll if not used all rolls
                reroll_indices = [i for i, should_reroll in enumerate(dice_action) if should_reroll]
                dice.roll_specific(reroll_indices)
            
            # Handle category scoring if applicable
            if category_idx is not None:
                category = ALL_CATEGORIES[category_idx]
                # Score the dice and calculate reward
                score = scoresheet.record_score(category, dice.get_values())
                
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
                    action=action,  # Store the unified action
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
                    
                    # Get action from unified action space
                    _, dice_action, category_idx, _ = self.agent.ac_net.get_action(obs, deterministic=True)
                    
                    # Handle dice reroll if applicable
                    if dice_action is not None and dice.roll_count < 3:
                        reroll_indices = [i for i, should_reroll in enumerate(dice_action) if should_reroll]
                        dice.roll_specific(reroll_indices)
                    
                    # Handle category scoring if applicable
                    if category_idx is not None:
                        category = ALL_CATEGORIES[category_idx]
                        score = scoresheet.record_score(category, dice.get_values())
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
    
    def load_pretrained_model(self, model_path: str):
        """Load a pre-trained model and prepare it for RL training."""
        logger.info(f"Loading pre-trained model from {model_path}")
        try:
            # Load just the model state dict
            state_dict = torch.load(model_path, map_location=self.device)
            self.agent.ac_net.load_state_dict(state_dict)
            
            # Initialize optimizer after loading model
            self.agent.optimizer = optim.Adam(
                self.agent.ac_net.parameters(),
                lr=self.learning_rate,
                weight_decay=1e-5  # Add L2 regularization
            )
            
            # Try to load metrics if they exist
            metrics_path = Path(model_path).parent / "metrics_final.json"
            if metrics_path.exists():
                with open(metrics_path, 'r') as f:
                    metrics = json.load(f)
                if 'avg_game_score' in metrics:
                    self.best_eval_score = metrics['avg_game_score']
                    logger.info(f"Loaded initial best score: {self.best_eval_score}")
            
            logger.info("Successfully loaded pre-trained model")
        except Exception as e:
            logger.error(f"Error loading pre-trained model: {e}")
            raise

    def train(self):
        """Train the agent using actor-critic with GAE."""
        logger.info(f"Starting training with experiment name: {self.experiment_name}")
        logger.info(f"Training device: {self.device}")
        
        start_time = time.time()
        episode = 0
        total_steps = 0
        
        # Initialize learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.agent.optimizer,
            mode='max',
            factor=0.5,
            patience=5,
            min_lr=1e-5
        )
        
        while episode < self.n_episodes:
            # Collect experience
            observations, episode_reward = self.collect_episode()
            total_steps += len(observations)
            
            # Update metrics
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(len(observations))
            
            # Log episode metrics
            self.writer.add_scalar('train/episode_reward', episode_reward, episode)
            self.writer.add_scalar('train/episode_length', len(observations), episode)
            
            # Perform evaluation
            if (episode + 1) % self.eval_frequency == 0:
                eval_metrics = self.evaluate()
                self.log_metrics(eval_metrics, episode, prefix='eval/')
                
                # Update learning rate based on evaluation performance
                scheduler.step(eval_metrics['avg_score'])
                
                # Save checkpoint if best model
                if eval_metrics['avg_score'] > self.best_eval_score:
                    self.best_eval_score = eval_metrics['avg_score']
                    self.save_checkpoint(episode, is_best=True)
                    logger.info(f"New best model with score: {self.best_eval_score:.1f}")
                
                # Regular checkpoint
                if (episode + 1) % self.checkpoint_frequency == 0:
                    self.save_checkpoint(episode)
                
                # Early stopping check
                recent_rewards = self.episode_rewards[-100:]
                if len(recent_rewards) >= 100:
                    avg_reward = sum(recent_rewards) / len(recent_rewards)
                    if avg_reward > 300:  # High average score threshold
                        logger.info(f"Reached target performance with average reward {avg_reward:.1f}")
                        break
            
            # Update policy
            if len(self.agent.transitions) >= self.batch_size:
                policy_loss, value_loss, entropy = self.agent.update()
                
                # Log training metrics
                self.writer.add_scalar('train/policy_loss', policy_loss, total_steps)
                self.writer.add_scalar('train/value_loss', value_loss, total_steps)
                self.writer.add_scalar('train/entropy', entropy, total_steps)
                
                # Store losses for tracking
                self.policy_losses.append(policy_loss)
                self.value_losses.append(value_loss)
                self.entropy_losses.append(entropy)
                
                # Adjust entropy coefficient
                self.agent.entropy_coef = max(0.001, self.agent.entropy_coef * 0.995)
            
            episode += 1
            
            # Log progress
            if episode % 10 == 0:
                elapsed_time = time.time() - start_time
                steps_per_sec = total_steps / elapsed_time
                logger.info(
                    f"Episode {episode}/{self.n_episodes} | "
                    f"Steps: {total_steps} | "
                    f"Steps/sec: {steps_per_sec:.1f} | "
                    f"Last reward: {episode_reward:.1f} | "
                    f"Best eval score: {self.best_eval_score:.1f}"
                )
        
        # Final evaluation
        final_metrics = self.evaluate(n_episodes=100)  # More episodes for final evaluation
        self.log_metrics(final_metrics, episode, prefix='final/')
        self.save_checkpoint(episode, is_best=True)
        
        # Save final metrics
        with open(self.metrics_dir / "final_metrics.json", 'w') as f:
            json.dump(final_metrics, f, indent=4)
        
        logger.info("Training completed!")
        logger.info(f"Final evaluation metrics: {final_metrics}")
        return final_metrics

if __name__ == "__main__":
    # Create trainer with default hyperparameters
    trainer = YahtzeeTrainer(
        experiment_name="yahtzee_ac_v1",
        checkpoint_frequency=1000,
        eval_frequency=100
    )
    
    # Start training
    trainer.train() 