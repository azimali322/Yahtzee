import torch
import torch.nn.functional as F
from yahtzee_ai import RLAgent, YahtzeeAI
from game_logic import ScoreSheet, Dice, ALL_CATEGORIES
import logging
import time
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import datetime
from tqdm import tqdm
import numpy as np
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
import io
import random
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress yahtzee_ai INFO messages
logging.getLogger('yahtzee_ai').setLevel(logging.WARNING)

class PreTrainer:
    def __init__(
        self,
        n_episodes: int = 10000,
        batch_size: int = 128,
        learning_rate: float = 5e-3,  # Updated to 5e-3
        experiment_name: str = None,
        device: str = None,  # Changed default to None
        checkpoint_interval: int = 1000,
        eval_interval: int = 1000,
        checkpoint_dir: str = "checkpoints",
        log_dir: str = "logs",
        max_grad_norm: float = 1.0,
        continue_from_checkpoint: bool = False,
        replay_buffer_size: int = 100000,  # Increased from 10000 to 100000
        dice_loss_weight: float = 1.0,
        category_loss_weight: float = 5.0,  # Increased from 2.0 to 5.0
        episodes_per_update: int = 10  # New parameter for batch updates
    ):
        self.n_episodes = n_episodes
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = torch.device(device if device is not None else "cpu")  # Modified device handling
        self.checkpoint_interval = checkpoint_interval
        self.eval_interval = eval_interval
        self.max_grad_norm = max_grad_norm
        self.continue_from_checkpoint = continue_from_checkpoint
        self.replay_buffer_size = replay_buffer_size
        self.dice_loss_weight = dice_loss_weight
        self.category_loss_weight = category_loss_weight
        self.episodes_per_update = episodes_per_update
        
        # Initialize replay buffer
        self.replay_buffer = []
        
        # Create observation space
        self.observation_space = {
            "dice": np.zeros(5, dtype=np.float32),
            "roll_number": np.array([0], dtype=np.float32),
            "scoresheet": np.zeros(13, dtype=np.float32),
            "upper_bonus": np.array([0], dtype=np.float32),
            "yahtzee_bonus": np.array([0], dtype=np.float32),
            "opponent_scores": np.array([0], dtype=np.float32),
            "relative_rank": np.array([0], dtype=np.float32)
        }
        
        # Initialize networks
        self.teacher = YahtzeeAI("greedy3")
        self.student = RLAgent(observation_space=self.observation_space, device=self.device)
        
        # Initialize optimizer with learning rate scheduling
        self.optimizer = torch.optim.Adam(
            self.student.ac_net.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-4  # Add L2 regularization
        )
        
        # Setup logging and checkpointing
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_name = experiment_name or f"pretrain_{timestamp}"
        self.checkpoint_dir = Path(checkpoint_dir) / self.experiment_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=f"{log_dir}/{self.experiment_name}")
        
        # Load checkpoint if continuing training
        if continue_from_checkpoint:
            self.load_best_checkpoint()
        
        # Metrics storage
        self.dice_losses = []
        self.category_losses = []
        self.dice_accuracies = []
        self.category_accuracies = []
        self.game_scores = []  # New: Track actual game scores
        self.best_eval_score = float('-inf')
        self.category_confusion = np.zeros((13, 13))
        self.dice_accuracy_history = []
        self.category_accuracy_history = []
        self.game_score_history = []  # New: Track game score history
    
    def load_best_checkpoint(self):
        """Load the best checkpoint from the previous training."""
        previous_checkpoint_dir = Path("checkpoints/yahtzee_pretrain_faster")
        best_model_path = previous_checkpoint_dir / "best_model.pt"
        
        if best_model_path.exists():
            logger.info(f"Loading checkpoint from {best_model_path}")
            # Set weights_only=False to load both model weights and optimizer state
            checkpoint = torch.load(best_model_path, weights_only=False)
            self.student.ac_net.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.best_eval_score = checkpoint.get('metrics', {}).get('dice_accuracy', 0.0) + \
                                 checkpoint.get('metrics', {}).get('category_accuracy', 0.0)
            logger.info(f"Loaded checkpoint with best score: {self.best_eval_score:.4f}")
        else:
            logger.warning("No previous checkpoint found, starting from scratch")
    
    def collect_teacher_decision(self, dice: Dice, scoresheet: ScoreSheet) -> int:
        """Collect teacher's decision for current game state."""
        current_values = dice.get_values()
        
        # Get teacher's decisions
        reroll_indices = self.teacher.decide_reroll(current_values, dice.roll_count)
        category = self.teacher.choose_category(current_values)
        
        if category is not None:
            # Category action: offset by 32 (number of possible dice combinations)
            action = ALL_CATEGORIES.index(category) + 32
        else:
            # Dice reroll action: convert reroll indices to binary number
            action = sum(1 << idx for idx in reroll_indices)
        
        return action
    
    def get_observation(self, dice: Dice, scoresheet: ScoreSheet):
        """Create observation dictionary from current game state."""
        # Convert all numpy arrays to float32 for PyTorch compatibility
        return {
            'dice': np.array(dice.get_values(), dtype=np.float32),
            'roll_number': np.array([dice.roll_count], dtype=np.float32),
            'scoresheet': np.array([
                scoresheet.scores.get(cat, -1) 
                for cat in ALL_CATEGORIES
            ], dtype=np.float32),
            'upper_bonus': np.array([float(scoresheet.get_upper_section_bonus() > 0)], dtype=np.float32),
            'yahtzee_bonus': np.array([scoresheet.yahtzee_bonus_score // 100], dtype=np.float32),
            'opponent_scores': np.array([0], dtype=np.float32),
            'relative_rank': np.array([0], dtype=np.float32)
        }
    
    def pretrain_episode(self):
        """Run one pre-training episode and collect experience."""
        dice = Dice()
        scoresheet = ScoreSheet()
        episode_experiences = []
        
        # Set game state for teacher
        self.teacher.set_game_state(scoresheet, dice)
        
        while not scoresheet.is_complete():
            # Get current observation
            obs = self.get_observation(dice, scoresheet)
            
            # Get teacher's decision as a single action
            action = self.collect_teacher_decision(dice, scoresheet)
            
            # Store experience
            episode_experiences.append((obs, action))
            
            # Apply teacher's decisions
            if action < 32:  # Dice reroll action
                reroll_indices = [i for i in range(5) if (action >> i) & 1]
                if dice.roll_count < 3 and reroll_indices:
                    dice.roll_specific(reroll_indices)
            else:  # Category action
                category = ALL_CATEGORIES[action - 32]
                scoresheet.record_score(category, dice.get_values())
                dice.reset_roll_count()
        
        # Get final game score
        final_score = scoresheet.get_grand_total()
        self.game_scores.append(final_score)
        
        # Add experiences to replay buffer
        for exp in episode_experiences:
            self.replay_buffer.append(exp)
            if len(self.replay_buffer) > self.replay_buffer_size:
                self.replay_buffer.pop(0)
        
        return final_score
    
    def plot_confusion_matrix(self, episode: int):
        """Plot and log category selection confusion matrix to TensorBoard."""
        plt.figure(figsize=(10, 8))
        # Use '.0f' format instead of 'd' to handle float values
        sns.heatmap(self.category_confusion, annot=True, fmt='.0f', cmap='Blues')
        plt.title('Category Selection Confusion Matrix')
        plt.xlabel('Predicted Category')
        plt.ylabel('Teacher Category')
        
        # Save plot to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()
        
        # Log to TensorBoard
        img = np.array(plt.imread(buf))
        self.writer.add_image('confusion_matrix', img.transpose(2, 0, 1), episode)
        
    def save_checkpoint(self, episode: int, metrics: Dict, is_best: bool = False):
        """Save training checkpoint."""
        # Save only the model state dict
        model_state = self.student.ac_net.state_dict()
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_{episode}.pt"
        torch.save(model_state, checkpoint_path)
        
        # Save best model if this is the best so far
        if is_best:
            best_model_path = self.checkpoint_dir / "best_model.pt"
            torch.save(model_state, best_model_path)
            logger.info(f"Saved new best model with score {metrics['avg_game_score']:.1f}")
        
        # Save metrics separately as JSON
        metrics_path = self.checkpoint_dir / f"metrics_{episode}.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        
        # Remove old checkpoints (keep only last 3)
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_*.pt"))
        if len(checkpoints) > 3:
            for checkpoint in checkpoints[:-3]:
                checkpoint.unlink()
                # Also remove corresponding metrics file
                metrics_file = self.checkpoint_dir / f"metrics_{checkpoint.stem.split('_')[-1]}.json"
                if metrics_file.exists():
                    metrics_file.unlink()
                
    def log_metrics(self, metrics: Dict, episode: int):
        """Log metrics to TensorBoard."""
        for name, value in metrics.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(f"metrics/{name}", value, episode)
            elif isinstance(value, dict):
                # For nested dictionaries (like per-category metrics)
                for subname, subvalue in value.items():
                    if isinstance(subvalue, (int, float)):
                        self.writer.add_scalar(f"metrics/{name}/{subname}", subvalue, episode)
                    elif isinstance(subvalue, dict) and all(isinstance(v, (int, float)) for v in subvalue.values()):
                        for k, v in subvalue.items():
                            self.writer.add_scalar(f"metrics/{name}/{subname}/{k}", v, episode)
            
    def evaluate(self, n_episodes: int = 50) -> Dict:
        """Evaluate how well the student matches the teacher's decisions."""
        total_dice_match = 0
        total_category_match = 0
        total_decisions = 0
        total_game_score = 0  # New: Track total game score
        per_category_accuracy = {cat: {'correct': 0, 'total': 0} for cat in ALL_CATEGORIES}
        dice_similarity_scores = []
        self.category_confusion = np.zeros((13, 13))
        
        logger.info(f"Evaluating on {n_episodes} episodes...")
        
        for _ in range(n_episodes):
            dice = Dice()
            scoresheet = ScoreSheet()
            self.teacher.set_game_state(scoresheet, dice)
            
            while not scoresheet.is_complete():
                obs = self.get_observation(dice, scoresheet)
                
                # Get teacher's decision as a single action
                teacher_action = self.collect_teacher_decision(dice, scoresheet)
                
                # Get student's decision
                with torch.no_grad():
                    student_action, student_dice_action, student_category, _ = self.student.ac_net.get_action(obs)
                
                # Compare decisions
                action_match = student_action == teacher_action
                total_dice_match += action_match
                total_category_match += action_match
                total_decisions += 1
                
                # Update confusion matrix
                teacher_category = teacher_action - 32 if teacher_action >= 32 else -1
                student_category = student_action - 32 if student_action >= 32 else -1
                if teacher_category >= 0 and student_category >= 0:
                    self.category_confusion[teacher_category][student_category] += 1
                
                # Calculate dice similarity score (Jaccard index)
                if teacher_action < 32 and student_action < 32:
                    teacher_binary = [(teacher_action >> i) & 1 for i in range(5)]
                    student_binary = [(student_action >> i) & 1 for i in range(5)]
                    intersection = sum(t & s for t, s in zip(teacher_binary, student_binary))
                    union = sum(t | s for t, s in zip(teacher_binary, student_binary))
                    similarity = intersection / (union if union > 0 else 1)
                    dice_similarity_scores.append(similarity)
                
                # Update per-category accuracy
                if teacher_action >= 32:
                    category = ALL_CATEGORIES[teacher_action - 32]
                    per_category_accuracy[category]['total'] += 1
                    if action_match:
                        per_category_accuracy[category]['correct'] += 1
                
                # Apply teacher's decisions to advance game
                if teacher_action < 32:  # Dice reroll action
                    reroll_indices = [i for i in range(5) if (teacher_action >> i) & 1]
                    if dice.roll_count < 3 and reroll_indices:
                        dice.roll_specific(reroll_indices)
                else:  # Category action
                    category = ALL_CATEGORIES[teacher_action - 32]
                    scoresheet.record_score(category, dice.get_values())
                    dice.reset_roll_count()
            
            # Add final game score
            total_game_score += scoresheet.get_grand_total()
        
        # Calculate metrics
        dice_accuracy = total_dice_match / total_decisions
        category_accuracy = total_category_match / total_decisions
        avg_dice_similarity = np.mean(dice_similarity_scores) if dice_similarity_scores else 0
        avg_game_score = total_game_score / n_episodes
        
        # Calculate per-category accuracies
        category_accuracies = {
            cat: stats['correct'] / max(stats['total'], 1)
            for cat, stats in per_category_accuracy.items()
        }
        
        return {
            'dice_accuracy': dice_accuracy,
            'category_accuracy': category_accuracy,
            'avg_dice_similarity': avg_dice_similarity,
            'avg_game_score': avg_game_score,
            'per_category_accuracy': category_accuracies
        }
    
    def pretrain(self):
        """Run pre-training using the teacher (Greedy3) as supervision."""
        logger.info(f"Starting pre-training for {self.n_episodes} episodes")
        start_time = time.time()
        
        # Training loop
        for episode in range(self.n_episodes):
            # Run episode and collect experience
            game_score = self.pretrain_episode()
            
            # Update network every episodes_per_update episodes
            if (episode + 1) % self.episodes_per_update == 0:
                dice_loss, category_loss = self.update_from_replay()
                self.dice_losses.append(dice_loss)
                self.category_losses.append(category_loss)
            
            # Print progress every 50 episodes
            if (episode + 1) % 50 == 0:
                elapsed_time = time.time() - start_time
                eps_per_sec = (episode + 1) / elapsed_time
                remaining_episodes = self.n_episodes - (episode + 1)
                eta = remaining_episodes / eps_per_sec
                logger.info(
                    f"Episode {episode + 1}/{self.n_episodes} | "
                    f"Speed: {eps_per_sec:.1f} eps/s | "
                    f"ETA: {eta:.1f}s | "
                    f"Game Score: {game_score:.1f}"
                )
            
            # Evaluate periodically
            if (episode + 1) % self.eval_interval == 0:
                metrics = self.evaluate()
                self.log_metrics(metrics, episode)
                
                # Save checkpoint if this is the best model
                combined_score = metrics['avg_game_score']
                is_best = combined_score > self.best_eval_score
                if is_best:
                    self.best_eval_score = combined_score
                
                self.save_checkpoint(episode, metrics, is_best=is_best)
                
                # Early stopping check
                if metrics['dice_accuracy'] > 0.95 and metrics['category_accuracy'] > 0.95:
                    logger.info("Reached target accuracy, stopping training early")
                    break
            
            # Learning rate scheduling
            if (episode + 1) % 1000 == 0:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = max(1e-5, param_group['lr'] * 0.95)
        
        # Final evaluation
        final_metrics = self.evaluate(n_episodes=100)  # More episodes for final evaluation
        self.save_checkpoint(self.n_episodes - 1, final_metrics, is_best=True)
        self.plot_confusion_matrix(self.n_episodes - 1)
        
        # Log training duration
        duration = time.time() - start_time
        logger.info(f"Pre-training completed in {duration:.2f} seconds")
        logger.info(f"Final Dice Accuracy: {final_metrics['dice_accuracy']:.4f}")
        logger.info(f"Final Category Accuracy: {final_metrics['category_accuracy']:.4f}")
        logger.info(f"Final Average Game Score: {final_metrics['avg_game_score']:.1f}")
        
        return final_metrics

    def update_from_replay(self):
        """Train on a batch from the replay buffer with combined dice and category losses."""
        if len(self.replay_buffer) < self.batch_size:
            return 0, 0
            
        # Sample batch with prioritization of recent experiences
        recent_weight = 0.7  # 70% chance to sample from recent experiences
        recent_size = min(len(self.replay_buffer) // 2, 1000)
        
        batch_indices = []
        for _ in range(self.batch_size):
            if random.random() < recent_weight and recent_size > 0:
                # Sample from recent experiences
                idx = random.randint(len(self.replay_buffer) - recent_size, len(self.replay_buffer) - 1)
            else:
                # Sample from all experiences
                idx = random.randint(0, len(self.replay_buffer) - 1)
            batch_indices.append(idx)
        
        batch = [self.replay_buffer[i] for i in batch_indices]
        obs_batch, actions_batch = zip(*batch)
        
        # Convert to tensors
        obs_tensor = torch.stack([self._process_observation(obs) for obs in obs_batch])
        actions_tensor = torch.tensor(actions_batch, device=self.device)
        
        # Forward pass
        action_logits, _ = self.student.ac_net(obs_tensor)
        
        # Calculate combined loss using a masked approach
        dice_mask = actions_tensor < 32
        category_mask = ~dice_mask
        
        # Get logits for both action types
        dice_logits = action_logits[:, :32]  # First 32 outputs are for dice actions
        category_logits = action_logits[:, 32:]  # Remaining outputs are for category actions
        
        # Calculate losses with masking
        batch_size = len(actions_batch)
        combined_loss = torch.zeros(1, device=self.device)
        
        # Process dice actions
        if dice_mask.any():
            dice_targets = actions_tensor[dice_mask]
            dice_probs = F.softmax(dice_logits[dice_mask], dim=-1)
            dice_log_probs = F.log_softmax(dice_logits[dice_mask], dim=-1)
            dice_loss = F.nll_loss(
                dice_log_probs,
                dice_targets,
                reduction='none'
            )
            combined_loss += dice_loss.mean()
        
        # Process category actions
        if category_mask.any():
            category_targets = actions_tensor[category_mask] - 32  # Adjust indices
            category_probs = F.softmax(category_logits[category_mask], dim=-1)
            category_log_probs = F.log_softmax(category_logits[category_mask], dim=-1)
            category_loss = F.nll_loss(
                category_log_probs,
                category_targets,
                reduction='none'
            )
            combined_loss += category_loss.mean()
        
        # Add regularization terms
        # 1. Entropy regularization to encourage exploration
        entropy_reg = -(
            (F.softmax(dice_logits, dim=-1) * F.log_softmax(dice_logits, dim=-1)).sum(dim=-1).mean() +
            (F.softmax(category_logits, dim=-1) * F.log_softmax(category_logits, dim=-1)).sum(dim=-1).mean()
        ) * 0.01  # Small entropy coefficient
        
        # 2. Action correlation regularization
        # This encourages meaningful relationships between dice and category actions
        dice_probs_all = F.softmax(dice_logits, dim=-1)
        category_probs_all = F.softmax(category_logits, dim=-1)
        correlation_reg = torch.zeros(1, device=self.device)
        
        # Calculate correlation between dice patterns and category probabilities
        for i in range(32):  # For each dice pattern
            for j in range(13):  # For each category
                # Higher probability of choosing certain categories with certain dice patterns
                if i & (1 << j % 5):  # If the dice pattern includes this position
                    correlation_reg += dice_probs_all[:, i].mean() * category_probs_all[:, j].mean()
        
        correlation_reg *= 0.005  # Small correlation coefficient
        
        # Final loss combines all components
        total_loss = combined_loss - entropy_reg + correlation_reg
        
        # Backward pass with gradient clipping
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.student.ac_net.parameters(), self.max_grad_norm)
        self.optimizer.step()
        
        # Return the individual loss components for logging
        return (
            combined_loss.item() if dice_mask.any() else 0,
            combined_loss.item() if category_mask.any() else 0
        )
        
    def _process_observation(self, obs):
        """Convert observation to tensor."""
        if isinstance(obs, torch.Tensor):
            return obs.to(self.device)
        return self.student.ac_net._process_observation(obs)

if __name__ == "__main__":
    # Training configuration
    config = {
        "batch_size": 128,
        "learning_rate": 5e-3,  # Updated to 5e-3
        "checkpoint_interval": 1000,
        "eval_interval": 1000,
        "n_episodes": 10000,
        "experiment_name": "yahtzee_pretrain_faster",
        "max_grad_norm": 1.0,
        "continue_from_checkpoint": False,
        "replay_buffer_size": 100000,
        "dice_loss_weight": 1.0,
        "category_loss_weight": 5.0,
        "episodes_per_update": 10
    }
    
    # Ask user if they want to continue from checkpoint
    while True:
        response = input("Do you want to continue training from the previous best checkpoint? (yes/no): ").lower()
        if response in ['yes', 'no']:
            config['continue_from_checkpoint'] = (response == 'yes')
            break
        print("Please answer 'yes' or 'no'")
    
    logger.info(f"\n{'='*50}")
    logger.info(f"Starting pre-training with {config['n_episodes']} episodes")
    logger.info(f"Learning rate: {config['learning_rate']}")
    logger.info(f"Evaluation interval: {config['eval_interval']} episodes")
    logger.info(f"Continue from checkpoint: {config['continue_from_checkpoint']}")
    logger.info(f"{'='*50}\n")
    
    # Initialize pre-trainer
    pretrainer = PreTrainer(**config)
    
    # Run pre-training
    final_metrics = pretrainer.pretrain()
    
    # Log final results
    logger.info("\nFinal Results:")
    logger.info(f"Dice action accuracy: {final_metrics['dice_accuracy']:.4f}")
    logger.info(f"Category accuracy: {final_metrics['category_accuracy']:.4f}")
    logger.info("\nPer-category accuracy:")
    for cat, acc in final_metrics['per_category_accuracy'].items():
        logger.info(f"{cat}: {acc:.4f}")
    
    # Save model
    logger.info(f"\nModel saved in {pretrainer.checkpoint_dir}") 