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
        learning_rate: float = 1e-3,  # Reduced from 5e-0 to 1e-3
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
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=1000,
            gamma=0.5
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
    
    def collect_teacher_decision(self, dice: Dice, scoresheet: ScoreSheet) -> Tuple[torch.Tensor, int]:
        """Collect teacher's decision for current game state."""
        current_values = dice.get_values()
        
        # Get teacher's decisions
        reroll_indices = self.teacher.decide_reroll(current_values, dice.roll_count)
        category = self.teacher.choose_category(current_values)
        
        # Convert reroll indices to binary action
        dice_action = torch.zeros(5, dtype=torch.bool, device=self.device)
        for idx in reroll_indices:
            dice_action[idx] = True
        
        # Convert category to index
        category_idx = ALL_CATEGORIES.index(category) if category else 0
        
        return dice_action, category_idx
    
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
        episode_experiences = []  # Store experiences for this episode
        
        # Set game state for teacher
        self.teacher.set_game_state(scoresheet, dice)
        
        while not scoresheet.is_complete():
            # Get current observation
            obs = self.get_observation(dice, scoresheet)
            
            # Get teacher's decisions
            teacher_dice_action, teacher_category_idx = self.collect_teacher_decision(
                dice, scoresheet
            )
            
            # Store experience
            episode_experiences.append((obs, teacher_dice_action, teacher_category_idx))
            
            # Apply teacher's decisions
            if dice.roll_count < 3:
                reroll_indices = [i for i, reroll in enumerate(teacher_dice_action) if reroll]
                dice.roll_specific(reroll_indices)
            
            # Score category if chosen
            if teacher_category_idx is not None:
                category = ALL_CATEGORIES[teacher_category_idx]
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
        """Save a checkpoint of the model and training state."""
        checkpoint = {
            'episode': episode,
            'model_state_dict': self.student.ac_net.state_dict(),
            'optimizer_state_dict': self.student.optimizer.state_dict(),
            'metrics': metrics,
            'dice_losses': self.dice_losses,
            'category_losses': self.category_losses,
            'category_confusion': self.category_confusion,
        }
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_ep{episode}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save as best model if applicable
        if is_best:
            best_model_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_model_path)
            logger.info(f"Saved new best model at episode {episode}")
            
        # Remove old checkpoints (keep only last 3)
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_ep*.pt"))
        if len(checkpoints) > 3:
            for checkpoint in checkpoints[:-3]:
                checkpoint.unlink()
                
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
                
                # Get teacher's decisions
                teacher_dice_action, teacher_category_idx = self.collect_teacher_decision(
                    dice, scoresheet
                )
                
                # Get student's decisions
                with torch.no_grad():
                    student_dice_action, dice_logits = self.student.ac_net.get_dice_action(obs)
                    student_category_idx, _ = self.student.ac_net.get_category_action(obs)
                
                # Compare decisions
                dice_match = torch.all(student_dice_action == teacher_dice_action)
                category_match = student_category_idx == teacher_category_idx
                
                # Update confusion matrix
                self.category_confusion[teacher_category_idx][student_category_idx] += 1
                
                # Calculate dice similarity score (Jaccard index)
                intersection = torch.sum(student_dice_action & teacher_dice_action).float()
                union = torch.sum(student_dice_action | teacher_dice_action).float()
                similarity = intersection / (union + 1e-8)
                dice_similarity_scores.append(similarity.item())
                
                # Update per-category accuracy
                if teacher_category_idx is not None:
                    category = ALL_CATEGORIES[teacher_category_idx]
                    per_category_accuracy[category]['total'] += 1
                    if category_match:
                        per_category_accuracy[category]['correct'] += 1
                
                total_dice_match += dice_match.item()
                total_category_match += category_match
                total_decisions += 1
                
                # Apply teacher's decisions to advance game
                if dice.roll_count < 3:
                    reroll_indices = [i for i, reroll in enumerate(teacher_dice_action) if reroll]
                    dice.roll_specific(reroll_indices)
                
                if teacher_category_idx is not None:
                    category = ALL_CATEGORIES[teacher_category_idx]
                    scoresheet.record_score(category, dice.get_values())
                    dice.reset_roll_count()
            
            # Add final game score
            total_game_score += scoresheet.get_grand_total()
        
        # Calculate metrics
        dice_accuracy = total_dice_match / total_decisions
        category_accuracy = total_category_match / total_decisions
        avg_dice_similarity = np.mean(dice_similarity_scores)
        avg_game_score = total_game_score / n_episodes  # New: Calculate average game score
        
        # Calculate per-category accuracies
        category_accuracies = {
            cat: stats['correct'] / max(stats['total'], 1)
            for cat, stats in per_category_accuracy.items()
        }
        
        return {
            'dice_accuracy': dice_accuracy,
            'category_accuracy': category_accuracy,
            'avg_dice_similarity': avg_dice_similarity,
            'avg_game_score': avg_game_score,  # New: Include average game score
            'per_category_accuracy': category_accuracies
        }
    
    def pretrain(self):
        """Run pre-training using the teacher (Greedy3) as supervision."""
        logger.info(f"Starting pre-training for {self.n_episodes} episodes")
        start_time = time.time()
        
        # Training loop
        for episode in tqdm(range(self.n_episodes)):
            # Run episode and collect experience
            episode_score = self.pretrain_episode()
            
            # Only update after collecting multiple episodes
            if (episode + 1) % self.episodes_per_update == 0:
                # Train on replay buffer
                dice_loss, category_loss = self.update_from_replay()
                
                # Store losses
                self.dice_losses.append(dice_loss)
                self.category_losses.append(category_loss)
                
                # Log training metrics
                self.writer.add_scalar('train/dice_loss', dice_loss, episode)
                self.writer.add_scalar('train/category_loss', category_loss, episode)
                self.writer.add_scalar('train/game_score', episode_score, episode)  # New: Log game score
                self.writer.add_scalar('train/learning_rate', self.optimizer.param_groups[0]['lr'], episode)
            
            # Regular checkpointing
            if (episode + 1) % self.checkpoint_interval == 0:
                metrics = {
                    'dice_loss': self.dice_losses[-1] if self.dice_losses else 0,
                    'category_loss': self.category_losses[-1] if self.category_losses else 0,
                    'game_score': episode_score,  # New: Include game score in metrics
                    'learning_rate': self.optimizer.param_groups[0]['lr'],
                    'episode': episode
                }
                self.save_checkpoint(episode, metrics)
            
            # Evaluate periodically
            if (episode + 1) % self.eval_interval == 0:
                eval_metrics = self.evaluate()
                
                # Log evaluation metrics
                self.log_metrics(eval_metrics, episode)
                
                # Plot and log confusion matrix
                self.plot_confusion_matrix(episode)
                
                # Track best model using combined metric including game score
                combined_score = (
                    eval_metrics['dice_accuracy'] + 
                    2 * eval_metrics['category_accuracy'] +  # Weight category accuracy more
                    eval_metrics['avg_game_score'] / 200.0  # Normalize game score to similar scale
                )
                
                if combined_score > self.best_eval_score:
                    self.best_eval_score = combined_score
                    self.save_checkpoint(episode, eval_metrics, is_best=True)
                    logger.info(f"New best model with combined score: {combined_score:.4f}")
                
                # Log detailed metrics
                logger.info(f"Episode {episode + 1}")
                logger.info(f"Dice Accuracy: {eval_metrics['dice_accuracy']:.4f}")
                logger.info(f"Category Accuracy: {eval_metrics['category_accuracy']:.4f}")
                logger.info(f"Average Game Score: {eval_metrics['avg_game_score']:.1f}")
                logger.info(f"Average Dice Similarity: {eval_metrics['avg_dice_similarity']:.4f}")
                
                # Store history for plotting
                self.dice_accuracy_history.append(eval_metrics['dice_accuracy'])
                self.category_accuracy_history.append(eval_metrics['category_accuracy'])
                self.game_score_history.append(eval_metrics['avg_game_score'])
        
        # Final evaluation and saving
        final_metrics = self.evaluate()
        self.save_checkpoint(self.n_episodes - 1, final_metrics)
        self.plot_confusion_matrix(self.n_episodes - 1)
        
        # Log training duration
        duration = time.time() - start_time
        logger.info(f"Pre-training completed in {duration:.2f} seconds")
        logger.info(f"Final Dice Accuracy: {final_metrics['dice_accuracy']:.4f}")
        logger.info(f"Final Category Accuracy: {final_metrics['category_accuracy']:.4f}")
        logger.info(f"Final Average Game Score: {final_metrics['avg_game_score']:.1f}")
        
        return final_metrics

    def update_from_replay(self):
        """Train on a batch from the replay buffer."""
        if len(self.replay_buffer) < self.batch_size:
            return 0, 0
            
        # Sample batch
        batch = random.sample(self.replay_buffer, self.batch_size)
        obs_batch, teacher_dice_batch, teacher_category_batch = zip(*batch)
        
        # Convert to tensors
        obs_tensor = torch.stack([self._process_observation(obs) for obs in obs_batch])
        dice_tensor = torch.stack([teacher_dice.clone().detach() for teacher_dice in teacher_dice_batch])
        category_tensor = torch.tensor(teacher_category_batch, device=self.device)
        
        # Forward pass
        dice_logits, category_logits, _ = self.student.ac_net(obs_tensor)
        
        # Calculate losses with weights
        dice_loss = self.dice_loss_weight * F.binary_cross_entropy_with_logits(
            dice_logits,
            dice_tensor.float()
        )
        category_loss = self.category_loss_weight * F.cross_entropy(
            category_logits,
            category_tensor
        )
        
        # Total loss
        total_loss = dice_loss + category_loss
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.student.ac_net.parameters(), self.max_grad_norm)
        self.optimizer.step()
        
        # Step the learning rate scheduler
        self.scheduler.step()
        
        return dice_loss.item(), category_loss.item()
        
    def _process_observation(self, obs):
        """Convert observation to tensor."""
        if isinstance(obs, torch.Tensor):
            return obs.to(self.device)
        return self.student.ac_net._process_observation(obs)

if __name__ == "__main__":
    # Training configuration
    config = {
        "batch_size": 128,
        "learning_rate": 1e-3,  # Reduced from 5e-0 to 1e-3
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