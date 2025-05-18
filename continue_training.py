import torch
from train_rl_agent import YahtzeeTrainer
import logging
import datetime
from torch.serialization import add_safe_globals
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Add safe globals for model loading
    add_safe_globals([
        np._core.multiarray.scalar,
        np._core.multiarray._reconstruct,
        np.ndarray,
        np.dtype
    ])
    
    # Training parameters
    params = {
        'n_episodes': 10000,  # Total episodes for continued training
        'batch_size': 32,
        'learning_rate': 5e-3,  # Optimized learning rate from previous runs
        'gamma': 0.99,  # Discount factor
        'gae_lambda': 0.95,  # GAE lambda parameter
        'entropy_coef': 0.01,  # Entropy coefficient for exploration
        'value_loss_coef': 0.5,  # Value loss coefficient
        'max_grad_norm': 0.5,  # Gradient clipping
        'eval_frequency': 100,  # Evaluate every 100 episodes
        'eval_episodes': 10,  # Number of episodes for evaluation
        'checkpoint_frequency': 1000,  # Save checkpoint every 1000 episodes
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }

    # Create a unique experiment name with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"actor_critic_training_{timestamp}"

    # Path to the pre-trained model
    pretrained_model_path = "checkpoints/yahtzee_pretrain_faster/best_model.pt"

    # Initialize trainer with pre-trained model
    trainer = YahtzeeTrainer(
        experiment_name=experiment_name,
        pretrained_model_path=pretrained_model_path,
        **params
    )

    # Start training
    logger.info(f"Starting actor-critic training from {pretrained_model_path}")
    logger.info(f"Experiment name: {experiment_name}")
    logger.info(f"Training device: {params['device']}")
    
    try:
        trainer.train()
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise
    finally:
        # Close tensorboard writer
        trainer.writer.close()

if __name__ == "__main__":
    main() 