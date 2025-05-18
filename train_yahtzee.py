import argparse
from pretrain_rl_agent import PreTrainer
from train_rl_agent import YahtzeeTrainer
import logging
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Train Yahtzee AI with pre-training')
    parser.add_argument('--skip-pretrain', action='store_true', help='Skip pre-training phase')
    parser.add_argument('--pretrain-episodes', type=int, default=10000, help='Number of pre-training episodes')
    parser.add_argument('--rl-episodes', type=int, default=100000, help='Number of RL training episodes')
    parser.add_argument('--experiment-name', type=str, default=None, help='Name for the experiment')
    parser.add_argument('--device', type=str, default=None, help='Device to train on (cuda/cpu)')
    parser.add_argument('--pretrained-model', type=str, default=None, 
                       help='Path to existing pre-trained model (skips pre-training)')
    args = parser.parse_args()

    # Determine device
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    logger.info(f"Using device: {device}")

    # Phase 1: Pre-training on Greedy3 AI
    if not args.skip_pretrain and not args.pretrained_model:
        logger.info("Starting pre-training phase...")
        pretrainer = PreTrainer(
            n_episodes=args.pretrain_episodes,
            batch_size=128,
            learning_rate=1e-3,
            experiment_name=f"{args.experiment_name}_pretrain" if args.experiment_name else None,
            device=device,
            replay_buffer_size=100000,
            dice_loss_weight=1.0,
            category_loss_weight=5.0,
            episodes_per_update=10
        )
        pretrained_model_path = pretrainer.pretrain()
        logger.info(f"Pre-training completed. Model saved at: {pretrained_model_path}")
    else:
        pretrained_model_path = args.pretrained_model
        if pretrained_model_path:
            logger.info(f"Using existing pre-trained model: {pretrained_model_path}")
        else:
            logger.info("Skipping pre-training phase")
            pretrained_model_path = None

    # Phase 2: RL Training
    logger.info("Starting RL training phase...")
    trainer = YahtzeeTrainer(
        n_episodes=args.rl_episodes,
        batch_size=32,
        learning_rate=1e-3,
        gamma=0.99,
        gae_lambda=0.2,
        entropy_coef=1e-3,
        value_loss_coef=0.5,
        max_grad_norm=0.5,
        eval_frequency=100,
        eval_episodes=10,
        checkpoint_frequency=1000,
        experiment_name=f"{args.experiment_name}_rl" if args.experiment_name else None,
        device=device,
        pretrained_model_path=pretrained_model_path
    )
    
    # Start RL training
    trainer.train()
    logger.info("Training completed!")

if __name__ == "__main__":
    main() 