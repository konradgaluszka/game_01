"""
Simplified CTDE Training Script

This version uses a simpler approach that works better with stable-baselines3:
- Uses flattened observations (345-dim) with standard PPO policy
- Enhances the environment to provide agent-specific features
- Maintains the CTDE observation structure but uses standard training
"""

import os
import sys
import time
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv

from ai.soccer_env_ctde import SoccerEnvCTDE


class CTDELoggingCallback(BaseCallback):
    """Simple callback for CTDE training logging."""
    
    def __init__(self, verbose=1):
        super().__init__(verbose)
    
    def _on_step(self) -> bool:
        if self.num_timesteps % 5000 == 0 and self.verbose >= 1:
            print(f"\n[CTDE Simple] Step {self.num_timesteps}")
        return True


def create_simple_ctde_env(curriculum=False, phase=None, total_timesteps=0, 
                          self_play=False, opponent_model_path=None):
    """Create simplified CTDE environment."""
    env = SoccerEnvCTDE(
        render_mode=None, 
        curriculum=curriculum, 
        phase=phase, 
        total_timesteps=total_timesteps,
        self_play=self_play, 
        opponent_model_path=opponent_model_path,
        observation_mode='agent'  # Use flattened agent observations
    )
    print(f"Created CTDE environment with obs space: {env.observation_space}")
    return env


def train_simple_ctde(total_timesteps=50000, model_name="simple_ctde", curriculum=False, 
                     phase=None, load_model=None, self_play=False, opponent_model_path=None, 
                     n_envs=4):
    """
    Train soccer AI using simplified CTDE approach with standard PPO.
    
    This approach uses:
    - Enhanced CTDE observations (345-dim flattened)  
    - Standard PPO policy (no custom policy complexity)
    - Role-based features built into observations
    - Multi-environment parallelization
    """
    
    print(f"Starting Simple CTDE training for {total_timesteps} timesteps with {n_envs} environments...")
    
    # Create directories
    os.makedirs("ai/models", exist_ok=True)
    os.makedirs("ai/logs", exist_ok=True)
    os.makedirs("ai/models/checkpoints", exist_ok=True)
    
    # Create vectorized environment
    def make_env():
        return create_simple_ctde_env(
            curriculum=curriculum, 
            phase=phase, 
            total_timesteps=0,
            self_play=self_play, 
            opponent_model_path=opponent_model_path
        )
    
    env = make_vec_env(make_env, n_envs=n_envs, vec_env_cls=SubprocVecEnv)
    
    # Create or load model using standard PPO
    if load_model and os.path.exists(load_model):
        print(f"Loading existing model from {load_model}")
        model = PPO.load(load_model, env=env)
        model.tensorboard_log = "ai/logs/"
    else:
        print("Creating new PPO model with CTDE observations...")
        
        model = PPO(
            "MlpPolicy",  # Standard MLP policy
            env,
            verbose=1,
            tensorboard_log="ai/logs/",
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=128,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=dict(
                net_arch=[512, 512, 256, 256]  # Larger network for 345-dim input
            )
        )
    
    print(f"\n=== Simple CTDE Configuration ===")
    print(f"Model: {model_name}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print(f"Total timesteps: {total_timesteps}")
    print(f"Environments: {n_envs}")
    
    if curriculum:
        print(f"Curriculum learning: enabled")
    if self_play:
        print(f"Self-play: enabled")
        if opponent_model_path:
            print(f"Opponent model: {opponent_model_path}")
    
    # Setup callbacks
    callbacks = []
    
    # Logging callback
    logging_callback = CTDELoggingCallback(verbose=1)
    callbacks.append(logging_callback)
    
    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=max(10000 // n_envs, 1000),
        save_path="ai/models/checkpoints/",
        name_prefix=f"{model_name}_simple_ctde"
    )
    callbacks.append(checkpoint_callback)
    
    # Train the model
    start_time = time.time()
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        tb_log_name=f"{model_name}_simple_ctde_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    
    training_time = time.time() - start_time
    print(f"\n=== Simple CTDE Training Completed ===")
    print(f"Training time: {training_time:.2f} seconds")
    print(f"Steps per second: {total_timesteps/training_time:.2f}")
    
    # Save final model
    final_model_path = f"ai/models/{model_name}_simple_ctde_final.zip"
    model.save(final_model_path)
    print(f"Final model saved: {final_model_path}")
    
    return model


def evaluate_simple_ctde(model_path, n_episodes=10, render=True):
    """Evaluate a trained simple CTDE model."""
    print(f"Evaluating Simple CTDE model: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return
    
    model = PPO.load(model_path)
    env = SoccerEnvCTDE(
        render_mode="human" if render else None,
        observation_mode='agent'
    )
    
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(n_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        
        print(f"\n=== Episode {episode + 1} ===")
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            episode_length += 1
            done = terminated or truncated
            
            if render:
                env.render()
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        print(f"Episode {episode + 1}: Reward={episode_reward:.2f}, Length={episode_length}")
    
    print(f"\n=== Simple CTDE Evaluation Results ===")
    print(f"Average reward: {sum(episode_rewards)/len(episode_rewards):.2f}")
    print(f"Average length: {sum(episode_lengths)/len(episode_lengths):.2f}")
    
    env.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Simple CTDE Soccer AI Training')
    parser.add_argument('command', choices=['train', 'evaluate'], help='Command to execute')
    parser.add_argument('--timesteps', type=int, default=50000, help='Total training timesteps')
    parser.add_argument('--model-name', default="simple_ctde", help='Model name for saving')
    parser.add_argument('--curriculum', action='store_true', help='Enable curriculum learning')
    parser.add_argument('--phase', choices=['ball_awareness', 'basic_soccer', 'competitive_soccer'], 
                       help='Manual phase selection')
    parser.add_argument('--load', help='Path to existing model to continue training')
    parser.add_argument('--self-play', action='store_true', help='Enable self-play training')
    parser.add_argument('--opponent-model', help='Path to opponent model for self-play')
    parser.add_argument('--n-envs', type=int, default=4, help='Number of parallel environments')
    parser.add_argument('--model-path', default="ai/models/simple_ctde_simple_ctde_final.zip", 
                       help='Model path for evaluation')
    parser.add_argument('--episodes', type=int, default=10, help='Number of episodes for evaluation')
    parser.add_argument('--no-render', action='store_true', help='Disable rendering during evaluation')
    
    args = parser.parse_args()
    
    if args.command == "train":
        train_simple_ctde(
            total_timesteps=args.timesteps,
            model_name=args.model_name,
            curriculum=args.curriculum,
            phase=args.phase,
            load_model=args.load,
            self_play=args.self_play,
            opponent_model_path=args.opponent_model,
            n_envs=args.n_envs
        )
    elif args.command == "evaluate":
        evaluate_simple_ctde(
            model_path=args.model_path,
            n_episodes=args.episodes,
            render=not args.no_render
        )
    
    print("\n=== Simple CTDE Examples ===")
    print("# Basic simple CTDE:")
    print("python ai/train_ctde_simple.py train --timesteps 50000 --model-name basic")
    print()
    print("# With curriculum:")
    print("python ai/train_ctde_simple.py train --timesteps 50000 --model-name curriculum --curriculum")
    print()
    print("# Evaluation:")
    print("python ai/train_ctde_simple.py evaluate --model-path ai/models/basic_simple_ctde_final.zip")