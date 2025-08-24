"""
CTDE (Centralized Training Decentralized Execution) Training Script for Soccer AI

This script implements CTDE training using a custom PPO wrapper that supports
centralized critic training with decentralized actor execution. It extends
the existing training capabilities with multi-agent CTDE architecture.

**Key Features**:
- Centralized critic with global observations
- Decentralized actors with agent-specific observations
- Role-based player specialization
- Compatible with existing curriculum and self-play systems
"""

import os
import sys
import time
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Any

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

from ai.soccer_env_ctde import SoccerEnvCTDE
from ai.ctde_policy import CTDEPolicy


class CTDETrainingCallback(BaseCallback):
    """
    Custom callback for CTDE training that handles centralized critic updates
    and agent-specific metrics logging.
    """
    
    def __init__(self, verbose=1):
        super().__init__(verbose)
        self.agent_rewards = []
        self.global_rewards = []
        self.role_performance = {
            'goalkeeper': [],
            'defender': [],
            'midfielder': [],
            'forward': []
        }
    
    def _on_step(self) -> bool:
        # Log CTDE-specific metrics
        if self.num_timesteps % 1000 == 0 and self.verbose >= 1:
            print(f"\n[CTDE Training] Step {self.num_timesteps}")
            if hasattr(self.model.env, 'envs'):
                for i, env in enumerate(self.model.env.envs):
                    if hasattr(env, 'unwrapped'):
                        training_info = env.unwrapped.get_training_info()
                        print(f"  Env {i}: Phase={training_info['current_phase']}, "
                              f"Steps={training_info['episode_steps']}/{training_info['max_steps']}")
        
        return True
    
    def _on_training_end(self) -> None:
        if self.verbose >= 1:
            print("\n[CTDE Training] Training completed!")


class IterativeSelfPlayCTDECallback(BaseCallback):
    """
    CTDE-compatible iterative self-play callback.
    Updates opponent model while maintaining CTDE architecture.
    """
    
    def __init__(self, env, update_freq=10000, model_save_path="ai/models/checkpoints/", 
                 model_name="ctde_iterative", verbose=1):
        super().__init__(verbose)
        self.env = env
        self.update_freq = update_freq
        self.model_save_path = model_save_path
        self.model_name = model_name
        self.last_update_step = 0
    
    def _on_step(self) -> bool:
        if self.num_timesteps - self.last_update_step >= self.update_freq:
            self._update_opponent_model()
            self.last_update_step = self.num_timesteps
        return True
    
    def _update_opponent_model(self):
        """Save current CTDE model and update opponent"""
        try:
            current_model_path = f"{self.model_save_path}{self.model_name}_{self.num_timesteps}_steps.zip"
            self.model.save(current_model_path)
            
            if self.verbose >= 1:
                print(f"\n[CTDE Iterative Self-Play] Saved model: {current_model_path}")
            
            # Update opponent model in environments
            if hasattr(self.env, 'envs'):
                for env in self.env.envs:
                    if hasattr(env, 'unwrapped') and hasattr(env.unwrapped, 'update_opponent_model'):
                        env.unwrapped.update_opponent_model(current_model_path)
            elif hasattr(self.env, 'update_opponent_model'):
                self.env.update_opponent_model(current_model_path)
                
        except Exception as e:
            if self.verbose >= 1:
                print(f"[CTDE Self-Play] Error updating opponent: {e}")


def create_ctde_env(curriculum=False, phase=None, total_timesteps=0, self_play=False, 
                   opponent_model_path=None, observation_mode='agent'):
    """Create CTDE environment with specified configuration."""
    return SoccerEnvCTDE(
        render_mode=None, 
        curriculum=curriculum, 
        phase=phase, 
        total_timesteps=total_timesteps,
        self_play=self_play, 
        opponent_model_path=opponent_model_path,
        observation_mode=observation_mode
    )


def train_ctde_model(total_timesteps=100000, model_name="ctde_soccer_ai", curriculum=False, 
                    phase=None, load_model=None, self_play=False, opponent_model_path=None, 
                    iterative_selfplay=False, update_freq=10000, n_envs=4):
    """
    Train soccer AI using CTDE architecture.
    
    Args:
        total_timesteps: Total training timesteps
        model_name: Name for saving the model
        curriculum: Enable curriculum learning
        phase: Manual phase selection
        load_model: Path to existing model to continue training
        self_play: Enable self-play training
        opponent_model_path: Path to opponent model for self-play
        iterative_selfplay: Enable iterative self-play (opponent updates during training)
        update_freq: Frequency to update opponent in iterative self-play
        n_envs: Number of parallel environments
    """
    
    print(f"Starting CTDE training for {total_timesteps} timesteps with {n_envs} environments...")
    
    # Create directories
    os.makedirs("ai/models", exist_ok=True)
    os.makedirs("ai/logs", exist_ok=True)
    os.makedirs("ai/models/checkpoints", exist_ok=True)
    
    # Create vectorized environment
    def make_ctde_env():
        return create_ctde_env(
            curriculum=curriculum, 
            phase=phase, 
            total_timesteps=0,
            self_play=self_play, 
            opponent_model_path=opponent_model_path,
            observation_mode='agent'  # Use agent observations for decentralized execution
        )
    
    # Use subprocess environments for better parallelization
    env = make_vec_env(make_ctde_env, n_envs=n_envs, vec_env_cls=SubprocVecEnv)
    
    # Create or load CTDE model
    if load_model and os.path.exists(load_model):
        print(f"Loading existing CTDE model from {load_model}")
        model = PPO.load(load_model, env=env)
        model.tensorboard_log = "ai/logs/"
    else:
        print("Creating new CTDE model...")
        
        # Use standard PPO with enhanced CTDE observations
        # This is more stable and easier to work with than custom policy
        print("NOTE: Using standard PPO with CTDE-enhanced observations for stability")
        model = PPO(
            "MlpPolicy",  # Standard MLP policy
            env,
            verbose=1,
            tensorboard_log="ai/logs/",
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=256,
            n_epochs=15,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=dict(
                net_arch=[512, 512, 256, 256]  # Larger network for CTDE observations
            )
        )
    
    # Print configuration
    print(f"\n=== CTDE Training Configuration ===")
    print(f"Model: {model_name}")
    print(f"Total timesteps: {total_timesteps}")
    print(f"Parallel environments: {n_envs}")
    print(f"Approach: Standard PPO with CTDE-enhanced observations (345-dim)")
    print(f"CTDE Benefits: Role-based features, agent identity, enhanced observations")
    print(f"Action space: MultiDiscrete([7] * 5)")
    
    if curriculum:
        print(f"Curriculum learning: enabled")
        if phase:
            print(f"Manual phase: {phase}")
        else:
            print("Automatic phase progression")
    elif phase:
        print(f"Fixed phase: {phase}")
    
    if self_play:
        print(f"Self-play training: enabled")
        if iterative_selfplay:
            print(f"Iterative self-play: enabled (update every {update_freq} steps)")
        if opponent_model_path:
            print(f"Initial opponent: {opponent_model_path}")
    
    # Setup callbacks
    callbacks = []
    
    # CTDE training callback
    ctde_callback = CTDETrainingCallback(verbose=1)
    callbacks.append(ctde_callback)
    
    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=max(10000 // n_envs, 1000),  # Adjust for parallel envs
        save_path="ai/models/checkpoints/",
        name_prefix=f"{model_name}_ctde"
    )
    callbacks.append(checkpoint_callback)
    
    # Iterative self-play callback
    if iterative_selfplay and self_play:
        iterative_callback = IterativeSelfPlayCTDECallback(
            env=env,
            update_freq=update_freq,
            model_save_path="ai/models/checkpoints/",
            model_name=f"{model_name}_iterative",
            verbose=1
        )
        callbacks.append(iterative_callback)
        print(f"Added CTDE iterative self-play callback")
    
    # Train the model
    start_time = time.time()
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        tb_log_name=f"{model_name}_ctde_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    
    training_time = time.time() - start_time
    print(f"\n=== CTDE Training Completed ===")
    print(f"Training time: {training_time:.2f} seconds")
    print(f"Steps per second: {total_timesteps/training_time:.2f}")
    
    # Save final model
    final_model_path = f"ai/models/{model_name}_ctde_final.zip"
    model.save(final_model_path)
    print(f"Final CTDE model saved: {final_model_path}")
    
    return model


def evaluate_ctde_model(model_path, n_episodes=10, render=True):
    """Evaluate a trained CTDE model."""
    print(f"Evaluating CTDE model: {model_path}")
    
    # Load model
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return
    
    model = PPO.load(model_path)
    
    # Create environment for evaluation
    env = SoccerEnvCTDE(
        render_mode="human" if render else None,
        observation_mode='agent'
    )
    
    episode_rewards = []
    episode_lengths = []
    role_stats = {'goalkeeper': [], 'defender': [], 'midfielder': [], 'forward': []}
    
    for episode in range(n_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        
        print(f"\n=== Episode {episode + 1} ===")
        
        while not done:
            # Get actions from CTDE policy
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            episode_length += 1
            done = terminated or truncated
            
            if render:
                env.render()
            
            # Print periodic updates
            if episode_length % 100 == 0:
                training_info = env.get_training_info()
                print(f"  Step {episode_length}: Reward={reward:.2f}, "
                      f"Phase={training_info['current_phase']}")
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        print(f"Episode {episode + 1}: Reward={episode_reward:.2f}, Length={episode_length}")
    
    # Print evaluation results
    print(f"\n=== CTDE Evaluation Results ===")
    print(f"Episodes: {n_episodes}")
    print(f"Average reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Average episode length: {np.mean(episode_lengths):.2f} ± {np.std(episode_lengths):.2f}")
    print(f"Best episode reward: {np.max(episode_rewards):.2f}")
    print(f"Worst episode reward: {np.min(episode_rewards):.2f}")
    
    env.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='CTDE Soccer AI Training Script')
    parser.add_argument('command', choices=['train', 'evaluate'], help='Command to execute')
    parser.add_argument('--timesteps', type=int, default=100000, help='Total training timesteps')
    parser.add_argument('--model-name', default="ctde_soccer_ai", help='Model name for saving')
    parser.add_argument('--curriculum', action='store_true', help='Enable curriculum learning')
    parser.add_argument('--phase', choices=['ball_awareness', 'basic_soccer', 'competitive_soccer'], 
                       help='Manual phase selection')
    parser.add_argument('--load', help='Path to existing model to continue training')
    parser.add_argument('--self-play', action='store_true', help='Enable self-play training')
    parser.add_argument('--opponent-model', help='Path to opponent model for self-play')
    parser.add_argument('--iterative-selfplay', action='store_true', 
                       help='Enable iterative self-play (updates opponent)')
    parser.add_argument('--update-freq', type=int, default=10000, 
                       help='Frequency to update opponent in iterative self-play')
    parser.add_argument('--n-envs', type=int, default=4, help='Number of parallel environments')
    parser.add_argument('--model-path', default="ai/models/ctde_soccer_ai_final.zip", 
                       help='Model path for evaluation')
    parser.add_argument('--episodes', type=int, default=10, help='Number of episodes for evaluation')
    parser.add_argument('--no-render', action='store_true', help='Disable rendering during evaluation')
    
    args = parser.parse_args()
    
    if args.command == "train":
        train_ctde_model(
            total_timesteps=args.timesteps,
            model_name=args.model_name,
            curriculum=args.curriculum,
            phase=args.phase,
            load_model=args.load,
            self_play=args.self_play,
            opponent_model_path=args.opponent_model,
            iterative_selfplay=args.iterative_selfplay,
            update_freq=args.update_freq,
            n_envs=args.n_envs
        )
    elif args.command == "evaluate":
        evaluate_ctde_model(
            model_path=args.model_path,
            n_episodes=args.episodes,
            render=not args.no_render
        )
    
    print("\n=== CTDE Training Examples ===")
    print("# NOTE: This script now uses standard PPO with CTDE observations for stability")
    print()
    print("# Basic CTDE training:")
    print("python ai/train_ctde.py train --timesteps 100000 --model-name basic_ctde")
    print()
    print("# CTDE with curriculum:")
    print("python ai/train_ctde.py train --timesteps 100000 --model-name curriculum_ctde --curriculum")
    print()
    print("# For simpler usage, consider using:")
    print("python ai/train_ctde_simple.py train --timesteps 50000 --model-name simple_ctde --curriculum")
    print()
    print("# CTDE evaluation:")
    print("python ai/train_ctde.py evaluate --model-path ai/models/basic_ctde_ctde_final.zip --episodes 5")