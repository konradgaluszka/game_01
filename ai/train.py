"""
Reinforcement Learning training script for soccer AI.
Uses stable-baselines3 with PPO algorithm.
"""

import os
import sys
import time
from datetime import datetime

# Add parent directory to path so we can import from ai module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor

from ai.soccer_env import SoccerEnv


class IterativeSelfPlayCallback(BaseCallback):
    """
    Custom callback for iterative self-play training.
    Updates the opponent model with the latest trained model at regular intervals.
    """
    
    def __init__(self, env, update_freq=10000, model_save_path="ai/models/checkpoints/", model_name="iterative", verbose=1):
        super().__init__(verbose)
        self.env = env
        self.update_freq = update_freq
        self.model_save_path = model_save_path
        self.model_name = model_name
        self.last_update_step = 0
        
    def _on_step(self) -> bool:
        # Check if it's time to update opponent model
        if self.num_timesteps - self.last_update_step >= self.update_freq:
            self._update_opponent_model()
            self.last_update_step = self.num_timesteps
        return True
    
    def _update_opponent_model(self):
        """Save current model and update opponent to use it"""
        try:
            # Save current model as new opponent
            current_model_path = f"{self.model_save_path}{self.model_name}_{self.num_timesteps}_steps.zip"
            self.model.save(current_model_path)
            
            if self.verbose >= 1:
                print(f"\n[Iterative Self-Play] Saved model at step {self.num_timesteps}: {current_model_path}")
            
            # Update opponent model in all environments
            if hasattr(self.env, 'envs'):
                # VecEnv - update all environments
                for env in self.env.envs:
                    if hasattr(env, 'unwrapped'):
                        soccer_env = env.unwrapped
                        if hasattr(soccer_env, 'update_opponent_model'):
                            soccer_env.update_opponent_model(current_model_path)
                            if self.verbose >= 1:
                                print(f"[Iterative Self-Play] Updated opponent model in environment")
            elif hasattr(self.env, 'update_opponent_model'):
                # Single environment
                self.env.update_opponent_model(current_model_path)
                if self.verbose >= 1:
                    print(f"[Iterative Self-Play] Updated opponent model in single environment")
                    
        except Exception as e:
            if self.verbose >= 1:
                print(f"[Iterative Self-Play] Error updating opponent model: {e}")


def create_env(curriculum=False, phase=None, total_timesteps=0, self_play=False, opponent_model_path=None):
    """Create and wrap the soccer environment with curriculum and self-play support"""
    return SoccerEnv(render_mode=None, curriculum=curriculum, phase=phase, total_timesteps=total_timesteps, 
                    self_play=self_play, opponent_model_path=opponent_model_path)


def train_model(total_timesteps=100000, model_name="soccer_ai", curriculum=False, phase=None, load_model=None, self_play=False, opponent_model_path=None, iterative_selfplay=False, update_freq=10000):
    """Train the PPO model on the soccer environment with curriculum, self-play, and iterative self-play support"""
    
    print(f"Starting training for {total_timesteps} timesteps...")
    
    # Create directories
    os.makedirs("ai/models", exist_ok=True)
    os.makedirs("ai/logs", exist_ok=True)
    
    # Create vectorized environment with curriculum and self-play support
    def make_curriculum_env():
        return create_env(curriculum=curriculum, phase=phase, total_timesteps=0, 
                         self_play=self_play, opponent_model_path=opponent_model_path)
    
    env = make_vec_env(make_curriculum_env, n_envs=1)
    
    # Create or load model
    if load_model and os.path.exists(load_model):
        print(f"Loading existing model from {load_model}")
        model = PPO.load(load_model, env=env)
        # Update tensorboard log name for continued training
        model.tensorboard_log = "ai/logs/"
    else:
        # Create new model with better hyperparameters for soccer
        print("Creating new model...")
        model = PPO(
            "MlpPolicy",  # Multi-layer perceptron policy
            env,
            verbose=1,
            tensorboard_log="ai/logs/",
            learning_rate=1e-4,  # Lower learning rate for stability
            n_steps=1024,  # Smaller steps for faster updates
            batch_size=128,  # Larger batch size
            n_epochs=20,  # More epochs per update
            gamma=0.98,  # Slightly lower discount for shorter episodes
            gae_lambda=0.95,
            clip_range=0.1,  # Smaller clip range for stability
            ent_coef=0.05,  # Higher entropy for exploration
            policy_kwargs=dict(
                net_arch=[256, 256, 128]  # Larger network
            )
        )
    
    # Print training configuration
    if self_play:
        print(f"Training with self-play")
        if iterative_selfplay:
            print(f"Iterative self-play enabled - updating opponent every {update_freq} steps")
        if opponent_model_path:
            print(f"Initial opponent model: {opponent_model_path}")
        else:
            print("No opponent model specified - using rule-based opponent")
    
    if curriculum:
        print(f"Training with curriculum learning")
        if phase:
            print(f"Manual phase: {phase}")
        else:
            print("Automatic phase progression based on timesteps")
    elif phase:
        print(f"Training with manual phase: {phase}")
    else:
        print("Training with full difficulty (no curriculum)")
    
    # Callbacks for saving and evaluation
    callbacks = []
    
    # Standard checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path="ai/models/checkpoints/",
        name_prefix=model_name
    )
    callbacks.append(checkpoint_callback)
    
    # Add iterative self-play callback if enabled
    if iterative_selfplay and self_play:
        iterative_callback = IterativeSelfPlayCallback(
            env=env,
            update_freq=update_freq,
            model_save_path="ai/models/checkpoints/",
            model_name=f"{model_name}_iterative",
            verbose=1
        )
        callbacks.append(iterative_callback)
        print(f"Added iterative self-play callback - opponent updates every {update_freq} steps")
    
    # Train the model
    start_time = time.time()
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        tb_log_name=f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Save final model
    final_model_path = f"ai/models/{model_name}_final.zip"
    model.save(final_model_path)
    print(f"Model saved to {final_model_path}")
    
    return model


def evaluate_model(model_path, n_episodes=10):
    """Evaluate a trained model"""
    print(f"Evaluating model: {model_path}")
    
    # Load model
    model = PPO.load(model_path)
    
    # Create environment for evaluation
    env = SoccerEnv(render_mode="human")  # With rendering for evaluation
    
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(n_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            episode_length += 1
            done = terminated or truncated
            
            env.render()
            
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, Length = {episode_length}")
    
    print(f"Average reward: {sum(episode_rewards) / len(episode_rewards):.2f}")
    print(f"Average episode length: {sum(episode_lengths) / len(episode_lengths):.2f}")
    
    env.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Soccer AI Training Script with Curriculum Learning')
    parser.add_argument('command', choices=['train', 'evaluate'], help='Command to execute')
    parser.add_argument('--timesteps', type=int, default=100000, help='Total training timesteps')
    parser.add_argument('--model-name', default="soccer_ai", help='Model name for saving')
    parser.add_argument('--curriculum', action='store_true', help='Enable curriculum learning')
    parser.add_argument('--phase', choices=['ball_awareness', 'basic_soccer', 'competitive_soccer'], help='Manual phase selection')
    parser.add_argument('--load', help='Path to existing model to continue training')
    parser.add_argument('--self-play', action='store_true', help='Enable self-play training')
    parser.add_argument('--opponent-model', help='Path to opponent model for self-play')
    parser.add_argument('--iterative-selfplay', action='store_true', help='Enable iterative self-play (updates opponent with latest model)')
    parser.add_argument('--update-freq', type=int, default=5000, help='Frequency to update opponent model in iterative self-play')
    parser.add_argument('--model-path', default="ai/models/soccer_ai_final.zip", help='Model path for evaluation')
    parser.add_argument('--episodes', type=int, default=10, help='Number of episodes for evaluation')
    
    args = parser.parse_args()
    
    if args.command == "train":
        train_model(
            total_timesteps=args.timesteps,
            model_name=args.model_name,
            curriculum=args.curriculum,
            phase=args.phase,
            load_model=args.load,
            self_play=args.self_play,
            opponent_model_path=args.opponent_model,
            iterative_selfplay=args.iterative_selfplay,
            update_freq=args.update_freq
        )
    elif args.command == "evaluate":
        evaluate_model(args.model_path, args.episodes)
    
    # Examples:
    # Curriculum learning:
    # python ai/train.py train --timesteps 100000 --model-name curriculum_ai --curriculum
    # python ai/train.py train --timesteps 25000 --model-name phase1_ai --phase ball_awareness
    # python ai/train.py train --timesteps 50000 --model-name phase2_ai --phase basic_soccer --load ai/models/phase1_ai_final.zip
    
    # Self-play training:
    # python ai/train.py train --timesteps 50000 --model-name selfplay_ai --self-play --opponent-model ai/models/improved_ball_chasing_final.zip
    # python ai/train.py train --timesteps 100000 --model-name selfplay_curriculum --curriculum --self-play --opponent-model ai/models/curriculum_ai_final.zip
    
    # Iterative self-play (opponent updates during training):
    # python ai/train.py train --timesteps 50000 --model-name iterative_selfplay --self-play --iterative-selfplay --opponent-model ai/models/basic_ai_final.zip --update-freq 5000
    # python ai/train.py train --timesteps 100000 --model-name iterative_curriculum --curriculum --self-play --iterative-selfplay --opponent-model ai/models/phase1_ai_final.zip
    
    # Evaluation:
    # python ai/train.py evaluate --model-path ai/models/selfplay_ai_final.zip --episodes 5