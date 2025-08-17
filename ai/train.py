"""
Reinforcement Learning training script for soccer AI.
Uses stable-baselines3 with PPO algorithm.
"""

import os
import time
from datetime import datetime

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from ai.soccer_env import SoccerEnv


def create_env():
    """Create and wrap the soccer environment"""
    return SoccerEnv(render_mode=None)  # No rendering during training


def train_model(total_timesteps=100000, model_name="soccer_ai"):
    """Train the PPO model on the soccer environment"""
    
    print(f"Starting training for {total_timesteps} timesteps...")
    
    # Create directories
    os.makedirs("ai/models", exist_ok=True)
    os.makedirs("ai/logs", exist_ok=True)
    
    # Create vectorized environment (single env for now)
    env = make_vec_env(create_env, n_envs=1)
    
    # Create model
    model = PPO(
        "MlpPolicy",  # Multi-layer perceptron policy
        env,
        verbose=1,
        tensorboard_log="ai/logs/",
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
    )
    
    # Callbacks for saving and evaluation
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path="ai/models/checkpoints/",
        name_prefix=model_name
    )
    
    # Train the model
    start_time = time.time()
    model.learn(
        total_timesteps=total_timesteps,
        callback=checkpoint_callback,
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
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "train":
            # Training mode
            timesteps = int(sys.argv[2]) if len(sys.argv) > 2 else 100000
            model_name = sys.argv[3] if len(sys.argv) > 3 else "soccer_ai"
            train_model(total_timesteps=timesteps, model_name=model_name)
            
        elif sys.argv[1] == "evaluate":
            # Evaluation mode
            model_path = sys.argv[2] if len(sys.argv) > 2 else "ai/models/soccer_ai_final.zip"
            n_episodes = int(sys.argv[3]) if len(sys.argv) > 3 else 10
            evaluate_model(model_path, n_episodes)
            
        else:
            print("Usage:")
            print("  python ai/train.py train [timesteps] [model_name]")
            print("  python ai/train.py evaluate [model_path] [n_episodes]")
    else:
        # Default: train for 100k timesteps
        train_model(total_timesteps=100000, model_name="soccer_ai")