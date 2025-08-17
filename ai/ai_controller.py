"""
AI Controller for team_2 using trained reinforcement learning models.
"""

import os
import numpy as np
from typing import Optional, List
from stable_baselines3 import PPO

from common.Vector import Vector


class AIController:
    """Controls team_2 using a trained RL model"""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.model_path = model_path
        self.is_loaded = False
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def load_model(self, model_path: str) -> bool:
        """Load a trained model from file"""
        try:
            self.model = PPO.load(model_path)
            self.model_path = model_path
            self.is_loaded = True
            print(f"AI model loaded from: {model_path}")
            return True
        except Exception as e:
            print(f"Failed to load AI model from {model_path}: {e}")
            self.is_loaded = False
            return False
    
    def get_observation(self, ball, all_players, match, width=800, height=600) -> np.ndarray:
        """Convert game state to observation format expected by the model"""
        obs = []
        
        # Ball position and velocity
        ball_pos = ball.ball_body.position
        ball_vel = ball.ball_body.velocity
        obs.extend([ball_pos.x / width, ball_pos.y / height])
        obs.extend([ball_vel.x / 1000, ball_vel.y / 1000])  # Normalize velocity
        
        # All players positions and velocities
        for player in all_players:
            pos = player.player_body.position
            vel = player.player_body.velocity
            obs.extend([pos.x / width, pos.y / height])
            obs.extend([vel.x / 1000, vel.y / 1000])
            
        # Goal positions (normalized)
        obs.extend([0.0, 0.5])  # Left goal center
        obs.extend([1.0, 0.5])  # Right goal center
        
        # Match state
        obs.extend([
            float(match.goal_scored),
            float(0.5)  # Normalized time (placeholder)
        ])
        
        return np.array(obs, dtype=np.float32)
    
    def get_actions(self, ball, all_players, match) -> List[int]:
        """Get actions for team_2 players from the AI model"""
        if not self.is_loaded or self.model is None:
            # Return no actions (do nothing) if model not loaded
            return [0, 0, 0, 0, 0]  # 5 players, action 0 = do nothing
        
        # Get observation
        obs = self.get_observation(ball, all_players, match)
        
        try:
            # Predict actions
            actions, _ = self.model.predict(obs, deterministic=True)
            return actions.tolist()
        except Exception as e:
            print(f"Error predicting actions: {e}")
            return [0, 0, 0, 0, 0]
    
    def control_team(self, team_2, ball, all_players, match):
        """Apply AI control to team_2 players"""
        if not self.is_loaded:
            return
            
        actions = self.get_actions(ball, all_players, match)
        team_2_players = team_2.players()
        
        for i, action in enumerate(actions):
            if i >= len(team_2_players):
                break
                
            player = team_2_players[i]
            
            # Apply the same action mapping as in training
            if action == 1:  # Up
                player.player_body.apply_force_at_local_point((0, -player.force))
            elif action == 2:  # Down
                player.player_body.apply_force_at_local_point((0, player.force))
            elif action == 3:  # Left
                player.player_body.apply_force_at_local_point((-player.force, 0))
            elif action == 4:  # Right
                player.player_body.apply_force_at_local_point((player.force, 0))
            elif action == 5:  # Shoot/Dribble
                diff = player.ball.ball_body.position - player.player_body.position
                if diff.length < player.DRIBBLE_DISTANCE:
                    direction = diff.normalized()
                    player.remove_ball_springs()
                    player.ball.ball_body.apply_impulse_at_local_point(
                        direction * player.SHOT_STRENGTH
                    )