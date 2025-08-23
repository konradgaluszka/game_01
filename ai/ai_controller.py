"""
AI Controller for Real-Time Soccer Gameplay

This module bridges trained reinforcement learning models with the live soccer game.
It loads PPO models trained in the SoccerEnv and translates their decisions into
real-time player actions during gameplay.

**Key Functions**:
1. **Model Management**: Load/unload trained RL models from .zip files
2. **State Translation**: Convert game state to training observation format  
3. **Action Execution**: Translate RL actions to player movement commands
4. **Graceful Fallback**: Handle missing models without crashing the game

**Integration**: Called every frame from main.py to control team_2 (blue team)
when a trained model is available. Provides seamless human vs AI gameplay.
"""

import os
import sys
import numpy as np
from typing import Optional, List
from stable_baselines3 import PPO

# Add parent directory to path so we can import game modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.Vector import Vector


class AIController:
    """
    Real-time AI controller that uses trained RL models to control team_2.
    
    **Purpose**: Bridge between offline RL training and live gameplay
    
    **Workflow**:
    1. Load trained PPO model from .zip file
    2. Each frame: Convert game state → observation vector
    3. Model predicts actions for all 5 team_2 players
    4. Apply actions as movement forces to player physics bodies
    
    **Key Features**:
    - Handles model loading errors gracefully
    - Maintains same observation format as training environment
    - Provides deterministic action selection for consistent play
    - Falls back to no-op if model unavailable
    """
    
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
        
        # Get ball state using public interface
        ball_pos = Vector(ball.ball_body.position.x, ball.ball_body.position.y)
        ball_vel = Vector(ball.ball_body.velocity.x, ball.ball_body.velocity.y)
        
        # Split players into teams
        team_2_players = all_players[5:]  # Last 5 players are team_2
        team_1_players = all_players[:5]  # First 5 players are team_1
        
        # === BALL INFORMATION ===
        # Ball position relative to each team_2 player
        for player in team_2_players:
            player_pos = player.position()  # Use Player's public method
            rel_x = (ball_pos.x - player_pos.x) / width
            rel_y = (ball_pos.y - player_pos.y) / height
            obs.extend([rel_x, rel_y])
        
        # Ball velocity (normalized)
        obs.extend([ball_vel.x / 1000, ball_vel.y / 1000])
        
        # === TEAM_2 PLAYER INFORMATION ===
        for player in team_2_players:
            player_pos = player.position()  # Use Player's public method
            player_vel = player.physics.get_velocity()  # Use physics subsystem
            
            # Absolute position (normalized)
            obs.extend([player_pos.x / width, player_pos.y / height])
            
            # Velocity (normalized)
            obs.extend([player_vel.x / 1000, player_vel.y / 1000])
            
            # Distance to ball (normalized)
            dist_to_ball = (ball_pos - player_pos).length() / width
            obs.append(dist_to_ball)
        
        # === TEAM_1 PLAYER INFORMATION ===
        for player in team_1_players:
            player_pos = player.position()  # Use Player's public method
            
            # Position relative to ball
            rel_x = (player_pos.x - ball_pos.x) / width
            rel_y = (player_pos.y - ball_pos.y) / height
            obs.extend([rel_x, rel_y])
            
            # Distance to nearest team_2 player
            min_dist = float('inf')
            for t2_player in team_2_players:
                t2_pos = t2_player.position()  # Use Player's public method
                dist = (player_pos - t2_pos).length()
                min_dist = min(min_dist, dist)
            obs.append(min_dist / width)
        
        # === FIELD CONTEXT ===
        # Ball distance to goals
        left_goal_dist = ball_pos.x / width  # Distance to left goal
        right_goal_dist = (width - ball_pos.x) / width  # Distance to right goal
        obs.extend([left_goal_dist, right_goal_dist])
        
        # Ball distance to field boundaries
        top_dist = ball_pos.y / height
        bottom_dist = (height - ball_pos.y) / height
        left_bound_dist = ball_pos.x / width
        right_bound_dist = (width - ball_pos.x) / width
        obs.extend([top_dist, bottom_dist, left_bound_dist, right_bound_dist])
        
        # === MATCH STATE ===
        obs.extend([
            float(match.goal1_score),  # Team_2 goals
            float(match.goal2_score),  # Team_1 goals
            float(match.goal1_score + match.goal2_score > 0),  # Goal scored this episode
            float(0.5)  # Time progress (placeholder)
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
        """Apply AI control to team_2 players using Player.apply_actions()"""
        if not self.is_loaded:
            return
            
        actions = self.get_actions(ball, all_players, match)
        team_2_players = team_2.players()
        
        for i, action in enumerate(actions):
            if i >= len(team_2_players):
                break
                
            player = team_2_players[i]
            
            # Get teammate positions for passing
            teammates_positions = [p.position() for j, p in enumerate(team_2_players) if j != i]
            
            # Map actions to movement + action booleans
            move_up = action == 1
            move_down = action == 2
            move_left = action == 3
            move_right = action == 4
            shoot = action == 5
            pass_ball = action == 6
            
            # Use Player's action system instead of direct physics manipulation
            player.apply_actions(
                move_up=move_up,
                move_down=move_down, 
                move_left=move_left,
                move_right=move_right,
                shoot=shoot,
                pass_ball=pass_ball,
                teammates_positions=teammates_positions
            )
            # Note: action 0 = do nothing (all booleans False)
    
    # Note: Custom _perform_shoot and _perform_pass methods removed
    # The Player.apply_actions() method now handles all shooting and passing
    # This follows proper separation of concerns:
    # - AI provides high-level actions (shoot=True, pass=True)  
    # - Player system handles all physics, targeting, cooldowns, etc.