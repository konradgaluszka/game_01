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
        
        ball_pos = ball.ball_body.position
        ball_vel = ball.ball_body.velocity
        
        # Split players into teams
        team_2_players = all_players[5:]  # Last 5 players are team_2
        team_1_players = all_players[:5]  # First 5 players are team_1
        
        # === BALL INFORMATION ===
        # Ball position relative to each team_2 player
        for player in team_2_players:
            rel_x = (ball_pos.x - player.player_body.position.x) / width
            rel_y = (ball_pos.y - player.player_body.position.y) / height
            obs.extend([rel_x, rel_y])
        
        # Ball velocity (normalized)
        obs.extend([ball_vel.x / 1000, ball_vel.y / 1000])
        
        # === TEAM_2 PLAYER INFORMATION ===
        for player in team_2_players:
            pos = player.player_body.position
            vel = player.player_body.velocity
            
            # Absolute position (normalized)
            obs.extend([pos.x / width, pos.y / height])
            
            # Velocity (normalized)
            obs.extend([vel.x / 1000, vel.y / 1000])
            
            # Distance to ball (normalized)
            dist_to_ball = (ball_pos - pos).length / width
            obs.append(dist_to_ball)
        
        # === TEAM_1 PLAYER INFORMATION ===
        for player in team_1_players:
            pos = player.player_body.position
            
            # Position relative to ball
            rel_x = (pos.x - ball_pos.x) / width
            rel_y = (pos.y - ball_pos.y) / height
            obs.extend([rel_x, rel_y])
            
            # Distance to nearest team_2 player
            min_dist = float('inf')
            for t2_player in team_2_players:
                dist = (pos - t2_player.player_body.position).length
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
            elif action == 5:  # Shoot toward goal
                self._perform_shoot(player, ball, all_players)
            elif action == 6:  # Pass to teammate
                self._perform_pass(player, ball, team_2_players, i)
            # Note: action 0 = do nothing
    
    def _perform_shoot(self, player, ball, all_players):
        """Perform shooting action for AI player"""
        ball_pos = ball.ball_body.position
        player_pos = player.player_body.position
        ball_distance = (ball_pos - player_pos).length
        
        # Check if player has ball control
        if ball_distance < player.DRIBBLE_DISTANCE:
            import time
            now = time.time()
            
            # Check cooldown to prevent spam shooting
            if now - player.player_last_shot_time > player.DRIBBLE_COOLDOWN:
                # Calculate direction toward left goal (team_2's target)
                import pymunk
                goal_pos = pymunk.Vec2d(50, 300)  # Left goal center
                shoot_direction = (goal_pos - ball_pos).normalized()
                
                # Add some aiming toward goal opening
                goal_opening_y = 300  # Center of goal
                if abs(ball_pos.y - goal_opening_y) > 50:  # If not aligned with goal
                    # Adjust direction slightly toward goal center
                    goal_center_dir = (pymunk.Vec2d(50, goal_opening_y) - ball_pos).normalized()
                    shoot_direction = (shoot_direction + goal_center_dir * 0.3).normalized()
                
                # Remove ball control springs
                player.remove_ball_springs()
                
                # Apply shooting impulse to ball
                shot_power = player.SHOT_STRENGTH * 1.5  # Stronger shots for goals
                ball.ball_body.apply_impulse_at_local_point(
                    (shoot_direction.x * shot_power, shoot_direction.y * shot_power)
                )
                
                # Update cooldown
                player.player_last_shot_time = time.time()
    
    def _perform_pass(self, player, ball, team_2_players, player_index):
        """Perform passing action for AI player"""
        ball_pos = ball.ball_body.position
        player_pos = player.player_body.position
        ball_distance = (ball_pos - player_pos).length
        
        # Check if player has ball control
        if ball_distance < player.DRIBBLE_DISTANCE:
            import time
            now = time.time()
            
            # Check cooldown
            if now - player.player_last_shot_time > player.DRIBBLE_COOLDOWN:
                # Find best teammate to pass to
                best_teammate = None
                best_score = -1
                
                for j, teammate in enumerate(team_2_players):
                    if j == player_index:  # Skip self
                        continue
                    
                    teammate_pos = teammate.player_body.position
                    pass_distance = (teammate_pos - ball_pos).length
                    
                    # Only consider reasonable pass distances
                    if pass_distance > 200 or pass_distance < 50:
                        continue
                    
                    # Calculate pass score (prefer forward teammates closer to goal)
                    goal_distance = abs(teammate_pos.x - 50)  # Distance to left goal
                    goal_progress = (800 - goal_distance) / 800  # 0 to 1, higher = closer to goal
                    
                    pass_score = goal_progress - (pass_distance / 300)  # Prefer closer, forward teammates
                    
                    if pass_score > best_score:
                        best_score = pass_score
                        best_teammate = teammate
                
                # Execute pass if good teammate found
                if best_teammate is not None:
                    pass_direction = (best_teammate.player_body.position - ball_pos).normalized()
                    
                    # Remove ball control springs
                    player.remove_ball_springs()
                    
                    # Apply pass impulse (lighter than shooting)
                    pass_power = player.SHOT_STRENGTH * 0.7
                    ball.ball_body.apply_impulse_at_local_point(
                        (pass_direction.x * pass_power, pass_direction.y * pass_power)
                    )
                    
                    # Update cooldown
                    player.player_last_shot_time = time.time()