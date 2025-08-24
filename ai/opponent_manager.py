"""
Opponent AI Management System

This module manages different types of opponent AI for training the soccer AI.
It handles self-play opponents, rule-based opponents, and phase-based difficulty
scaling to provide appropriate challenge levels during curriculum learning.

**Opponent Types**:
1. **Self-Play**: Uses trained models as opponents for advanced training
2. **OpponentAI**: Sophisticated rule-based AI with role-based behaviors  
3. **Phase-Based**: Simple rule-based opponents that scale with curriculum phase
"""

import time
from typing import List, Optional
import pymunk


class OpponentManager:
    """
    Manages different opponent AI systems for training.
    
    **Purpose**: Provide appropriate opposition during training phases
    
    **Features**:
    - Self-play model management and fallbacks
    - Phase-appropriate opponent difficulty
    - Consistent opponent interfaces
    - Graceful error handling and fallbacks
    """
    
    def __init__(self, field_width: int = 800, field_height: int = 600):
        self.field_width = field_width
        self.field_height = field_height
        
        # Initialize available opponent systems
        self.opponent_ai = self._init_opponent_ai()
        self.self_play_model = None
        
    def _init_opponent_ai(self):
        """Initialize the sophisticated OpponentAI system if available"""
        try:
            from ai.opponent_ai import OpponentAI
            return OpponentAI(team_side="left", field_width=self.field_width, field_height=self.field_height)
        except ImportError:
            return None
    
    def set_self_play_model(self, model):
        """Set the model to use for self-play opponents"""
        self.self_play_model = model
    
    def control_opponent_team(self, team_1_players: List, ball, team_2_players: List, 
                             opponent_type: str, current_phase: str, observation_builder=None):
        """
        Control team_1 players using the specified opponent type.
        
        Args:
            team_1_players: List of team_1 player objects
            ball: Ball object
            team_2_players: List of team_2 player objects for context
            opponent_type: "self_play", "opponent_ai", or "phase_based"
            current_phase: Current curriculum phase for phase-based opponents
            observation_builder: ObservationBuilder for self-play observations
        """
        try:
            if opponent_type == "self_play" and self.self_play_model and observation_builder:
                self._control_self_play_opponent(team_1_players, ball, team_2_players, observation_builder)
            elif opponent_type == "opponent_ai" and self.opponent_ai:
                self._control_opponent_ai(team_1_players, ball, team_2_players)
            else:
                self._control_phase_based_opponent(team_1_players, ball, team_2_players, current_phase)
                
        except Exception as e:
            print(f"Error in opponent control ({opponent_type}): {e}")
            # Fallback to phase-based opponent
            self._control_phase_based_opponent(team_1_players, ball, team_2_players, current_phase)
    
    def _control_self_play_opponent(self, team_1_players: List, ball, team_2_players: List, observation_builder):
        """Control team_1 using self-play AI model"""
        if not self.self_play_model:
            return
        
        try:
            # Get observation from team_1's perspective
            # Create dummy match and step info for observation
            dummy_match = type('Match', (), {
                'goal1_score': 0, 'goal2_score': 0
            })()
            
            opponent_obs = observation_builder.build_opponent_observation(
                ball, team_1_players, team_2_players, dummy_match, 0, 500
            )
            
            # Get actions from opponent model
            opponent_actions, _ = self.self_play_model.predict(opponent_obs, deterministic=True)
            
            # Apply actions to team_1 players
            self._apply_actions_to_team(team_1_players, opponent_actions)
            
        except Exception as e:
            print(f"Error in self-play opponent: {e}")
            # Fallback to simple opponent
            self._control_simple_opponent(team_1_players, ball)
    
    def _control_opponent_ai(self, team_1_players: List, ball, team_2_players: List):
        """Control team_1 using sophisticated OpponentAI system"""
        self.opponent_ai.control_team(team_1_players, ball, team_2_players)
    
    def _control_phase_based_opponent(self, team_1_players: List, ball, team_2_players: List, phase: str):
        """Control team_1 using phase-appropriate difficulty"""
        if phase == "ball_awareness":
            # Phase 1: Stationary opponent
            return
        elif phase == "basic_soccer":
            # Phase 2: Weak opponent
            self._control_weak_opponent(team_1_players, ball)
        else:
            # Phase 3: Competitive opponent
            self._control_simple_opponent(team_1_players, ball)
    
    def _control_weak_opponent(self, team_1_players: List, ball):
        """Weak opponent that only contests the ball"""
        ball_pos = ball.ball_body.position
        
        # All players move toward ball with reduced force
        for player in team_1_players:
            to_ball = ball_pos - player.player_body.position
            if to_ball.length > 15:  # Don't cluster
                direction = to_ball.normalized()
                # Use weak force (30% of normal)
                force_magnitude = player.force * 0.3
                player.player_body.apply_force_at_local_point(
                    (direction.x * force_magnitude, direction.y * force_magnitude)
                )
    
    def _control_simple_opponent(self, team_1_players: List, ball):
        """Simple but competitive opponent AI"""
        ball_pos = ball.ball_body.position
        ball_vel = ball.ball_body.velocity
        
        # Determine ball control
        closest_player = None
        closest_dist = float('inf')
        
        for player in team_1_players:
            dist = (ball_pos - player.player_body.position).length
            if dist < closest_dist:
                closest_dist = dist
                closest_player = player
        
        # Simple strategy based on ball control
        if closest_player and closest_dist < closest_player.DRIBBLE_DISTANCE:
            self._opponent_attack_mode(closest_player, ball_pos, team_1_players)
        else:
            self._opponent_contest_mode(ball_pos, team_1_players)
    
    def _opponent_attack_mode(self, ball_carrier, ball_pos, team_1_players: List):
        """Simple attack behavior when opponent has ball"""
        import time
        
        carrier_pos = ball_carrier.player_body.position
        goal_pos = pymunk.Vec2d(self.field_width - 50, 300)  # Right goal
        goal_distance = (goal_pos - ball_pos).length
        
        # Try to shoot if close enough
        if goal_distance < 200:
            now = time.time()
            if now - ball_carrier.player_last_shot_time > ball_carrier.DRIBBLE_COOLDOWN:
                shoot_direction = (goal_pos - ball_pos).normalized()
                
                ball_carrier.remove_ball_springs()
                shot_power = ball_carrier.SHOT_STRENGTH * 1.2
                ball_carrier.ball.ball_body.apply_impulse_at_local_point(
                    (shoot_direction.x * shot_power, shoot_direction.y * shot_power)
                )
                ball_carrier.player_last_shot_time = time.time()
                return
        
        # Otherwise dribble toward goal
        to_goal = (goal_pos - carrier_pos).normalized()
        force_magnitude = ball_carrier.force * 0.8
        ball_carrier.player_body.apply_force_at_local_point(
            (to_goal.x * force_magnitude, to_goal.y * force_magnitude)
        )
        
        # Other players support attack
        for i, player in enumerate(team_1_players):
            if player != ball_carrier:
                # Move into attacking positions
                target_x = self.field_width * (0.4 + 0.15 * (i / len(team_1_players)))
                target_y = self.field_height * (0.2 + 0.15 * i)
                
                to_target = pymunk.Vec2d(target_x, target_y) - player.player_body.position
                if to_target.length > 30:
                    direction = to_target.normalized()
                    force_magnitude = player.force * 0.6
                    player.player_body.apply_force_at_local_point(
                        (direction.x * force_magnitude, direction.y * force_magnitude)
                    )
    
    def _opponent_contest_mode(self, ball_pos, team_1_players: List):
        """Contest loose ball"""
        for i, player in enumerate(team_1_players):
            to_ball = ball_pos - player.player_body.position
            if to_ball.length > 10:
                direction = to_ball.normalized()
                # Add spread to avoid clustering
                spread_factor = 0.1 * (i - 2)
                perpendicular = pymunk.Vec2d(-direction.y, direction.x) * spread_factor
                adjusted_direction = (direction + perpendicular).normalized()
                
                force_magnitude = player.force * 0.8
                player.player_body.apply_force_at_local_point(
                    (adjusted_direction.x * force_magnitude, adjusted_direction.y * force_magnitude)
                )
    
    def _apply_actions_to_team(self, players: List, actions):
        """Apply actions to team_1 players (for self-play)"""
        for i, action in enumerate(actions):
            if i >= len(players):
                break
            
            player = players[i]
            teammates_positions = [p.position() for j, p in enumerate(players) if j != i]
            
            # Map actions to movement booleans
            move_up = action == 1
            move_down = action == 2
            move_left = action == 3
            move_right = action == 4
            shoot = action == 5
            pass_ball = action == 6
            
            # Use Player's action system
            player.apply_actions(
                move_up=move_up,
                move_down=move_down,
                move_left=move_left,
                move_right=move_right,
                shoot=shoot,
                pass_ball=pass_ball,
                teammates_positions=teammates_positions
            )
    
    def update_self_play_model(self, new_model_path: str):
        """Update the self-play model with a new trained model"""
        try:
            from stable_baselines3 import PPO
            self.self_play_model = PPO.load(new_model_path)
            print(f"Updated self-play opponent model: {new_model_path}")
        except Exception as e:
            print(f"Failed to update self-play model: {e}")
    
    def get_opponent_info(self, opponent_type: str) -> dict:
        """Get information about the current opponent configuration"""
        info = {
            "opponent_type": opponent_type,
            "available_systems": {
                "self_play": self.self_play_model is not None,
                "opponent_ai": self.opponent_ai is not None,
                "phase_based": True  # Always available
            }
        }
        
        if opponent_type == "opponent_ai" and self.opponent_ai:
            info["formation_info"] = self.opponent_ai.get_team_formation_info()
        
        return info