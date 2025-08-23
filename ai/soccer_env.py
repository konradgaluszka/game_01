"""
Soccer Environment for Reinforcement Learning

This file implements a Gymnasium-compatible environment that wraps the soccer game
for AI training. It provides:

1. **Action Space**: Multi-discrete actions for 5 players (move up/down/left/right/nothing)
2. **Observation Space**: 62-dimensional state vector with relative positions and context  
3. **Reward Function**: Comprehensive reward system encouraging soccer-like behavior
4. **Episode Management**: Handles resets, termination conditions, and game state
5. **Opponent AI**: Simple rule-based team_1 to provide realistic opposition

The environment is designed to train team_2 (blue, right side) to play soccer
against a simple opponent while learning proper positioning, ball control, and scoring.
"""

import gymnasium as gym
import numpy as np
import pygame
import pymunk
import sys
import os
from gymnasium import spaces
from typing import Dict, Tuple, Any

# Add parent directory to path so we can import game modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.Vector import Vector
from player.ball import Ball
from player.team import Team, TeamAreaDimensions
from stadium.pitch import Pitch
from game.match import Match


class SoccerEnv(gym.Env):
    """
    Gymnasium environment for training soccer AI using reinforcement learning.
    
    **Purpose**: Train team_2 (blue, right side) to play soccer effectively
    
    **Key Features**:
    - Multi-agent control: AI controls all 5 players of team_2 simultaneously
    - Rich observation space: Relative positions, velocities, and game context
    - Shaped rewards: Encourages ball control, positioning, goals, and teamwork
    - Opponent AI: Team_1 provides basic opposition for realistic training
    - Episode management: Handles resets when goals scored or time limit reached
    
    **Training Approach**:
    - PPO (Proximal Policy Optimization) algorithm recommended
    - Episodes are 500 steps (8.3 seconds at 60 FPS) for quick feedback
    - Rewards range from -500 to +1000 with most actions giving small positive/negative feedback
    - Ball starting position randomized each episode for variety
    """
    
    def __init__(self, render_mode=None, curriculum=False, phase=None, total_timesteps=0, self_play=False, opponent_model_path=None):
        """
        Initialize the soccer training environment.
        
        Args:
            render_mode: Optional rendering mode ("human" for visual display, None for headless training)
            curriculum: Enable curriculum learning with automatic phase progression
            phase: Manual phase selection ("ball_awareness", "basic_soccer", "competitive_soccer")
            total_timesteps: Current total timesteps for automatic phase detection
            self_play: Enable self-play training (team_1 controlled by AI)
            opponent_model_path: Path to opponent model for self-play
        
        Sets up:
        - Action and observation spaces for the RL algorithm
        - Game physics and objects (field, ball, teams)
        - Episode tracking variables
        - Curriculum learning parameters
        - Self-play opponent AI
        """
        super().__init__()
        
        self.WIDTH, self.HEIGHT = 800, 600
        self.render_mode = render_mode
        
        # Curriculum learning settings
        self.curriculum = curriculum
        self.manual_phase = phase
        self.total_timesteps = total_timesteps
        self.current_phase = self._get_current_phase()
        
        # Self-play settings
        self.self_play = self_play
        self.opponent_model_path = opponent_model_path
        self.opponent_model = None
        self._load_opponent_model()
        
        # OpponentAI settings
        try:
            from ai.opponent_ai import OpponentAI
            self.opponent_ai = OpponentAI(team_side="left", field_width=self.WIDTH, field_height=self.HEIGHT)
            print("OpponentAI loaded for team_1")
        except ImportError:
            self.opponent_ai = None
            print("OpponentAI not available, using fallback opponent")
        
        # === ACTION SPACE DEFINITION ===
        # Each of 5 team_2 players can perform one of 7 actions per timestep:
        # 0 = do nothing, 1 = move up, 2 = move down, 3 = move left, 4 = move right
        # 5 = shoot (toward goal), 6 = pass (toward nearest teammate)
        self.action_space = spaces.MultiDiscrete([7] * 5)  # 5 players, 7 actions each
        
        # === OBSERVATION SPACE DEFINITION ===
        # Carefully designed 62-dimensional observation vector providing relative context:
        # 
        # Ball information (12 dims):
        #   - Ball position relative to each team_2 player (2*5 = 10)
        #   - Ball velocity (2)
        #
        # Team_2 players (25 dims):
        #   - Absolute positions (2*5 = 10) 
        #   - Velocities (2*5 = 10)
        #   - Distance to ball for each player (5)
        #
        # Team_1 players (15 dims):
        #   - Positions relative to ball (2*5 = 10)
        #   - Distance to nearest team_2 player (5)
        #
        # Field context (6 dims):
        #   - Ball distances to left/right goals (2)
        #   - Ball distances to field boundaries (4)
        #
        # Match state (4 dims):
        #   - Team_2 goals scored, Team_1 goals scored, any goal this episode, time progress
        obs_dim = 12 + 25 + 15 + 6 + 4  # 62 total
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        
        # Initialize pygame and physics
        self._init_game()
        
        # Tracking
        self.steps = 0
        self.max_steps = self._get_episode_length()  # Phase-dependent episode length
        self.last_ball_touch_team = None
        self.initial_ball_pos = pymunk.Vec2d(400, 300)  # Center field
        self.episode_start_scores = (0, 0)
        self.boundary_penalty_count = 0  # Track boundary violations for progressive penalties
        
    def _init_game(self):
        """Initialize the game environment"""
        # Always initialize pygame for font support
        pygame.init()
        
        if self.render_mode == "human":
            self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
            pygame.display.set_caption("Soccer AI Training")
            self.clock = pygame.time.Clock()
        else:
            # For training mode, create a dummy surface
            self.screen = None
            self.clock = None
        
        # Physics space
        self.space = pymunk.Space()
        self.space.gravity = (0, 0)
        self.space.damping = 0.6
        
        # Game objects
        self.pitch = Pitch(self.space)
        self.ball = Ball(400, 300, self.space, pygame.Color("yellow"))
        
        # Teams
        self.team_1 = Team(
            side="left", 
            team_area_dimensions=TeamAreaDimensions(
                top_left=Vector(0, 0), 
                bottom_right=Vector(self.WIDTH/2, self.HEIGHT)
            ), 
            space=self.space, 
            color="red", 
            ball=self.ball
        )
        
        self.team_2 = Team(
            side="right", 
            team_area_dimensions=TeamAreaDimensions(
                top_left=Vector(self.WIDTH/2, 0), 
                bottom_right=Vector(self.WIDTH, self.HEIGHT)
            ), 
            space=self.space, 
            color="blue", 
            ball=self.ball
        )
        
        self.all_players = self.team_1.players() + self.team_2.players()
        
        # Match
        self.match = Match(
            self.pitch.goal_left.is_ball_inside_goal,
            self.pitch.goal_right.is_ball_inside_goal,
            self.ball,
            resettable_objects=[self.ball, self.team_1, self.team_2]
        )
    
    def _load_opponent_model(self):
        """Load opponent model for self-play"""
        if self.self_play and self.opponent_model_path:
            try:
                from stable_baselines3 import PPO
                self.opponent_model = PPO.load(self.opponent_model_path)
                print(f"Loaded opponent model from {self.opponent_model_path}")
            except Exception as e:
                print(f"Failed to load opponent model: {e}")
                self.opponent_model = None
    
    def update_opponent_model(self, new_model_path):
        """Update the opponent model during training"""
        self.opponent_model_path = new_model_path
        self._load_opponent_model()
    
    def _get_opponent_observation(self) -> np.ndarray:
        """
        Get observation for team_1 (opponent) in self-play.
        This mirrors team_2's observation but from team_1's perspective.
        """
        obs = []
        
        ball_pos = self.ball.ball_body.position
        ball_vel = self.ball.ball_body.velocity
        
        # Split players - NOTE: team_1 is first 5, team_2 is last 5
        team_1_players = self.team_1.players()  # Opponent (we're controlling)
        team_2_players = self.team_2.players()  # Training agent
        
        # === BALL INFORMATION (from team_1 perspective) ===
        # Ball position relative to each team_1 player
        for player in team_1_players:
            rel_x = (ball_pos.x - player.player_body.position.x) / self.WIDTH
            rel_y = (ball_pos.y - player.player_body.position.y) / self.HEIGHT
            obs.extend([rel_x, rel_y])
        
        # Ball velocity (normalized)
        obs.extend([ball_vel.x / 1000, ball_vel.y / 1000])
        
        # === TEAM_1 PLAYER INFORMATION ===
        for player in team_1_players:
            pos = player.player_body.position
            vel = player.player_body.velocity
            
            # Absolute position (normalized)
            obs.extend([pos.x / self.WIDTH, pos.y / self.HEIGHT])
            
            # Velocity (normalized)
            obs.extend([vel.x / 1000, vel.y / 1000])
            
            # Distance to ball (normalized)
            dist_to_ball = (ball_pos - pos).length / self.WIDTH
            obs.append(dist_to_ball)
        
        # === TEAM_2 PLAYER INFORMATION (opponents from team_1's view) ===
        for player in team_2_players:
            pos = player.player_body.position
            
            # Position relative to ball
            rel_x = (pos.x - ball_pos.x) / self.WIDTH
            rel_y = (pos.y - ball_pos.y) / self.HEIGHT
            obs.extend([rel_x, rel_y])
            
            # Distance to nearest team_1 player
            min_dist = float('inf')
            for t1_player in team_1_players:
                dist = (pos - t1_player.player_body.position).length
                min_dist = min(min_dist, dist)
            obs.append(min_dist / self.WIDTH)
        
        # === FIELD CONTEXT (from team_1 perspective) ===
        # Ball distance to goals (flipped for team_1)
        right_goal_dist = (self.WIDTH - ball_pos.x) / self.WIDTH  # Team_1's target goal
        left_goal_dist = ball_pos.x / self.WIDTH  # Team_1's own goal
        obs.extend([left_goal_dist, right_goal_dist])
        
        # Ball distance to field boundaries
        top_dist = ball_pos.y / self.HEIGHT
        bottom_dist = (self.HEIGHT - ball_pos.y) / self.HEIGHT
        left_bound_dist = ball_pos.x / self.WIDTH
        right_bound_dist = (self.WIDTH - ball_pos.x) / self.WIDTH
        obs.extend([top_dist, bottom_dist, left_bound_dist, right_bound_dist])
        
        # === MATCH STATE (from team_1 perspective) ===
        obs.extend([
            float(self.match.goal2_score),  # Team_1 goals
            float(self.match.goal1_score),  # Team_2 goals  
            float(self.match.goal1_score + self.match.goal2_score > 0),  # Goal scored this episode
            float(self.steps / self.max_steps)  # Time progress
        ])
        
        return np.array(obs, dtype=np.float32)
        
    def _get_current_phase(self):
        """Determine current training phase based on timesteps or manual setting"""
        if self.manual_phase:
            return self.manual_phase
        
        if not self.curriculum:
            return "competitive_soccer"  # Default full difficulty
            
        if self.total_timesteps < 25000:
            return "ball_awareness"
        elif self.total_timesteps < 75000:
            return "basic_soccer"
        else:
            return "competitive_soccer"
    
    def _get_episode_length(self):
        """Get episode length based on current phase"""
        if self.current_phase == "ball_awareness":
            return 300  # 5 seconds
        elif self.current_phase == "basic_soccer":
            return 400  # 6.7 seconds  
        else:
            return 500  # 8.3 seconds
    
    def _get_ball_start_position(self):
        """Get ball starting position based on current phase"""
        import random
        
        if self.current_phase == "ball_awareness":
            # Start ball in middle area, not too close but visible to team_2
            # This forces team_2 players to actively move toward it
            offset_x = random.uniform(-80, 80)
            offset_y = random.uniform(-80, 80)
            return (self.WIDTH * 0.5 + offset_x, self.HEIGHT * 0.5 + offset_y)
        else:
            # Random position across field
            offset_x = random.uniform(-100, 100)
            offset_y = random.uniform(-50, 50)
            return (self.initial_ball_pos.x + offset_x, self.initial_ball_pos.y + offset_y)
        
    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment to initial state"""
        super().reset(seed=seed)
        
        # Reset game objects
        self.match.reset()
        self.steps = 0
        self.last_ball_touch_team = None
        self.episode_start_scores = (self.match.goal1_score, self.match.goal2_score)
        self.boundary_penalty_count = 0
        
        # Set ball position based on current training phase
        ball_x, ball_y = self._get_ball_start_position()
        self.ball.ball_body.position = (ball_x, ball_y)
        self.ball.ball_body.velocity = (0, 0)
        
        return self._get_observation(), {}
        
    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one environment step"""
        # Apply actions to team_2 players
        self._apply_actions(action)
        
        # Control team_1 based on mode
        if self.self_play:
            self._self_play_opponent_ai()
        elif self.opponent_ai is not None:
            # Use new OpponentAI for team_1
            self._opponent_ai_control()
        else:
            # Phase-based opponent AI for team_1 (fallback)
            self._phase_based_opponent_ai()
        
        # Update physics
        self.space.step(1/60)  # 60 FPS
        
        # Update game objects
        self.team_1.simulate()
        self.team_2.simulate()
        self.ball.simulate()
        # Create a fake keys dict for the match update
        fake_keys = {}
        for i in range(512):  # Cover all possible pygame keys
            fake_keys[i] = False
        self.match.update(fake_keys)  # No keyboard input during training
        
        # Get new observation
        obs = self._get_observation()
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Check if episode is done
        self.steps += 1
        # Episode ends if a goal is scored or max steps reached
        goal_scored = (self.match.goal1_score > 0 or self.match.goal2_score > 0)
        terminated = goal_scored or self.steps >= self.max_steps
        truncated = False
        
        return obs, reward, terminated, truncated, {}
        
    def _apply_actions(self, actions):
        """Apply actions to team_2 players using Player.apply_actions interface"""
        team_2_players = self.team_2.players()
        
        for i, action in enumerate(actions):
            if i >= len(team_2_players):
                break
                
            player = team_2_players[i]
            
            # Get teammate positions for passing
            teammates_positions = [p.position() for j, p in enumerate(team_2_players) if j != i]
            
            # Map actions to movement + action booleans (same as AI controller)
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
    
    # Note: _perform_shoot method removed - now handled by Player.apply_actions()
    # This follows proper separation of concerns:
    # - Training environment provides high-level actions (shoot=True)
    # - Player system handles all physics, targeting, cooldowns, etc.
    
    # Note: _perform_pass method removed - now handled by Player.apply_actions()
    # This follows proper separation of concerns:
    # - Training environment provides high-level actions (pass=True)
    # - Player system handles all physics, targeting, teammate selection, etc.
    
    def _self_play_opponent_ai(self):
        """Control team_1 using self-play AI model"""
        if not self.opponent_model:
            # Fall back to simple AI if no opponent model loaded
            self._phase_based_opponent_ai()
            return
        
        try:
            # Get observation from team_1's perspective
            opponent_obs = self._get_opponent_observation()
            
            # Get actions from opponent model
            opponent_actions, _ = self.opponent_model.predict(opponent_obs, deterministic=True)
            
            # Apply actions to team_1 players (similar to _apply_actions but for team_1)
            self._apply_opponent_actions(opponent_actions)
            
        except Exception as e:
            print(f"Error in self-play opponent AI: {e}")
            # Fall back to rule-based AI
            self._phase_based_opponent_ai()
    
    def _apply_opponent_actions(self, actions):
        """Apply actions to team_1 players (opponent in self-play) using Player.apply_actions interface"""
        team_1_players = self.team_1.players()
        
        for i, action in enumerate(actions):
            if i >= len(team_1_players):
                break
                
            player = team_1_players[i]
            
            # Get teammate positions for passing
            teammates_positions = [p.position() for j, p in enumerate(team_1_players) if j != i]
            
            # Map actions to movement + action booleans (same as team_2)
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
    
    def _perform_opponent_shoot(self, player, ball_pos, player_pos):
        """Perform shooting action for opponent AI player (team_1)"""
        ball_distance = (ball_pos - player_pos).length
        
        # Check if player has ball control
        if ball_distance < player.DRIBBLE_DISTANCE:
            import time
            now = time.time()
            
            # Check cooldown to prevent spam shooting
            if now - player.player_last_shot_time > player.DRIBBLE_COOLDOWN:
                # Calculate direction toward right goal (team_1's target)
                import pymunk
                goal_pos = pymunk.Vec2d(self.WIDTH - 50, 300)  # Right goal center
                shoot_direction = (goal_pos - ball_pos).normalized()
                
                # Add some aiming toward goal opening
                goal_opening_y = 300  # Center of goal
                if abs(ball_pos.y - goal_opening_y) > 50:  # If not aligned with goal
                    # Adjust direction slightly toward goal center
                    goal_center_dir = (pymunk.Vec2d(self.WIDTH - 50, goal_opening_y) - ball_pos).normalized()
                    shoot_direction = (shoot_direction + goal_center_dir * 0.3).normalized()
                
                # Remove ball control springs
                player.remove_ball_springs()
                
                # Apply shooting impulse to ball
                shot_power = player.SHOT_STRENGTH * 1.5  # Stronger shots for goals
                self.ball.ball_body.apply_impulse_at_local_point(
                    (shoot_direction.x * shot_power, shoot_direction.y * shot_power)
                )
                
                # Update cooldown
                player.player_last_shot_time = time.time()
    
    def _perform_opponent_pass(self, player, ball_pos, player_pos, team_1_players, player_index):
        """Perform passing action for opponent AI player (team_1)"""
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
                
                for j, teammate in enumerate(team_1_players):
                    if j == player_index:  # Skip self
                        continue
                    
                    teammate_pos = teammate.player_body.position
                    pass_distance = (teammate_pos - ball_pos).length
                    
                    # Only consider reasonable pass distances
                    if pass_distance > 200 or pass_distance < 50:
                        continue
                    
                    # Calculate pass score (prefer forward teammates closer to right goal)
                    goal_distance = abs(teammate_pos.x - (self.WIDTH - 50))  # Distance to right goal
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
                    self.ball.ball_body.apply_impulse_at_local_point(
                        (pass_direction.x * pass_power, pass_direction.y * pass_power)
                    )
                    
                    # Update cooldown
                    player.player_last_shot_time = time.time()
    
    def _phase_based_opponent_ai(self):
        """Phase-based opponent AI with different difficulty levels"""
        if self.current_phase == "ball_awareness":
            # Phase 1: Stationary opponent - no movement
            return
        elif self.current_phase == "basic_soccer":
            # Phase 2: Weak opponent - only contests ball, no shooting/passing
            self._weak_opponent_ai()
        else:
            # Phase 3: Full competitive opponent
            self._simple_opponent_ai()
    
    def _weak_opponent_ai(self):
        """Weak opponent AI that only contests the ball without shooting/passing"""
        ball_pos = self.ball.ball_body.position
        team_1_players = self.team_1.players()
        
        # All players just move toward ball with reduced force
        for player in team_1_players:
            to_ball = ball_pos - player.player_body.position
            if to_ball.length > 15:  # Don't cluster too much
                direction = to_ball.normalized()
                # Weak force (0.3x normal)
                force_magnitude = player.force * 0.3
                player.player_body.apply_force_at_local_point(
                    (direction.x * force_magnitude, direction.y * force_magnitude)
                )
    
    def _simple_opponent_ai(self):
        """
        Intelligent opponent AI for team_1 to provide realistic competition during training.
        
        **Major Improvements Over Previous Version**:
        1. **Ball Control**: Actively tries to dribble and control ball
        2. **Shooting**: Attempts to score goals when in good position
        3. **Passing**: Makes forward passes to advance ball
        4. **Proper Goal Direction**: Attacks right goal (team_2's goal)
        5. **Stronger Forces**: Uses full player force for competitive play
        6. **Adaptive Strategy**: Switches between attack/defense based on ball position
        7. **Player Pressure**: Actively contests team_2 players near ball
        
        **Strategy**:
        - When team_1 has ball control: Attack toward right goal with passing/shooting
        - When team_2 has ball: Pressure ball carrier and defend left goal
        - Maintain formation with attackers and defenders
        """
        ball_pos = self.ball.ball_body.position
        ball_vel = self.ball.ball_body.velocity
        team_1_players = self.team_1.players()
        team_2_players = self.team_2.players()
        
        # Determine who has ball control
        team_1_ball_control = False
        team_2_ball_control = False
        closest_t1_dist = float('inf')
        closest_t2_dist = float('inf')
        closest_t1_player = None
        closest_t2_player = None
        
        # Find closest players from each team
        for player in team_1_players:
            dist = (ball_pos - player.player_body.position).length
            if dist < closest_t1_dist:
                closest_t1_dist = dist
                closest_t1_player = player
        
        for player in team_2_players:
            dist = (ball_pos - player.player_body.position).length
            if dist < closest_t2_dist:
                closest_t2_dist = dist
                closest_t2_player = player
        
        # Determine ball control (within dribble distance)
        if closest_t1_player and closest_t1_dist < closest_t1_player.DRIBBLE_DISTANCE:
            team_1_ball_control = True
        elif closest_t2_player and closest_t2_dist < closest_t2_player.DRIBBLE_DISTANCE:
            team_2_ball_control = True
        
        # === TEAM_1 HAS BALL CONTROL - ATTACK MODE ===
        if team_1_ball_control and closest_t1_player:
            self._opponent_attack_mode(closest_t1_player, ball_pos, team_1_players)
        
        # === TEAM_2 HAS BALL CONTROL - DEFENSE MODE ===
        elif team_2_ball_control:
            self._opponent_defense_mode(closest_t2_player, ball_pos, team_1_players, team_2_players)
        
        # === LOOSE BALL - CONTEST MODE ===
        else:
            self._opponent_contest_mode(ball_pos, team_1_players)
    
    def _opponent_attack_mode(self, ball_carrier, ball_pos, team_1_players):
        """Team_1 attacks toward right goal when they have ball control"""
        import time
        
        # Ball carrier decides: shoot or pass
        carrier_pos = ball_carrier.player_body.position
        goal_pos = pymunk.Vec2d(self.WIDTH - 50, 300)  # Right goal (team_2's goal)
        goal_distance = (goal_pos - ball_pos).length
        
        # Try to shoot if close to goal
        if goal_distance < 200:  # Within shooting range
            now = time.time()
            if now - ball_carrier.player_last_shot_time > ball_carrier.DRIBBLE_COOLDOWN:
                # Shoot toward right goal
                shoot_direction = (goal_pos - ball_pos).normalized()
                
                # Add goal center aiming
                goal_center_y = 300
                if abs(ball_pos.y - goal_center_y) > 50:
                    goal_center_dir = (pymunk.Vec2d(self.WIDTH - 50, goal_center_y) - ball_pos).normalized()
                    shoot_direction = (shoot_direction + goal_center_dir * 0.3).normalized()
                
                # Remove ball control and shoot
                ball_carrier.remove_ball_springs()
                shot_power = ball_carrier.SHOT_STRENGTH * 1.2
                self.ball.ball_body.apply_impulse_at_local_point(
                    (shoot_direction.x * shot_power, shoot_direction.y * shot_power)
                )
                ball_carrier.player_last_shot_time = time.time()
                return
        
        # Look for forward pass opportunity
        best_teammate = None
        best_score = -1
        
        for teammate in team_1_players:
            if teammate == ball_carrier:
                continue
            
            teammate_pos = teammate.player_body.position
            pass_distance = (teammate_pos - ball_pos).length
            
            # Only consider reasonable pass distances
            if 50 < pass_distance < 250:
                # Prefer teammates closer to right goal
                teammate_goal_dist = (goal_pos - teammate_pos).length
                current_goal_dist = (goal_pos - ball_pos).length
                
                # Score based on: goal progress + reasonable distance
                goal_progress = max(0, current_goal_dist - teammate_goal_dist)
                pass_score = goal_progress - (pass_distance / 200)
                
                if pass_score > best_score:
                    best_score = pass_score
                    best_teammate = teammate
        
        # Execute pass if good opportunity
        if best_teammate and best_score > 0:
            now = time.time()
            if now - ball_carrier.player_last_shot_time > ball_carrier.DRIBBLE_COOLDOWN:
                pass_direction = (best_teammate.player_body.position - ball_pos).normalized()
                ball_carrier.remove_ball_springs()
                pass_power = ball_carrier.SHOT_STRENGTH * 0.8
                self.ball.ball_body.apply_impulse_at_local_point(
                    (pass_direction.x * pass_power, pass_direction.y * pass_power)
                )
                ball_carrier.player_last_shot_time = time.time()
                return
        
        # Default: dribble toward goal
        to_goal = (goal_pos - carrier_pos).normalized()
        force_magnitude = ball_carrier.force * 0.8  # Strong but not full force
        ball_carrier.player_body.apply_force_at_local_point(
            (to_goal.x * force_magnitude, to_goal.y * force_magnitude)
        )
        
        # Other players move into attacking positions
        for i, player in enumerate(team_1_players):
            if player != ball_carrier:
                # Spread out in attacking formation toward right side
                target_x = self.WIDTH * (0.4 + 0.15 * (i / len(team_1_players)))
                target_y = self.HEIGHT * (0.2 + 0.15 * i)
                
                to_target = pymunk.Vec2d(target_x, target_y) - player.player_body.position
                if to_target.length > 30:
                    direction = to_target.normalized()
                    force_magnitude = player.force * 0.6
                    player.player_body.apply_force_at_local_point(
                        (direction.x * force_magnitude, direction.y * force_magnitude)
                    )
    
    def _opponent_defense_mode(self, team_2_ball_carrier, ball_pos, team_1_players, team_2_players):
        """Team_1 defends when team_2 has ball control"""
        # Pressure the ball carrier
        carrier_pos = team_2_ball_carrier.player_body.position
        
        # Find closest team_1 player to pressure ball carrier
        closest_defender = None
        closest_dist = float('inf')
        for player in team_1_players:
            dist = (carrier_pos - player.player_body.position).length
            if dist < closest_dist:
                closest_dist = dist
                closest_defender = player
        
        # Primary defender pressures ball carrier aggressively
        if closest_defender:
            to_carrier = (carrier_pos - closest_defender.player_body.position).normalized()
            force_magnitude = closest_defender.force * 0.9  # Very aggressive
            closest_defender.player_body.apply_force_at_local_point(
                (to_carrier.x * force_magnitude, to_carrier.y * force_magnitude)
            )
        
        # Other players take defensive positions between ball and left goal
        left_goal_pos = pymunk.Vec2d(50, 300)
        for i, player in enumerate(team_1_players):
            if player != closest_defender:
                # Position between ball and goal
                ball_to_goal = (left_goal_pos - ball_pos).normalized()
                base_defensive_pos = ball_pos + ball_to_goal * 80  # 80 pixels toward goal
                
                # Spread defenders vertically
                vertical_spread = (i - 2) * 40  # Spread around center
                defensive_pos = pymunk.Vec2d(base_defensive_pos.x, base_defensive_pos.y + vertical_spread)
                
                to_defensive_pos = defensive_pos - player.player_body.position
                if to_defensive_pos.length > 25:
                    direction = to_defensive_pos.normalized()
                    force_magnitude = player.force * 0.7
                    player.player_body.apply_force_at_local_point(
                        (direction.x * force_magnitude, direction.y * force_magnitude)
                    )
    
    def _opponent_contest_mode(self, ball_pos, team_1_players):
        """Contest loose ball aggressively"""
        # All players converge on ball with high intensity
        for i, player in enumerate(team_1_players):
            to_ball = ball_pos - player.player_body.position
            if to_ball.length > 10:  # Don't clump too much
                direction = to_ball.normalized()
                # Add some spread to avoid all players on same spot
                spread_factor = 0.1 * (i - 2)  # -0.2 to +0.2
                perpendicular = pymunk.Vec2d(-direction.y, direction.x) * spread_factor
                adjusted_direction = (direction + perpendicular).normalized()
                
                force_magnitude = player.force * 0.8  # Strong contest
                player.player_body.apply_force_at_local_point(
                    (adjusted_direction.x * force_magnitude, adjusted_direction.y * force_magnitude)
                )
                    
    def _get_observation(self) -> np.ndarray:
        """
        Generate the 62-dimensional observation vector for the RL agent.
        
        **Design Philosophy**: Provide relative context rather than absolute positions
        to help the AI understand spatial relationships and make position-invariant decisions.
        
        **Key Features**:
        - All positions normalized to [0,1] range for stability
        - Relative positions help with generalization
        - Ball-centric information for better decision making
        - Opponent awareness for competitive play
        
        Returns:
            np.ndarray: 62-dimensional observation vector
        """
        obs = []
        
        ball_pos = self.ball.ball_body.position
        ball_vel = self.ball.ball_body.velocity
        team_2_players = self.team_2.players()
        team_1_players = self.team_1.players()
        
        # === BALL INFORMATION ===
        # Ball position relative to each team_2 player
        for player in team_2_players:
            rel_x = (ball_pos.x - player.player_body.position.x) / self.WIDTH
            rel_y = (ball_pos.y - player.player_body.position.y) / self.HEIGHT
            obs.extend([rel_x, rel_y])
        
        # Ball velocity (normalized)
        obs.extend([ball_vel.x / 1000, ball_vel.y / 1000])
        
        # === TEAM_2 PLAYER INFORMATION ===
        for player in team_2_players:
            pos = player.player_body.position
            vel = player.player_body.velocity
            
            # Absolute position (normalized)
            obs.extend([pos.x / self.WIDTH, pos.y / self.HEIGHT])
            
            # Velocity (normalized)
            obs.extend([vel.x / 1000, vel.y / 1000])
            
            # Distance to ball (normalized)
            dist_to_ball = (ball_pos - pos).length / self.WIDTH
            obs.append(dist_to_ball)
        
        # === TEAM_1 PLAYER INFORMATION ===
        for player in team_1_players:
            pos = player.player_body.position
            
            # Position relative to ball
            rel_x = (pos.x - ball_pos.x) / self.WIDTH
            rel_y = (pos.y - ball_pos.y) / self.HEIGHT
            obs.extend([rel_x, rel_y])
            
            # Distance to nearest team_2 player
            min_dist = float('inf')
            for t2_player in team_2_players:
                dist = (pos - t2_player.player_body.position).length
                min_dist = min(min_dist, dist)
            obs.append(min_dist / self.WIDTH)
        
        # === FIELD CONTEXT ===
        # Ball distance to goals
        left_goal_dist = ball_pos.x / self.WIDTH  # Distance to left goal
        right_goal_dist = (self.WIDTH - ball_pos.x) / self.WIDTH  # Distance to right goal
        obs.extend([left_goal_dist, right_goal_dist])
        
        # Ball distance to field boundaries
        top_dist = ball_pos.y / self.HEIGHT
        bottom_dist = (self.HEIGHT - ball_pos.y) / self.HEIGHT
        left_bound_dist = ball_pos.x / self.WIDTH
        right_bound_dist = (self.WIDTH - ball_pos.x) / self.WIDTH
        obs.extend([top_dist, bottom_dist, left_bound_dist, right_bound_dist])
        
        # === MATCH STATE ===
        obs.extend([
            float(self.match.goal1_score),  # Team_2 goals
            float(self.match.goal2_score),  # Team_1 goals
            float(self.match.goal1_score + self.match.goal2_score > 0),  # Goal scored this episode
            float(self.steps / self.max_steps)  # Time progress
        ])
        
        return np.array(obs, dtype=np.float32)
        
    def _calculate_reward(self) -> float:
        """
        Phase-based curriculum reward function that adapts based on training progress.
        
        **Phase 1 - Ball Awareness**: Focus on ball-seeking and boundary avoidance
        **Phase 2 - Basic Soccer**: Add shooting, passing, and positioning rewards  
        **Phase 3 - Competitive Soccer**: Full strategic rewards and advanced tactics
        
        Returns:
            float: Phase-appropriate reward value
        """
        if self.current_phase == "ball_awareness":
            return self._ball_awareness_rewards()
        elif self.current_phase == "basic_soccer":
            return self._basic_soccer_rewards()
        else:
            return self._competitive_soccer_rewards()
    
    def _ball_awareness_rewards(self) -> float:
        """Phase 1: Simple ball-seeking and boundary avoidance rewards"""
        reward = 0.0
        ball_pos = self.ball.ball_body.position
        team_2_players = self.team_2.players()
        
        # === GOAL REWARDS ===
        if self.match.goal1_score > 0:
            reward += 500.0  # Team_2 scored
        if self.match.goal2_score > 0:
            reward -= 250.0  # Team_1 scored
        
        # === BALL PROXIMITY REWARDS FOR ALL PLAYERS ===
        # Give each player individual rewards for getting closer to ball
        total_proximity_reward = 0.0
        ball_control_bonus = 0.0
        min_distance = float('inf')
        
        for player in team_2_players:
            player_pos = player.player_body.position
            distance = (ball_pos - player_pos).length
            min_distance = min(min_distance, distance)
            
            # Individual proximity reward for each player
            max_meaningful_distance = 300.0  # Reasonable chase distance
            if distance < max_meaningful_distance:
                # Exponential reward - gets much higher as player gets closer
                proximity_factor = 1 - (distance / max_meaningful_distance)
                individual_reward = 15.0 * (proximity_factor ** 2)  # Squared for exponential effect
                total_proximity_reward += individual_reward
            
            # Ball control bonus for the player with the ball
            if distance < player.DRIBBLE_DISTANCE:
                ball_control_bonus = 50.0
        
        reward += total_proximity_reward
        reward += ball_control_bonus
        
        # === ADDITIONAL BALL-SEEKING INCENTIVES ===
        # Extra reward if ANY player is making progress toward ball
        for player in team_2_players:
            player_pos = player.player_body.position
            player_vel = player.player_body.velocity
            
            # Vector from player to ball
            to_ball = ball_pos - player_pos
            if to_ball.length > 0:
                to_ball_normalized = to_ball.normalized()
                
                # Check if player velocity is aligned with ball direction
                vel_magnitude = player_vel.length
                if vel_magnitude > 10:  # Player is moving
                    vel_normalized = player_vel.normalized()
                    alignment = to_ball_normalized.dot(vel_normalized)
                    
                    # Reward movement toward ball
                    if alignment > 0.5:  # Moving toward ball
                        movement_reward = 8.0 * alignment  # Up to +8 for perfect alignment
                        reward += movement_reward
        
        # === PENALTY FOR IGNORING BALL ===
        # Strong penalty if no one is close to ball
        if min_distance > 150:  # If everyone is far from ball
            reward -= 15.0  # Strong penalty for ignoring ball
        
        # === MOVEMENT TOWARD BALL BONUS ===
        # Reward any movement that reduces distance to ball
        for player in team_2_players:
            player_pos = player.player_body.position
            player_vel = player.player_body.velocity
            distance_to_ball = (ball_pos - player_pos).length
            
            # Only reward movement if player is not already at ball
            if distance_to_ball > player.DRIBBLE_DISTANCE:
                # Calculate if player is moving toward ball
                to_ball_vector = ball_pos - player_pos
                if to_ball_vector.length > 0 and player_vel.length > 5:
                    # Dot product to check alignment
                    to_ball_normalized = to_ball_vector.normalized()
                    vel_normalized = player_vel.normalized()
                    alignment = to_ball_normalized.dot(vel_normalized)
                    
                    # Significant reward for moving toward ball
                    if alignment > 0.3:  # Moving somewhat toward ball
                        movement_bonus = 5.0 * alignment * (player_vel.length / 100.0)
                        reward += movement_bonus
        
        # === BOUNDARY PENALTIES ===
        boundary_penalty = self._calculate_boundary_penalties(team_2_players)
        reward += boundary_penalty
        
        return reward
    
    def _basic_soccer_rewards(self) -> float:
        """Phase 2: Add shooting, passing, and basic positioning"""
        reward = 0.0
        ball_pos = self.ball.ball_body.position
        ball_vel = self.ball.ball_body.velocity
        team_2_players = self.team_2.players()
        team_1_players = self.team_1.players()
        
        # === GOAL REWARDS ===
        if self.match.goal1_score > 0:
            reward += 750.0  # Team_2 scored
        if self.match.goal2_score > 0:
            reward -= 400.0  # Team_1 scored
        
        # === BALL INTERACTION REWARDS ===
        team_2_has_ball = False
        min_distance = float('inf')
        
        for player in team_2_players:
            distance = (ball_pos - player.player_body.position).length
            min_distance = min(min_distance, distance)
            if distance < player.DRIBBLE_DISTANCE:
                team_2_has_ball = True
                reward += 30.0  # Ball control
                break
        
        # Ball proximity reward
        if min_distance < self.WIDTH:
            proximity_reward = 15.0 * (1 - min_distance / (self.WIDTH * 0.7))
            reward += max(0, proximity_reward)
        
        # === ACTION REWARDS ===
        # Shooting attempt (ball moving fast toward goal)
        if ball_vel.length > 100:
            goal_pos = (50, 300)  # Left goal
            ball_to_goal = ((goal_pos[0] - ball_pos.x)**2 + (goal_pos[1] - ball_pos.y)**2)**0.5
            if ball_to_goal < 200:  # Within shooting range
                ball_dir = (ball_vel.x, ball_vel.y)
                goal_dir = (goal_pos[0] - ball_pos.x, goal_pos[1] - ball_pos.y)
                goal_dir_length = (goal_dir[0]**2 + goal_dir[1]**2)**0.5
                if goal_dir_length > 0:
                    dot_product = (ball_dir[0] * goal_dir[0] + ball_dir[1] * goal_dir[1]) / goal_dir_length
                    if dot_product > 50:  # Ball moving toward goal
                        reward += 15.0  # Shooting attempt
        
        # Passing attempt (ball moving toward teammate)
        if ball_vel.length > 80:
            for player in team_2_players:
                player_pos = player.player_body.position
                ball_to_player_dist = ((player_pos.x - ball_pos.x)**2 + (player_pos.y - ball_pos.y)**2)**0.5
                if 50 < ball_to_player_dist < 150:  # Reasonable pass distance
                    ball_dir = (ball_vel.x, ball_vel.y)
                    player_dir = (player_pos.x - ball_pos.x, player_pos.y - ball_pos.y)
                    player_dir_length = (player_dir[0]**2 + player_dir[1]**2)**0.5
                    if player_dir_length > 0:
                        dot_product = (ball_dir[0] * player_dir[0] + ball_dir[1] * player_dir[1]) / player_dir_length
                        if dot_product > 40:  # Ball moving toward teammate
                            reward += 12.0  # Passing attempt
                            break
        
        # === POSITIONING REWARDS ===
        # Basic formation spread
        team_spread = self._calculate_team_spread(team_2_players)
        if team_spread > 100:  # Not too clustered
            reward += 5.0
        
        # === BOUNDARY PENALTIES ===
        boundary_penalty = self._calculate_boundary_penalties(team_2_players)
        reward += boundary_penalty
        
        return reward
    
    def _competitive_soccer_rewards(self) -> float:
        """Phase 3: Full strategic and tactical rewards"""
        reward = 0.0
        ball_pos = self.ball.ball_body.position
        ball_vel = self.ball.ball_body.velocity
        team_2_players = self.team_2.players()
        team_1_players = self.team_1.players()
        
        # === GOAL REWARDS ===
        if self.match.goal1_score > 0:
            reward += 1000.0  # Team_2 scored
        if self.match.goal2_score > 0:
            reward -= 500.0   # Team_1 scored
        
        # === BALL CONTROL REWARDS ===
        team_2_has_ball = False
        team_1_has_ball = False
        min_distance = float('inf')
        
        for player in team_2_players:
            distance = (ball_pos - player.player_body.position).length
            min_distance = min(min_distance, distance)
            if distance < player.DRIBBLE_DISTANCE:
                team_2_has_ball = True
                reward += 25.0  # Ball control
                break
        
        if not team_2_has_ball:
            for player in team_1_players:
                distance = (ball_pos - player.player_body.position).length
                if distance < player.DRIBBLE_DISTANCE:
                    team_1_has_ball = True
                    break
        
        # Ball proximity
        if min_distance < self.WIDTH:
            proximity_reward = 12.0 * (1 - min_distance / (self.WIDTH * 0.7))
            reward += max(0, proximity_reward)
        
        # === STRATEGIC REWARDS ===
        if team_2_has_ball:
            reward += 2.0  # Possession bonus
        elif team_1_has_ball:
            reward -= 3.0  # Opponent has ball
        
        # Successful passes (high velocity toward teammates in good positions)
        if ball_vel.length > 80:
            for player in team_2_players:
                player_pos = player.player_body.position
                ball_to_player_dist = ((player_pos.x - ball_pos.x)**2 + (player_pos.y - ball_pos.y)**2)**0.5
                if 60 < ball_to_player_dist < 200:
                    # Check if this is a forward pass (toward opponent goal)
                    if player_pos.x < ball_pos.x:  # Forward pass
                        ball_dir = (ball_vel.x, ball_vel.y)
                        player_dir = (player_pos.x - ball_pos.x, player_pos.y - ball_pos.y)
                        player_dir_length = (player_dir[0]**2 + player_dir[1]**2)**0.5
                        if player_dir_length > 0:
                            dot_product = (ball_dir[0] * player_dir[0] + ball_dir[1] * player_dir[1]) / player_dir_length
                            if dot_product > 50:
                                reward += 25.0  # Successful forward pass
                                break
        
        # Shot accuracy (ball moving toward goal center)
        if ball_vel.length > 120:
            goal_center = (50, 300)
            goal_dir = (goal_center[0] - ball_pos.x, goal_center[1] - ball_pos.y)
            goal_distance = (goal_dir[0]**2 + goal_dir[1]**2)**0.5
            if goal_distance < 250:  # Within shooting range
                ball_dir = (ball_vel.x, ball_vel.y)
                if goal_distance > 0:
                    dot_product = (ball_dir[0] * goal_dir[0] + ball_dir[1] * goal_dir[1]) / goal_distance
                    if dot_product > 60:  # Good shot toward goal
                        reward += 35.0  # Shot accuracy
        
        # === FORMATION AND POSITIONING ===
        # Team spread
        team_spread = self._calculate_team_spread(team_2_players)
        if team_spread > 120:
            reward += 8.0  # Good formation quality
        
        # Role-based positioning bonus (simplified)
        for i, player in enumerate(team_2_players):
            player_pos = player.player_body.position
            # Forward players should be closer to opponent goal
            if i < 2 and player_pos.x < self.WIDTH * 0.6:
                reward += 6.0  # Forward in good position
            # Defenders should be closer to own goal  
            elif i >= 3 and player_pos.x > self.WIDTH * 0.7:
                reward += 6.0  # Defender in good position
        
        return reward
    
    def _calculate_boundary_penalties(self, team_2_players) -> float:
        """Calculate progressive boundary penalties that increase over time"""
        penalty = 0.0
        boundary_margin = 50  # Pixels from edge to start penalty
        
        # Progressive penalty strength based on phase
        if self.current_phase == "ball_awareness":
            base_penalty = -30.0
        elif self.current_phase == "basic_soccer":
            base_penalty = -20.0
        else:
            base_penalty = -15.0
        
        for player in team_2_players:
            pos = player.player_body.position
            
            # Check each boundary
            distances_to_boundaries = [
                pos.x,  # Left edge
                self.WIDTH - pos.x,  # Right edge  
                pos.y,  # Top edge
                self.HEIGHT - pos.y  # Bottom edge
            ]
            
            for distance in distances_to_boundaries:
                if distance < boundary_margin:
                    # Penalty increases as player gets closer to boundary
                    boundary_violation = (boundary_margin - distance) / boundary_margin
                    penalty += base_penalty * boundary_violation
                    self.boundary_penalty_count += 1
        
        return penalty
    
    def _calculate_team_spread(self, players) -> float:
        """Calculate how spread out the team is (higher = better)"""
        if len(players) < 2:
            return 0
        
        total_distance = 0
        count = 0
        
        for i, player1 in enumerate(players):
            for j, player2 in enumerate(players):
                if i < j:  # Avoid double counting
                    pos1 = player1.player_body.position
                    pos2 = player2.player_body.position
                    distance = ((pos1.x - pos2.x)**2 + (pos1.y - pos2.y)**2)**0.5
                    total_distance += distance
                    count += 1
        
        return total_distance / count if count > 0 else 0
    
        
    def _opponent_ai_control(self):
        """Control team_1 using the new OpponentAI system"""
        if self.opponent_ai is not None:
            try:
                # Use OpponentAI to control team_1 players
                team_1_players = self.team_1.players()
                team_2_players = self.team_2.players()
                
                # OpponentAI controls team_1 against team_2
                self.opponent_ai.control_team(team_1_players, self.ball, team_2_players)
                
            except Exception as e:
                print(f"Error in OpponentAI control: {e}")
                # Fall back to simple opponent AI
                self._phase_based_opponent_ai()

    def render(self):
        """Render the environment"""
        if self.render_mode == "human":
            self.screen.fill((0, 100, 0))  # Green background
            
            # Draw game objects
            self.pitch.draw_pitch(self.screen)
            self.ball.draw(self.screen)
            self.team_1.draw(self.screen)
            self.team_2.draw(self.screen)
            self.match.draw(self.screen)
            
            pygame.display.flip()
            self.clock.tick(60)
            
    def close(self):
        """Clean up resources"""
        if self.render_mode == "human":
            pygame.quit()