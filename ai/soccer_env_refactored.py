"""Refactored Soccer Environment for Reinforcement Learning

This file implements a clean, maintainable Gymnasium-compatible environment
for soccer AI training. The monolithic original has been refactored into
focused, single-responsibility components:

**Architecture**:
- **ObservationBuilder**: Handles 62-dimensional state vector generation
- **RewardCalculator**: Phase-based curriculum reward strategies  
- **CurriculumManager**: Progressive difficulty and phase management
- **OpponentManager**: Multiple opponent AI types and self-play
- **SoccerEnv**: Core gym interface and coordination

**Benefits**:
- Clear separation of concerns
- Easier testing and maintenance
- Configurable curriculum and opponents
- Reduced coupling between systems
- Extensible reward and opponent strategies
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

# Import new modular components
from ai.observation_builder import ObservationBuilder
from ai.reward_calculator import RewardCalculator
from ai.curriculum_manager import CurriculumManager
from ai.opponent_manager import OpponentManager


class SoccerEnv(gym.Env):
    """
    Refactored Gymnasium environment for soccer AI training.
    
    **Purpose**: Train team_2 (blue, right side) to play soccer effectively
    
    **New Architecture Benefits**:
    - Modular components for observation, rewards, curriculum, and opponents
    - Clear separation of concerns and single responsibility principle
    - Easier testing, maintenance, and extension
    - Configurable curriculum learning and opponent strategies
    
    **Core Responsibilities**:
    - Implement Gymnasium interface (reset, step, render, close)
    - Coordinate between game objects and AI components
    - Manage episode lifecycle and termination conditions
    - Handle action application to player objects
    """
    
    def __init__(self, render_mode=None, curriculum=False, phase=None, total_timesteps=0, 
                 self_play=False, opponent_model_path=None):
        """
        Initialize the refactored soccer training environment.
        
        Args:
            render_mode: Optional rendering mode ("human" for visual display, None for headless)
            curriculum: Enable curriculum learning with automatic phase progression
            phase: Manual phase selection ("ball_awareness", "basic_soccer", "competitive_soccer")
            total_timesteps: Current total timesteps for automatic phase detection
            self_play: Enable self-play training (team_1 controlled by AI)
            opponent_model_path: Path to opponent model for self-play
        """
        super().__init__()
        
        self.WIDTH, self.HEIGHT = 800, 600
        self.render_mode = render_mode
        
        # Initialize modular components
        self.curriculum_manager = CurriculumManager(manual_phase=phase, enable_curriculum=curriculum)
        self.observation_builder = ObservationBuilder(field_width=self.WIDTH, field_height=self.HEIGHT)
        self.reward_calculator = RewardCalculator()
        self.opponent_manager = OpponentManager(field_width=self.WIDTH, field_height=self.HEIGHT)
        
        # Load self-play model if provided
        if self_play and opponent_model_path:
            self._load_opponent_model(opponent_model_path)
        
        # Determine current phase and opponent type
        self.current_phase = self.curriculum_manager.get_current_phase(total_timesteps)
        self.opponent_type = "self_play" if self_play else "opponent_ai"
        
        print(f"Environment initialized - Phase: {self.current_phase}, Opponent: {self.opponent_type}")
        
        # === ACTION SPACE DEFINITION ===
        # Each of 5 team_2 players can perform one of 7 actions per timestep:
        # 0 = do nothing, 1 = move up, 2 = move down, 3 = move left, 4 = move right
        # 5 = shoot (toward goal), 6 = pass (toward nearest teammate)
        self.action_space = spaces.MultiDiscrete([7] * 5)  # 5 players, 7 actions each
        
        # === OBSERVATION SPACE DEFINITION ===
        # 62-dimensional observation vector (managed by ObservationBuilder)
        obs_dim = 12 + 25 + 15 + 6 + 4  # 62 total
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        
        # Initialize pygame and physics
        self._init_game()
        
        # Tracking
        self.steps = 0
        self.max_steps = self.curriculum_manager.get_episode_length(self.current_phase)
        self.last_ball_touch_team = None
        self.episode_start_scores = (0, 0)

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
    
    def _load_opponent_model(self, model_path: str):
        """Load opponent model for self-play"""
        try:
            from stable_baselines3 import PPO
            model = PPO.load(model_path)
            self.opponent_manager.set_self_play_model(model)
            print(f"Loaded opponent model from {model_path}")
        except Exception as e:
            print(f"Failed to load opponent model: {e}")
    
    def update_opponent_model(self, new_model_path):
        """Update the opponent model during training"""
        self.opponent_manager.update_self_play_model(new_model_path)
    
    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment to initial state"""
        super().reset(seed=seed)
        
        # Reset game objects
        self.match.reset()
        self.steps = 0
        self.last_ball_touch_team = None
        self.episode_start_scores = (self.match.goal1_score, self.match.goal2_score)
        
        # Set ball position based on current training phase
        ball_x, ball_y = self.curriculum_manager.get_ball_start_position(self.current_phase)
        self.ball.ball_body.position = (ball_x, ball_y)
        self.ball.ball_body.velocity = (0, 0)
        
        return self._get_observation(), {}
        
    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one environment step"""
        # Apply actions to team_2 players
        self._apply_actions(action)
        
        # Control team_1 based on opponent type
        self.opponent_manager.control_opponent_team(
            team_1_players=self.team_1.players(),
            ball=self.ball,
            team_2_players=self.team_2.players(),
            opponent_type=self.opponent_type,
            current_phase=self.current_phase,
            observation_builder=self.observation_builder
        )
        
        # Update physics
        self.space.step(1/60)  # 60 FPS
        
        # Update game objects
        self.team_1.simulate()
        self.team_2.simulate()
        self.ball.simulate()
        
        # Create fake keys dict for the match update (no keyboard input during training)
        fake_keys = {}
        for i in range(512):  # Cover all possible pygame keys
            fake_keys[i] = False
        self.match.update(fake_keys)
        
        # Get new observation
        obs = self._get_observation()
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Check if episode is done
        self.steps += 1
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
            
            # Map actions to movement + action booleans
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
    
    def _get_observation(self) -> np.ndarray:
        """Generate observation using the dedicated ObservationBuilder"""
        return self.observation_builder.build_observation(
            ball=self.ball,
            team_2_players=self.team_2.players(),
            team_1_players=self.team_1.players(),
            match=self.match,
            steps=self.steps,
            max_steps=self.max_steps
        )
        
    def _calculate_reward(self) -> float:
        """Calculate reward using the dedicated RewardCalculator"""
        # Build game state for reward calculation
        game_state = {
            'ball_position': self.ball.ball_body.position,
            'ball_velocity': self.ball.ball_body.velocity,
            'team_2_players': self.team_2.players(),
            'team_1_players': self.team_1.players(),
            'match': self.match,
            'field_width': self.WIDTH,
            'field_height': self.HEIGHT,
            'steps': self.steps,
            'max_steps': self.max_steps
        }
        
        return self.reward_calculator.calculate_reward(self.current_phase, game_state)

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
    
    # === ADDITIONAL HELPER METHODS ===
    
    def get_training_info(self) -> dict:
        """Get detailed training information"""
        return {
            'current_phase': self.current_phase,
            'episode_steps': self.steps,
            'max_steps': self.max_steps,
            'opponent_type': self.opponent_type,
            'curriculum_summary': self.curriculum_manager.get_training_summary(0),
            'opponent_info': self.opponent_manager.get_opponent_info(self.opponent_type)
        }
    
    def set_phase(self, phase: str):
        """Manually set the training phase"""
        if phase in self.reward_calculator.get_available_phases():
            self.current_phase = phase
            self.max_steps = self.curriculum_manager.get_episode_length(phase)
            print(f"Phase set to: {phase} (episode length: {self.max_steps})")
        else:
            print(f"Invalid phase: {phase}. Available: {self.reward_calculator.get_available_phases()}")
    
    def set_opponent_type(self, opponent_type: str):
        """Set the opponent type for training"""
        valid_types = ["self_play", "opponent_ai", "phase_based"]
        if opponent_type in valid_types:
            self.opponent_type = opponent_type
            print(f"Opponent type set to: {opponent_type}")
        else:
            print(f"Invalid opponent type: {opponent_type}. Available: {valid_types}")