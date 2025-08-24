"""
CTDE-Compatible Soccer Environment for Reinforcement Learning

This environment extends the existing soccer environment to support Centralized Training
Decentralized Execution (CTDE). It provides both agent-specific observations for 
decentralized execution and global observations for centralized training.

**Key Features**:
- Agent-specific observations (68-dim each) for decentralized execution  
- Global observations (102-dim) for centralized critic training
- Role-based player assignments and observations
- Backward compatibility with existing training scripts
- Support for both CTDE and traditional multi-agent training
"""

import gymnasium as gym
import numpy as np
import pygame
import pymunk
import sys
import os
from gymnasium import spaces
from typing import Dict, Tuple, Any, List

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.Vector import Vector
from player.ball import Ball
from player.team import Team, TeamAreaDimensions
from stadium.pitch import Pitch
from game.match import Match

# Import modular components
from ai.observation_builder import ObservationBuilder
from ai.reward_calculator import RewardCalculator
from ai.curriculum_manager import CurriculumManager
from ai.opponent_manager import OpponentManager


class SoccerEnvCTDE(gym.Env):
    """
    CTDE-compatible Gymnasium environment for soccer AI training.
    
    **Purpose**: Train team_2 (blue, right side) using CTDE architecture
    
    **Key Differences from Standard Environment**:
    - Supports individual agent observations (68-dim per agent)
    - Provides global observations for centralized critic (102-dim)
    - Action space remains MultiDiscrete for compatibility
    - Enhanced observation builder with role-based features
    
    **Observation Modes**:
    - 'agent': Returns list of agent-specific observations
    - 'global': Returns global observation for critic
    - 'combined': Returns both (for training)
    """
    
    def __init__(self, render_mode=None, curriculum=False, phase=None, total_timesteps=0,
                 self_play=False, opponent_model_path=None, observation_mode='agent'):
        """
        Initialize CTDE soccer environment.
        
        Args:
            observation_mode: 'agent', 'global', or 'combined'
            Other args same as original SoccerEnv
        """
        super().__init__()
        
        self.WIDTH, self.HEIGHT = 800, 600
        self.render_mode = render_mode
        self.observation_mode = observation_mode
        
        # Initialize modular components with CTDE support
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
        
        # === ACTION SPACE DEFINITION ===
        # Remains the same for compatibility: 5 players, 7 actions each
        self.action_space = spaces.MultiDiscrete([7] * 5)
        
        # === OBSERVATION SPACE DEFINITION ===
        if observation_mode == 'agent':
            # Individual agent observations flattened for SB3 compatibility: 5 * 69 = 345 dimensions
            # We'll adjust this after initialization if needed
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(345,), dtype=np.float32
            )
        elif observation_mode == 'global':
            # Global observation for critic: 102 dimensions
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(102,), dtype=np.float32
            )
        elif observation_mode == 'combined':
            # Dictionary space for both agent and global observations
            self.observation_space = spaces.Dict({
                'agent_obs': spaces.Box(low=-np.inf, high=np.inf, shape=(345,), dtype=np.float32),
                'global_obs': spaces.Box(low=-np.inf, high=np.inf, shape=(102,), dtype=np.float32)
            })
        else:
            raise ValueError(f"Invalid observation_mode: {observation_mode}")
        
        print(f"SoccerEnvCTDE initialized - Phase: {self.current_phase}, Opponent: {self.opponent_type}, Obs Mode: {observation_mode}")
        print(f"   Expected obs space: {self.observation_space.shape}")
        
        # Initialize game
        self._init_game()
        
        # Tracking
        self.steps = 0
        self.max_steps = self.curriculum_manager.get_episode_length(self.current_phase)
        self.last_ball_touch_team = None
        self.episode_start_scores = (0, 0)
        
        # Store last global observation for critic
        self._last_global_obs = None
        
        # Adjust observation space to match actual dimensions after initialization
        if observation_mode == 'agent':
            temp_obs = self._get_temp_observation_for_sizing()
            if temp_obs is not None:
                actual_dims = temp_obs.shape[0]
                print(f"CTDE Environment: Actual observation dims = {actual_dims}")
                if actual_dims != 345:
                    print(f"CTDE Environment: Adjusting observation space from 345 to {actual_dims} dimensions")
                    self.observation_space = spaces.Box(
                        low=-np.inf, high=np.inf, shape=(actual_dims,), dtype=np.float32
                    )
                else:
                    print("CTDE Environment: Observation space matches expected 345 dimensions")
    
    def _get_temp_observation_for_sizing(self):
        """Get a temporary observation to determine the actual observation space size"""
        try:
            # Temporarily create a minimal observation to determine dimensions
            temp_agent_obs = self.observation_builder.build_agent_observations(
                ball=self.ball,
                team_2_players=self.team_2.players(),
                team_1_players=self.team_1.players(),
                match=self.match,
                steps=0,
                max_steps=500
            )
            return np.concatenate(temp_agent_obs, dtype=np.float32)
        except:
            # If initialization fails, return None to use default
            return None
    
    def _init_game(self):
        """Initialize the game environment (same as original)"""
        pygame.init()
        
        if self.render_mode == "human":
            self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
            pygame.display.set_caption("Soccer AI CTDE Training")
            self.clock = pygame.time.Clock()
        else:
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
    
    def reset(self, seed=None, options=None) -> Tuple[Any, Dict[str, Any]]:
        """Reset environment and return observations based on mode"""
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
        
        obs = self._get_observations()
        return obs, {}
    
    def step(self, action) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        """Execute one environment step with CTDE observations"""
        
        # Handle different action formats
        if isinstance(action, (list, tuple)):
            action = np.array(action)
        if len(action.shape) == 2 and action.shape[0] == 1:
            action = action[0]  # Remove batch dimension if present
        
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
        
        # Update match (with fake keys)
        fake_keys = {i: False for i in range(512)}
        self.match.update(fake_keys)
        
        # Get observations
        obs = self._get_observations()
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Check termination
        self.steps += 1
        goal_scored = (self.match.goal1_score > 0 or self.match.goal2_score > 0)
        terminated = goal_scored or self.steps >= self.max_steps
        truncated = False
        
        return obs, reward, terminated, truncated, {}
    
    def _apply_actions(self, actions):
        """Apply actions to team_2 players (same as original)"""
        team_2_players = self.team_2.players()
        
        for i, action in enumerate(actions):
            if i >= len(team_2_players):
                break
                
            player = team_2_players[i]
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
    
    def _get_observations(self) -> Any:
        """Get observations based on the current observation mode"""
        
        if self.observation_mode == 'agent':
            # Return agent-specific observations flattened for SB3
            agent_obs = self.observation_builder.build_agent_observations(
                ball=self.ball,
                team_2_players=self.team_2.players(),
                team_1_players=self.team_1.players(),
                match=self.match,
                steps=self.steps,
                max_steps=self.max_steps
            )
            # Flatten the list of observations into a single vector
            flattened_obs = np.concatenate(agent_obs, dtype=np.float32)
            return flattened_obs
        
        elif self.observation_mode == 'global':
            # Return global observation
            global_obs = self.observation_builder.build_global_observation(
                ball=self.ball,
                team_2_players=self.team_2.players(),
                team_1_players=self.team_1.players(),
                match=self.match,
                steps=self.steps,
                max_steps=self.max_steps
            )
            self._last_global_obs = global_obs
            return global_obs
        
        elif self.observation_mode == 'combined':
            # Return both agent and global observations
            agent_obs = self.observation_builder.build_agent_observations(
                ball=self.ball,
                team_2_players=self.team_2.players(),
                team_1_players=self.team_1.players(),
                match=self.match,
                steps=self.steps,
                max_steps=self.max_steps
            )
            global_obs = self.observation_builder.build_global_observation(
                ball=self.ball,
                team_2_players=self.team_2.players(),
                team_1_players=self.team_1.players(),
                match=self.match,
                steps=self.steps,
                max_steps=self.max_steps
            )
            self._last_global_obs = global_obs
            
            return {
                'agent_obs': np.concatenate(agent_obs, dtype=np.float32),
                'global_obs': global_obs
            }
        
        else:
            raise ValueError(f"Unknown observation mode: {self.observation_mode}")
    
    def get_global_observation(self) -> np.ndarray:
        """Get the latest global observation for critic training"""
        if self._last_global_obs is not None:
            return self._last_global_obs
        else:
            # Generate global observation if not cached
            return self.observation_builder.build_global_observation(
                ball=self.ball,
                team_2_players=self.team_2.players(),
                team_1_players=self.team_1.players(),
                match=self.match,
                steps=self.steps,
                max_steps=self.max_steps
            )
    
    def _calculate_reward(self) -> float:
        """Calculate reward (same as original)"""
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
        """Render the environment (same as original)"""
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
    
    # === ADDITIONAL CTDE-SPECIFIC METHODS ===
    
    def get_agent_observations(self) -> List[np.ndarray]:
        """Get individual agent observations for decentralized execution"""
        return self.observation_builder.build_agent_observations(
            ball=self.ball,
            team_2_players=self.team_2.players(),
            team_1_players=self.team_1.players(),
            match=self.match,
            steps=self.steps,
            max_steps=self.max_steps
        )
    
    def get_training_info(self) -> dict:
        """Get detailed training information including CTDE metrics"""
        agent_obs = self.get_agent_observations()
        global_obs = self.get_global_observation()
        
        return {
            'current_phase': self.current_phase,
            'episode_steps': self.steps,
            'max_steps': self.max_steps,
            'opponent_type': self.opponent_type,
            'observation_mode': self.observation_mode,
            'agent_obs_shape': [obs.shape for obs in agent_obs],
            'global_obs_shape': global_obs.shape,
            'curriculum_summary': self.curriculum_manager.get_training_summary(0),
            'opponent_info': self.opponent_manager.get_opponent_info(self.opponent_type)
        }
    
    def set_observation_mode(self, mode: str):
        """Change observation mode dynamically"""
        if mode in ['agent', 'global', 'combined']:
            self.observation_mode = mode
            print(f"Observation mode set to: {mode}")
        else:
            print(f"Invalid observation mode: {mode}. Valid modes: ['agent', 'global', 'combined']")


def create_ctde_env(observation_mode='agent', **kwargs):
    """
    Convenience function to create CTDE environment.
    
    Args:
        observation_mode: 'agent', 'global', or 'combined'
        **kwargs: Other environment arguments
        
    Returns:
        SoccerEnvCTDE: Configured CTDE environment
    """
    return SoccerEnvCTDE(observation_mode=observation_mode, **kwargs)