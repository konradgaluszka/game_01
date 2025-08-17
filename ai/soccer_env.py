import gymnasium as gym
import numpy as np
import pygame
import pymunk
from gymnasium import spaces
from typing import Dict, Tuple, Any

from common.Vector import Vector
from player.ball import Ball
from player.team import Team, TeamAreaDimensions
from stadium.pitch import Pitch
from game.match import Match


class SoccerEnv(gym.Env):
    """
    Custom Environment for soccer AI training using reinforcement learning.
    The AI controls team_2 (right side, blue team).
    """
    
    def __init__(self, render_mode=None):
        super().__init__()
        
        self.WIDTH, self.HEIGHT = 800, 600
        self.render_mode = render_mode
        
        # Action space: Each of 5 players can move in 4 directions + shoot/dribble
        # Action per player: [up, down, left, right, shoot] (5 discrete actions)
        self.action_space = spaces.MultiDiscrete([5] * 5)  # 5 players, 5 actions each
        
        # Observation space: positions and velocities of all entities
        # Ball: pos(2) + vel(2) = 4
        # Each player (10 total): pos(2) + vel(2) = 40
        # Goal positions and match state = 6
        obs_dim = 4 + 40 + 6  # 50 total
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        
        # Initialize pygame and physics
        self._init_game()
        
        # Tracking
        self.steps = 0
        self.max_steps = 1000
        self.last_ball_touch_team = None
        
    def _init_game(self):
        """Initialize the game environment"""
        if self.render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
            pygame.display.set_caption("Soccer AI Training")
            self.clock = pygame.time.Clock()
        
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
        
    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment to initial state"""
        super().reset(seed=seed)
        
        # Reset game objects
        self.match.reset()
        self.steps = 0
        self.last_ball_touch_team = None
        
        return self._get_observation(), {}
        
    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one environment step"""
        # Apply actions to team_2 players
        self._apply_actions(action)
        
        # Team_1 does nothing (or could have simple AI behavior)
        # In training, we focus on team_2 learning
        
        # Update physics
        self.space.step(1/60)  # 60 FPS
        
        # Update game objects
        self.team_1.simulate()
        self.team_2.simulate()
        self.ball.simulate()
        self.match.update({})  # No keyboard input during training
        
        # Get new observation
        obs = self._get_observation()
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Check if episode is done
        self.steps += 1
        terminated = self.match.goal_scored or self.steps >= self.max_steps
        truncated = False
        
        return obs, reward, terminated, truncated, {}
        
    def _apply_actions(self, actions):
        """Apply actions to team_2 players"""
        team_2_players = self.team_2.players()
        
        for i, action in enumerate(actions):
            if i >= len(team_2_players):
                break
                
            player = team_2_players[i]
            
            # Action mapping: 0=nothing, 1=up, 2=down, 3=left, 4=right, 5=shoot
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
                    
    def _get_observation(self) -> np.ndarray:
        """Get current state observation"""
        obs = []
        
        # Ball position and velocity
        ball_pos = self.ball.ball_body.position
        ball_vel = self.ball.ball_body.velocity
        obs.extend([ball_pos.x / self.WIDTH, ball_pos.y / self.HEIGHT])
        obs.extend([ball_vel.x / 1000, ball_vel.y / 1000])  # Normalize velocity
        
        # All players positions and velocities
        for player in self.all_players:
            pos = player.player_body.position
            vel = player.player_body.velocity
            obs.extend([pos.x / self.WIDTH, pos.y / self.HEIGHT])
            obs.extend([vel.x / 1000, vel.y / 1000])
            
        # Goal positions (normalized)
        obs.extend([0.0, 0.5])  # Left goal center
        obs.extend([1.0, 0.5])  # Right goal center
        
        # Match state
        obs.extend([
            float(self.match.goal_scored),
            float(self.steps / self.max_steps)
        ])
        
        return np.array(obs, dtype=np.float32)
        
    def _calculate_reward(self) -> float:
        """Calculate reward for the current state"""
        reward = 0.0
        
        # Goal rewards
        if self.match.goal_scored:
            if self.match.team_2_score > self.match.team_1_score:
                reward += 100.0  # Team_2 scored
            else:
                reward -= 50.0   # Team_1 scored
                
        # Ball proximity rewards for team_2 players
        ball_pos = self.ball.ball_body.position
        team_2_players = self.team_2.players()
        
        min_distance = float('inf')
        for player in team_2_players:
            distance = (ball_pos - player.player_body.position).length
            min_distance = min(min_distance, distance)
            
        # Reward being close to ball
        reward += max(0, (100 - min_distance) / 100) * 0.1
        
        # Ball movement towards opponent goal (left goal)
        ball_to_goal = abs(ball_pos.x - 0)  # Distance to left goal
        reward += (self.WIDTH - ball_to_goal) / self.WIDTH * 0.05
        
        # Small penalty for time to encourage faster play
        reward -= 0.01
        
        return reward
        
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