"""
Reward Calculation System for Soccer Environment

This module implements phase-based reward strategies for curriculum learning.
Each phase focuses on different aspects of soccer gameplay, from basic ball
awareness to complex strategic play.

**Key Features**:
- Phase-specific reward strategies
- Configurable reward values
- Clear separation of reward logic
- Extensible design for new reward types
"""

from abc import ABC, abstractmethod
from typing import List
import numpy as np
from config.game_config import GameConfig


class RewardStrategy(ABC):
    """Abstract base class for reward calculation strategies"""
    
    @abstractmethod
    def calculate_reward(self, game_state: dict) -> float:
        """Calculate reward based on current game state"""
        pass


class BallAwarenessRewards(RewardStrategy):
    """
    Phase 1 rewards focusing on basic ball-seeking behavior.
    
    **Goals**:
    - Encourage players to approach the ball
    - Prevent boundary camping
    - Reward ball control and movement toward ball
    - Simple goal scoring rewards
    """
    
    def __init__(self):
        # Reward constants
        self.GOAL_SCORED_REWARD = 500.0
        self.GOAL_CONCEDED_PENALTY = -250.0
        self.BALL_CONTROL_BONUS = 50.0
        self.PROXIMITY_REWARD_BASE = 15.0
        self.MOVEMENT_REWARD_BASE = 8.0
        self.IGNORE_BALL_PENALTY = -15.0
        self.BOUNDARY_BASE_PENALTY = -30.0
        
        # Thresholds
        self.MAX_MEANINGFUL_DISTANCE = 300.0
        self.MIN_BALL_ATTENTION_DISTANCE = 150.0
        self.MIN_MOVEMENT_SPEED = 10.0
        
        # Game constants
        self.config = GameConfig()
        self.DRIBBLE_DISTANCE = self.config.ball_control.DRIBBLE_DISTANCE
        self.MOVEMENT_ALIGNMENT_THRESHOLD = 0.5
        self.BOUNDARY_MARGIN = 50.0
    
    def calculate_reward(self, game_state: dict) -> float:
        """Calculate ball awareness phase rewards"""
        reward = 0.0
        
        # Extract game state
        ball_pos = game_state['ball_position']
        team_2_players = game_state['team_2_players']
        match = game_state['match']
        field_width = game_state['field_width']
        field_height = game_state['field_height']
        
        # === GOAL REWARDS ===
        if match.goal1_score > 0:
            reward += self.GOAL_SCORED_REWARD
        if match.goal2_score > 0:
            reward += self.GOAL_CONCEDED_PENALTY
        
        # === BALL INTERACTION REWARDS ===
        total_proximity_reward = 0.0
        ball_control_bonus = 0.0
        min_distance = float('inf')
        
        for player in team_2_players:
            player_pos = player.player_body.position
            player_vel = player.player_body.velocity
            distance = (ball_pos - player_pos).length
            min_distance = min(min_distance, distance)
            
            # Individual proximity reward (exponential)
            if distance < self.MAX_MEANINGFUL_DISTANCE:
                proximity_factor = 1 - (distance / self.MAX_MEANINGFUL_DISTANCE)
                individual_reward = self.PROXIMITY_REWARD_BASE * (proximity_factor ** 2)
                total_proximity_reward += individual_reward
            
            # Ball control bonus
            if distance < self.DRIBBLE_DISTANCE:
                ball_control_bonus = self.BALL_CONTROL_BONUS
            
            # Movement toward ball reward
            to_ball = ball_pos - player_pos
            if (to_ball.length > self.DRIBBLE_DISTANCE and 
                to_ball.length > 0 and player_vel.length > self.MIN_MOVEMENT_SPEED):
                
                to_ball_normalized = to_ball.normalized()
                vel_normalized = player_vel.normalized()
                alignment = to_ball_normalized.dot(vel_normalized)
                
                if alignment > self.MOVEMENT_ALIGNMENT_THRESHOLD:
                    movement_reward = (self.MOVEMENT_REWARD_BASE * alignment * 
                                     (player_vel.length / 100.0))
                    reward += movement_reward
        
        reward += total_proximity_reward
        reward += ball_control_bonus
        
        # === PENALTY FOR IGNORING BALL ===
        if min_distance > self.MIN_BALL_ATTENTION_DISTANCE:
            reward += self.IGNORE_BALL_PENALTY
        
        # === BOUNDARY PENALTIES ===
        boundary_penalty = self._calculate_boundary_penalties(
            team_2_players, field_width, field_height
        )
        reward += boundary_penalty
        
        return reward
    
    def _calculate_boundary_penalties(self, players: List, width: int, height: int) -> float:
        """Calculate penalties for being near field boundaries"""
        penalty = 0.0
        
        for player in players:
            pos = player.player_body.position
            
            distances_to_boundaries = [
                pos.x,                # Left edge
                width - pos.x,        # Right edge
                pos.y,                # Top edge
                height - pos.y        # Bottom edge
            ]
            
            for distance in distances_to_boundaries:
                if distance < self.BOUNDARY_MARGIN:
                    violation = (self.BOUNDARY_MARGIN - distance) / self.BOUNDARY_MARGIN
                    penalty += self.BOUNDARY_BASE_PENALTY * violation
        
        return penalty


class BasicSoccerRewards(RewardStrategy):
    """
    Phase 2 rewards adding shooting, passing, and basic positioning.
    
    **Goals**:
    - Encourage shooting and passing actions
    - Reward team coordination
    - Basic formation maintenance
    - Enhanced goal scoring rewards
    """
    
    def __init__(self):
        # Enhanced reward constants
        self.GOAL_SCORED_REWARD = 750.0
        self.GOAL_CONCEDED_PENALTY = -400.0
        self.BALL_CONTROL_BONUS = 30.0
        self.PROXIMITY_REWARD_BASE = 15.0
        self.SHOOTING_ATTEMPT_REWARD = 15.0
        self.PASSING_ATTEMPT_REWARD = 12.0
        self.FORMATION_BONUS = 5.0
        self.BOUNDARY_BASE_PENALTY = -20.0
        
        # Thresholds
        self.MIN_SHOT_VELOCITY = 100.0
        self.MIN_PASS_VELOCITY = 80.0
        self.SHOT_RANGE_THRESHOLD = 200.0
        self.PASS_DISTANCE_MIN = 50.0
        self.PASS_DISTANCE_MAX = 150.0
        self.MIN_TEAM_SPREAD = 100.0
        
        # Game constants
        self.config = GameConfig()
        self.DRIBBLE_DISTANCE = self.config.ball_control.DRIBBLE_DISTANCE
    
    def calculate_reward(self, game_state: dict) -> float:
        """Calculate basic soccer phase rewards"""
        reward = 0.0
        
        # Extract game state
        ball_pos = game_state['ball_position']
        ball_vel = game_state['ball_velocity']
        team_2_players = game_state['team_2_players']
        match = game_state['match']
        field_width = game_state['field_width']
        field_height = game_state['field_height']
        
        # === ENHANCED GOAL REWARDS ===
        if match.goal1_score > 0:
            reward += self.GOAL_SCORED_REWARD
        if match.goal2_score > 0:
            reward += self.GOAL_CONCEDED_PENALTY
        
        # === BALL INTERACTION REWARDS ===
        team_2_has_ball = False
        min_distance = float('inf')
        
        for player in team_2_players:
            distance = (ball_pos - player.player_body.position).length
            min_distance = min(min_distance, distance)
            if distance < self.DRIBBLE_DISTANCE:
                team_2_has_ball = True
                reward += self.BALL_CONTROL_BONUS
                break
        
        # Ball proximity reward
        if min_distance < field_width:
            proximity_reward = self.PROXIMITY_REWARD_BASE * (
                1 - min_distance / (field_width * 0.7)
            )
            reward += max(0, proximity_reward)
        
        # === ACTION REWARDS ===
        reward += self._calculate_action_rewards(ball_pos, ball_vel, team_2_players)
        
        # === POSITIONING REWARDS ===
        team_spread = self._calculate_team_spread(team_2_players)
        if team_spread > self.MIN_TEAM_SPREAD:
            reward += self.FORMATION_BONUS
        
        # === BOUNDARY PENALTIES ===
        boundary_penalty = self._calculate_boundary_penalties(
            team_2_players, field_width, field_height
        )
        reward += boundary_penalty
        
        return reward
    
    def _calculate_action_rewards(self, ball_pos, ball_vel, team_2_players: List) -> float:
        """Calculate rewards for shooting and passing actions"""
        reward = 0.0
        
        # Shooting attempt detection
        if ball_vel.length > self.MIN_SHOT_VELOCITY:
            goal_pos = (50, 300)  # Left goal
            ball_to_goal_dist = ((goal_pos[0] - ball_pos.x)**2 + (goal_pos[1] - ball_pos.y)**2)**0.5
            
            if ball_to_goal_dist < self.SHOT_RANGE_THRESHOLD:
                # Check if ball is moving toward goal
                ball_dir = (ball_vel.x, ball_vel.y)
                goal_dir = (goal_pos[0] - ball_pos.x, goal_pos[1] - ball_pos.y)
                goal_dir_length = (goal_dir[0]**2 + goal_dir[1]**2)**0.5
                
                if goal_dir_length > 0:
                    dot_product = ((ball_dir[0] * goal_dir[0] + ball_dir[1] * goal_dir[1]) / 
                                  goal_dir_length)
                    if dot_product > 50:  # Ball moving toward goal
                        reward += self.SHOOTING_ATTEMPT_REWARD
        
        # Passing attempt detection
        if ball_vel.length > self.MIN_PASS_VELOCITY:
            for player in team_2_players:
                player_pos = player.player_body.position
                ball_to_player_dist = ((player_pos.x - ball_pos.x)**2 + 
                                      (player_pos.y - ball_pos.y)**2)**0.5
                
                if self.PASS_DISTANCE_MIN < ball_to_player_dist < self.PASS_DISTANCE_MAX:
                    ball_dir = (ball_vel.x, ball_vel.y)
                    player_dir = (player_pos.x - ball_pos.x, player_pos.y - ball_pos.y)
                    player_dir_length = (player_dir[0]**2 + player_dir[1]**2)**0.5
                    
                    if player_dir_length > 0:
                        dot_product = ((ball_dir[0] * player_dir[0] + 
                                       ball_dir[1] * player_dir[1]) / player_dir_length)
                        if dot_product > 40:  # Ball moving toward teammate
                            reward += self.PASSING_ATTEMPT_REWARD
                            break
        
        return reward
    
    def _calculate_team_spread(self, players: List) -> float:
        """Calculate how spread out the team is"""
        if len(players) < 2:
            return 0
        
        total_distance = 0
        count = 0
        
        for i, player1 in enumerate(players):
            for j, player2 in enumerate(players):
                if i < j:
                    pos1 = player1.player_body.position
                    pos2 = player2.player_body.position
                    distance = ((pos1.x - pos2.x)**2 + (pos1.y - pos2.y)**2)**0.5
                    total_distance += distance
                    count += 1
        
        return total_distance / count if count > 0 else 0
    
    def _calculate_boundary_penalties(self, players: List, width: int, height: int) -> float:
        """Calculate penalties for being near field boundaries"""
        penalty = 0.0
        boundary_margin = 50
        
        for player in players:
            pos = player.player_body.position
            
            distances = [pos.x, width - pos.x, pos.y, height - pos.y]
            
            for distance in distances:
                if distance < boundary_margin:
                    violation = (boundary_margin - distance) / boundary_margin
                    penalty += self.BOUNDARY_BASE_PENALTY * violation
        
        return penalty


class CompetitiveSoccerRewards(RewardStrategy):
    """
    Phase 3 rewards for full strategic and tactical gameplay.
    
    **Goals**:
    - Advanced tactical rewards
    - Role-based positioning
    - Strategic passing and shooting
    - Maximum challenge and realism
    """
    
    def __init__(self):
        # Advanced reward constants
        self.GOAL_SCORED_REWARD = 1000.0
        self.GOAL_CONCEDED_PENALTY = -500.0
        self.BALL_CONTROL_BONUS = 25.0
        self.PROXIMITY_REWARD_BASE = 12.0
        self.POSSESSION_BONUS = 2.0
        self.OPPONENT_POSSESSION_PENALTY = -3.0
        self.FORWARD_PASS_REWARD = 25.0
        self.SHOT_ACCURACY_REWARD = 35.0
        self.FORMATION_BONUS = 8.0
        self.POSITIONING_BONUS = 6.0
        self.BOUNDARY_BASE_PENALTY = -15.0
        
        # Advanced thresholds
        self.MIN_FORWARD_PASS_VELOCITY = 80.0
        self.MIN_SHOT_VELOCITY = 120.0
        self.PASS_DISTANCE_MIN = 60.0
        self.PASS_DISTANCE_MAX = 200.0
        self.SHOT_RANGE_THRESHOLD = 250.0
        self.MIN_TEAM_SPREAD = 120.0
        
        # Game constants
        self.config = GameConfig()
        self.DRIBBLE_DISTANCE = self.config.ball_control.DRIBBLE_DISTANCE
    
    def calculate_reward(self, game_state: dict) -> float:
        """Calculate competitive soccer phase rewards"""
        reward = 0.0
        
        # Extract game state
        ball_pos = game_state['ball_position']
        ball_vel = game_state['ball_velocity']
        team_2_players = game_state['team_2_players']
        team_1_players = game_state['team_1_players']
        match = game_state['match']
        field_width = game_state['field_width']
        field_height = game_state['field_height']
        
        # === MAXIMUM GOAL REWARDS ===
        if match.goal1_score > 0:
            reward += self.GOAL_SCORED_REWARD
        if match.goal2_score > 0:
            reward += self.GOAL_CONCEDED_PENALTY
        
        # === ADVANCED BALL CONTROL ===
        team_2_has_ball, team_1_has_ball, min_distance = self._analyze_ball_control(
            team_2_players, team_1_players, ball_pos
        )
        
        if team_2_has_ball:
            reward += self.BALL_CONTROL_BONUS
            reward += self.POSSESSION_BONUS
        elif team_1_has_ball:
            reward += self.OPPONENT_POSSESSION_PENALTY
        
        # Ball proximity reward
        if min_distance < field_width:
            proximity_reward = self.PROXIMITY_REWARD_BASE * (
                1 - min_distance / (field_width * 0.7)
            )
            reward += max(0, proximity_reward)
        
        # === STRATEGIC ACTION REWARDS ===
        reward += self._calculate_strategic_rewards(ball_pos, ball_vel, team_2_players)
        
        # === FORMATION AND POSITIONING ===
        reward += self._calculate_formation_rewards(team_2_players, field_width, field_height)
        
        return reward
    
    def _analyze_ball_control(self, team_2_players: List, team_1_players: List, ball_pos):
        """Analyze which team has ball control"""
        team_2_has_ball = False
        team_1_has_ball = False
        min_distance = float('inf')
        
        # Check team_2 control
        for player in team_2_players:
            distance = (ball_pos - player.player_body.position).length
            min_distance = min(min_distance, distance)
            if distance < self.DRIBBLE_DISTANCE:
                team_2_has_ball = True
                break
        
        # Check team_1 control if team_2 doesn't have it
        if not team_2_has_ball:
            for player in team_1_players:
                distance = (ball_pos - player.player_body.position).length
                if distance < self.DRIBBLE_DISTANCE:
                    team_1_has_ball = True
                    break
        
        return team_2_has_ball, team_1_has_ball, min_distance
    
    def _calculate_strategic_rewards(self, ball_pos, ball_vel, team_2_players: List) -> float:
        """Calculate rewards for strategic passing and shooting"""
        reward = 0.0
        
        # Forward passing rewards
        if ball_vel.length > self.MIN_FORWARD_PASS_VELOCITY:
            for player in team_2_players:
                player_pos = player.player_body.position
                ball_to_player_dist = ((player_pos.x - ball_pos.x)**2 + 
                                      (player_pos.y - ball_pos.y)**2)**0.5
                
                if self.PASS_DISTANCE_MIN < ball_to_player_dist < self.PASS_DISTANCE_MAX:
                    # Check if this is a forward pass
                    if player_pos.x < ball_pos.x:  # Forward toward goal
                        ball_dir = (ball_vel.x, ball_vel.y)
                        player_dir = (player_pos.x - ball_pos.x, player_pos.y - ball_pos.y)
                        player_dir_length = (player_dir[0]**2 + player_dir[1]**2)**0.5
                        
                        if player_dir_length > 0:
                            dot_product = ((ball_dir[0] * player_dir[0] + 
                                           ball_dir[1] * player_dir[1]) / player_dir_length)
                            if dot_product > 50:
                                reward += self.FORWARD_PASS_REWARD
                                break
        
        # Shot accuracy rewards
        if ball_vel.length > self.MIN_SHOT_VELOCITY:
            goal_center = (50, 300)
            goal_dir = (goal_center[0] - ball_pos.x, goal_center[1] - ball_pos.y)
            goal_distance = (goal_dir[0]**2 + goal_dir[1]**2)**0.5
            
            if goal_distance < self.SHOT_RANGE_THRESHOLD:
                ball_dir = (ball_vel.x, ball_vel.y)
                if goal_distance > 0:
                    dot_product = ((ball_dir[0] * goal_dir[0] + ball_dir[1] * goal_dir[1]) / 
                                  goal_distance)
                    if dot_product > 60:  # Good shot toward goal
                        reward += self.SHOT_ACCURACY_REWARD
        
        return reward
    
    def _calculate_formation_rewards(self, team_2_players: List, width: int, height: int) -> float:
        """Calculate rewards for good team formation and positioning"""
        reward = 0.0
        
        # Team spread reward
        team_spread = self._calculate_team_spread(team_2_players)
        if team_spread > self.MIN_TEAM_SPREAD:
            reward += self.FORMATION_BONUS
        
        # Role-based positioning (simplified)
        for i, player in enumerate(team_2_players):
            player_pos = player.player_body.position
            
            # Forward players should be closer to opponent goal
            if i < 2 and player_pos.x < width * 0.6:
                reward += self.POSITIONING_BONUS
            # Defenders should be closer to own goal
            elif i >= 3 and player_pos.x > width * 0.7:
                reward += self.POSITIONING_BONUS
        
        return reward
    
    def _calculate_team_spread(self, players: List) -> float:
        """Calculate how spread out the team is"""
        if len(players) < 2:
            return 0
        
        total_distance = 0
        count = 0
        
        for i, player1 in enumerate(players):
            for j, player2 in enumerate(players):
                if i < j:
                    pos1 = player1.player_body.position
                    pos2 = player2.player_body.position
                    distance = ((pos1.x - pos2.x)**2 + (pos1.y - pos2.y)**2)**0.5
                    total_distance += distance
                    count += 1
        
        return total_distance / count if count > 0 else 0


class RewardCalculator:
    """
    Main reward calculator that manages phase-based reward strategies.
    
    **Usage**:
    ```python
    calculator = RewardCalculator()
    reward = calculator.calculate_reward(current_phase, game_state)
    ```
    """
    
    def __init__(self):
        self.strategies = {
            "ball_awareness": BallAwarenessRewards(),
            "basic_soccer": BasicSoccerRewards(),
            "competitive_soccer": CompetitiveSoccerRewards()
        }
    
    def calculate_reward(self, phase: str, game_state: dict) -> float:
        """Calculate reward using the appropriate strategy for the current phase"""
        if phase not in self.strategies:
            phase = "competitive_soccer"  # Default to full difficulty
        
        return self.strategies[phase].calculate_reward(game_state)
    
    def get_available_phases(self) -> List[str]:
        """Get list of available reward phases"""
        return list(self.strategies.keys())