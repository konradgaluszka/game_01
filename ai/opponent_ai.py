"""
Comprehensive Opponent AI System for Soccer Training

This module provides a sophisticated rule-based AI opponent that uses the same
player physics interface as human control, ensuring identical behavior and fairness.
The AI implements realistic soccer behaviors including goalkeeper positioning,
defensive lines, strategic passing, and coordinated attacks.

**Key Features**:
1. **Goalkeeper AI**: Keeps goalkeeper in goal area with intelligent positioning
2. **Defensive Line**: Maintains defensive formation and intercepts passes
3. **Strategic Passing**: Defenders pass to attackers when receiving the ball
4. **Attack Coordination**: Attackers attempt to score with positioning and shots
5. **Role-Based Behavior**: Each player has a specific role and position

**Usage**:
```python
from ai.opponent_ai import OpponentAI

# Create opponent AI for team_1 (left side)
opponent = OpponentAI(team_side="left", field_width=800, field_height=600)

# Each frame, control the team
opponent.control_team(team_1_players, ball, team_2_players)
```
"""

import time
import numpy as np
import pymunk
from typing import List, Tuple, Optional
from enum import Enum


class PlayerRole(Enum):
    """Define player roles for position-based behavior"""
    GOALKEEPER = "goalkeeper"
    DEFENDER = "defender" 
    ATTACKER = "attacker"


class GameState(Enum):
    """Define game states for tactical decisions"""
    ATTACK = "attack"           # Our team has ball control
    DEFENSE = "defense"         # Opponent team has ball control  
    CONTEST = "contest"         # Loose ball, neither team in control


class OpponentAI:
    """
    Rule-based AI opponent that uses the same player interface as human control.
    
    **Design Philosophy**:
    - Use player.apply_actions() method for all control (same as keyboard)
    - Realistic soccer behavior patterns
    - Position-based roles and responsibilities
    - Adaptive tactics based on game state
    
    **Team Structure** (5 players):
    - Player 0: Goalkeeper (stays in goal area)
    - Player 1-2: Defenders (maintain defensive line)
    - Player 3-4: Attackers (press forward, attempt goals)
    """
    
    def __init__(self, team_side: str = "left", field_width: int = 800, field_height: int = 600):
        """
        Initialize the opponent AI system.
        
        Args:
            team_side: "left" or "right" - determines goal direction and positioning
            field_width: Width of the soccer field in pixels
            field_height: Height of the soccer field in pixels
        """
        self.team_side = team_side
        self.field_width = field_width
        self.field_height = field_height
        
        # Define team-specific parameters
        if team_side == "left":
            self.own_goal_x = 50          # Left goal x-coordinate
            self.opponent_goal_x = 750    # Right goal x-coordinate
            self.attack_direction = 1     # Attack toward positive x
            self.defensive_line_x = 200   # Defensive line position
        else:
            self.own_goal_x = 750         # Right goal x-coordinate
            self.opponent_goal_x = 50     # Left goal x-coordinate
            self.attack_direction = -1    # Attack toward negative x
            self.defensive_line_x = 600   # Defensive line position
        
        # Goal area definitions
        self.goal_center_y = field_height // 2
        self.goal_area_width = 100
        self.goal_area_height = 150
        
        # Tactical parameters
        self.ball_control_distance = 30.0    # Distance considered "ball control"
        self.pass_distance_min = 50.0        # Minimum passing distance
        self.pass_distance_max = 200.0       # Maximum passing distance
        self.shot_distance_threshold = 150.0  # Distance to attempt shots
        self.pressure_distance = 100.0       # Distance to pressure opponents
        
        # Cooldown timers (to prevent spam actions)
        self.player_action_cooldowns = [0.0] * 5  # Last action time for each player
        self.action_cooldown_time = 0.5  # Seconds between actions
        
    def control_team(self, team_players: List, ball, opponent_players: List):
        """
        Main control method - analyzes game state and controls all team players.
        Uses the same player.apply_actions() interface as human keyboard control.
        """
        if not team_players or len(team_players) != 5:
            return
            
        current_time = time.time()
        
        # Determine game state
        game_state = self._analyze_game_state(team_players, opponent_players, ball)
        ball_controller = self._find_ball_controller(team_players + opponent_players, ball)
        
        # Assign roles to players based on their position index
        roles = [
            PlayerRole.GOALKEEPER,  # Player 0
            PlayerRole.DEFENDER,    # Player 1
            PlayerRole.DEFENDER,    # Player 2
            PlayerRole.ATTACKER,    # Player 3
            PlayerRole.ATTACKER     # Player 4
        ]
        
        # Control each player based on their role and game state
        for i, player in enumerate(team_players):
            role = roles[i]
            # Get teammate positions for passing (excluding this player)
            teammate_positions = [p.position() for j, p in enumerate(team_players) if j != i]
            
            # Determine actions for this player
            actions = self._get_player_actions(
                player, role, i, game_state, ball_controller, 
                ball, team_players, opponent_players, current_time
            )
            
            # Apply actions using the same interface as human control
            player.apply_actions(
                actions['move_up'], actions['move_down'], 
                actions['move_left'], actions['move_right'],
                actions['shoot'], actions['pass'], 
                teammate_positions
            )
    
    def _analyze_game_state(self, team_players: List, opponent_players: List, ball) -> GameState:
        """Analyze current game situation to determine tactical approach."""
        # Check ball control for both teams
        team_ball_control = self._team_has_ball_control(team_players, ball)
        opponent_ball_control = self._team_has_ball_control(opponent_players, ball)
        
        if team_ball_control:
            return GameState.ATTACK
        elif opponent_ball_control:
            return GameState.DEFENSE
        else:
            return GameState.CONTEST
    
    def _team_has_ball_control(self, players: List, ball) -> bool:
        """Check if any player in the team has ball control"""
        ball_pos = ball.ball_body.position
        
        for player in players:
            player_pos = player.player_body.position
            distance = (ball_pos - player_pos).length
            if distance < self.ball_control_distance:
                return True
        return False
    
    def _find_ball_controller(self, all_players: List, ball) -> Optional:
        """Find which player currently has ball control, if any"""
        ball_pos = ball.ball_body.position
        
        for player in all_players:
            player_pos = player.player_body.position
            distance = (ball_pos - player_pos).length
            if distance < self.ball_control_distance:
                return player
        return None
    
    def _get_player_actions(self, player, role: PlayerRole, player_index: int, 
                           game_state: GameState, ball_controller, ball, 
                           team_players: List, opponent_players: List, current_time: float) -> dict:
        """
        Determine actions for individual player based on their assigned role and current game state.
        
        Returns:
            dict: Actions with keys: move_up, move_down, move_left, move_right, shoot, pass
        """
        # Initialize all actions as False
        actions = {
            'move_up': False,
            'move_down': False, 
            'move_left': False,
            'move_right': False,
            'shoot': False,
            'pass': False
        }
        
        ball_pos = ball.ball_body.position
        player_pos = player.player_body.position
        ball_distance = (ball_pos - player_pos).length
        
        if role == PlayerRole.GOALKEEPER:
            return self._get_goalkeeper_actions(player, ball, actions)
        elif role == PlayerRole.DEFENDER:
            return self._get_defender_actions(player, player_index, ball, ball_controller, 
                                            game_state, team_players, actions, current_time)
        elif role == PlayerRole.ATTACKER:
            return self._get_attacker_actions(player, player_index, ball, ball_controller,
                                            game_state, team_players, actions, current_time)
        
        return actions
    
    def _get_goalkeeper_actions(self, goalkeeper, ball, actions: dict) -> dict:
        """Goalkeeper AI - stay in goal area and position between ball and goal"""
        ball_pos = ball.ball_body.position
        keeper_pos = goalkeeper.player_body.position
        
        # Define goal area boundaries
        goal_area_left = self.own_goal_x - 30
        goal_area_right = self.own_goal_x + 30
        goal_area_top = self.goal_center_y - self.goal_area_height // 2
        goal_area_bottom = self.goal_center_y + self.goal_area_height // 2
        
        # Calculate ideal goalkeeper position (between ball and goal center)
        target_x = max(goal_area_left, min(goal_area_right, self.own_goal_x))
        ball_to_goal_y = ball_pos.y - self.goal_center_y
        target_y = self.goal_center_y + (ball_to_goal_y * 0.3)  # 30% toward ball
        target_y = max(goal_area_top, min(goal_area_bottom, target_y))
        
        # If ball is very close to goal, move toward ball
        ball_distance_to_goal = abs(ball_pos.x - self.own_goal_x)
        if ball_distance_to_goal < 100:
            ball_in_goal_area = (goal_area_left <= ball_pos.x <= goal_area_right and 
                               goal_area_top <= ball_pos.y <= goal_area_bottom)
            if ball_in_goal_area:
                target_x = ball_pos.x
                target_y = ball_pos.y
        
        # Determine movement actions
        pos_diff_x = target_x - keeper_pos.x
        pos_diff_y = target_y - keeper_pos.y
        
        if abs(pos_diff_x) > 5:
            if pos_diff_x > 0:
                actions['move_right'] = True
            else:
                actions['move_left'] = True
        
        if abs(pos_diff_y) > 5:
            if pos_diff_y > 0:
                actions['move_down'] = True
            else:
                actions['move_up'] = True
        
        # Clear ball if very close
        ball_distance = (ball_pos - keeper_pos).length
        if ball_distance < self.ball_control_distance:
            actions['shoot'] = True  # Use shoot to clear ball away
        
        return actions
    
    def _get_defender_actions(self, defender, player_index: int, ball, ball_controller,
                             game_state: GameState, team_players: List, actions: dict, 
                             current_time: float) -> dict:
        """Defender AI - maintain defensive line and support team tactics"""
        ball_pos = ball.ball_body.position
        defender_pos = defender.player_body.position
        ball_distance = (ball_pos - defender_pos).length
        
        if game_state == GameState.ATTACK and ball_controller in team_players:
            if ball_controller == defender:
                # Defender has ball - move forward to create attack
                target_x = defender_pos.x + (80 * self.attack_direction)  # Move forward with ball
                target_y = self.goal_center_y  # Move toward center of field
            else:
                # Teammate has ball - maintain defensive shape but support attack
                target_x = self.defensive_line_x + (50 * self.attack_direction)
                target_y = ball_pos.y + ((defender_pos.y - ball_pos.y) * 0.5)
            
        elif game_state == GameState.DEFENSE and ball_controller:
            # Opponent has ball - pressure or maintain position
            if ball_distance < self.pressure_distance:
                # Close enough to pressure - move toward ball carrier
                target_x = ball_controller.player_body.position.x
                target_y = ball_controller.player_body.position.y
            else:
                # Too far - maintain defensive line
                target_x = self.defensive_line_x
                target_y = ball_pos.y
                
        else:
            # Loose ball - move toward it
            target_x = ball_pos.x
            target_y = ball_pos.y
        
        # Apply movement toward target
        pos_diff_x = target_x - defender_pos.x
        pos_diff_y = target_y - defender_pos.y
        
        if abs(pos_diff_x) > 10:
            if pos_diff_x > 0:
                actions['move_right'] = True
            else:
                actions['move_left'] = True
        
        if abs(pos_diff_y) > 10:
            if pos_diff_y > 0:
                actions['move_down'] = True
            else:
                actions['move_up'] = True
        
        # Pass to attackers if has ball control - more aggressive passing
        if (ball_controller == defender and 
            self._can_take_action(current_time, player_index)):
            
            # Try to find any good passing target
            best_pass_target = None
            best_pass_score = -1
            
            # Check attackers first (preferred targets)
            attackers = team_players[3:5]  # Players 3-4 are attackers
            for attacker in attackers:
                attacker_pos = attacker.player_body.position
                pass_distance = (attacker_pos - ball_pos).length
                
                if (self.pass_distance_min <= pass_distance <= self.pass_distance_max):
                    # Check if attacker is forward of defender
                    forward_progress = (attacker_pos.x - defender_pos.x) * self.attack_direction
                    if forward_progress > 0:  # Any forward progress is good
                        # Score based on how forward they are
                        pass_score = forward_progress / pass_distance  # Forward progress per distance
                        if pass_score > best_pass_score:
                            best_pass_score = pass_score
                            best_pass_target = attacker
            
            # If found good pass, execute it
            if best_pass_target is not None:
                actions['pass'] = True
                # Don't move when passing
                actions['move_up'] = False
                actions['move_down'] = False
                actions['move_left'] = False
                actions['move_right'] = False
        
        return actions
    
    def _get_attacker_actions(self, attacker, player_index: int, ball, ball_controller,
                             game_state: GameState, team_players: List, actions: dict,
                             current_time: float) -> dict:
        """Attacker AI - focus on scoring opportunities and forward play"""
        ball_pos = ball.ball_body.position
        attacker_pos = attacker.player_body.position
        ball_distance = (ball_pos - attacker_pos).length
        
        # Default target position (will be overridden based on game state)
        target_x = ball_pos.x
        target_y = ball_pos.y
        
        if game_state == GameState.ATTACK and ball_controller in team_players:
            if ball_controller == attacker:
                # Attacker has ball - decide to shoot or dribble toward goal
                goal_distance = abs(ball_pos.x - self.opponent_goal_x)
                goal_y_distance = abs(ball_pos.y - self.goal_center_y)
                
                # Shoot if in good position
                if (goal_distance < self.shot_distance_threshold and 
                    goal_y_distance < 100 and 
                    self._can_take_action(current_time, player_index)):
                    actions['shoot'] = True
                    # Don't move when shooting
                    return actions
                else:
                    # Dribble toward goal
                    target_x = self.opponent_goal_x
                    target_y = self.goal_center_y
                    
            else:
                # Teammate has ball - make attacking run
                run_target_x = self.opponent_goal_x + (100 * -self.attack_direction)
                # Spread out vertically
                if attacker == team_players[3]:  # First attacker
                    run_target_y = self.goal_center_y - 50
                else:  # Second attacker  
                    run_target_y = self.goal_center_y + 50
                target_x = run_target_x
                target_y = run_target_y
                
        elif game_state == GameState.DEFENSE and ball_controller:
            # Opponent has ball - press the ball carrier
            target_x = ball_controller.player_body.position.x
            target_y = ball_controller.player_body.position.y
            
        else:
            # Loose ball - compete aggressively
            target_x = ball_pos.x
            target_y = ball_pos.y
        
        # Apply movement toward target
        pos_diff_x = target_x - attacker_pos.x
        pos_diff_y = target_y - attacker_pos.y
        
        if abs(pos_diff_x) > 10:
            if pos_diff_x > 0:
                actions['move_right'] = True
            else:
                actions['move_left'] = True
        
        if abs(pos_diff_y) > 10:
            if pos_diff_y > 0:
                actions['move_down'] = True
            else:
                actions['move_up'] = True
        
        return actions
    
    def _can_take_action(self, current_time: float, player_index: int) -> bool:
        """Check if player can take action (not in cooldown)"""
        return current_time - self.player_action_cooldowns[player_index] > self.action_cooldown_time
    
    def get_team_formation_info(self) -> dict:
        """
        Get information about team formation and tactical setup.
        
        Returns:
            dict: Formation information for debugging/visualization
        """
        return {
            "team_side": self.team_side,
            "own_goal": self.own_goal_x,
            "opponent_goal": self.opponent_goal_x,
            "defensive_line": self.defensive_line_x,
            "attack_direction": self.attack_direction,
            "player_roles": [
                "Goalkeeper",
                "Defender", 
                "Defender",
                "Attacker",
                "Attacker"
            ]
        }