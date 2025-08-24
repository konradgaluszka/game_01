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
        
        # Enhanced ball stealing parameters - Very aggressive settings
        self.coordinated_press_distance = 250.0  # Distance to trigger coordinated pressing (very wide)
        self.support_distance = 150.0            # Distance for support positioning
        self.intercept_distance = 180.0          # Distance to position for interceptions
        self.press_commitment_threshold = 1      # Min players to commit to pressing
        self.max_press_participants = 3         # Maximum players in coordinated press
        
        # Cooldown timers (to prevent spam actions)
        self.player_action_cooldowns = [0.0] * 5  # Last action time for each player
        self.action_cooldown_time = 0.5  # Seconds between actions
        
        # Ball stealing state tracking
        self.ball_steal_mode = False            # Whether team is in coordinated steal mode
        self.ball_steal_target = None           # Current ball carrier being pressed
        self.ball_steal_participants = []       # Players participating in ball steal
        self.last_ball_steal_time = 0.0        # Time of last ball steal attempt
        
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
        
        # Update ball stealing coordination
        self._update_ball_stealing_coordination(team_players, opponent_players, ball, ball_controller, current_time)
        
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
        
        # Check if should participate in coordinated ball stealing
        if self._should_participate_in_ball_steal(player_index, PlayerRole.DEFENDER):
            return self._get_coordinated_steal_actions(defender, player_index, ball_controller, 
                                                     ball, team_players, actions)
        
        # If we have ball control and just stole it, use enhanced attack transition
        if (ball_controller == defender and game_state == GameState.ATTACK and 
            current_time - self.last_ball_steal_time < 2.0):  # Within 2 seconds of steal
            return self._get_post_steal_attack_actions(defender, PlayerRole.DEFENDER, ball, actions)
        
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
            # Opponent has ball - MORE AGGRESSIVE pressure
            ball_controller_pos = ball_controller.player_body.position
            
            # More aggressive pressure distance and behavior
            if ball_distance < self.pressure_distance * 1.5:  # Increased pressure range
                # Move aggressively toward ball carrier
                target_x = ball_controller_pos.x
                target_y = ball_controller_pos.y
            elif ball_distance < self.coordinated_press_distance:
                # Even if not close, move to support the press
                # Position to cut off passing lanes or support nearest teammate
                target_x = ball_controller_pos.x + (defender_pos.x - ball_controller_pos.x) * 0.5
                target_y = ball_controller_pos.y + (defender_pos.y - ball_controller_pos.y) * 0.5
            else:
                # Too far - but still move toward ball instead of just maintaining line
                target_x = ball_pos.x + (self.defensive_line_x - ball_pos.x) * 0.3  # Compromise between ball and defensive line
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
        
        # Check if should participate in coordinated ball stealing
        if self._should_participate_in_ball_steal(player_index, PlayerRole.ATTACKER):
            return self._get_coordinated_steal_actions(attacker, player_index, ball_controller, 
                                                     ball, team_players, actions)
        
        # If we have ball control and just stole it, use enhanced attack transition
        if (ball_controller == attacker and game_state == GameState.ATTACK and 
            current_time - self.last_ball_steal_time < 2.0):  # Within 2 seconds of steal
            return self._get_post_steal_attack_actions(attacker, PlayerRole.ATTACKER, ball, actions)
        
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
            # Opponent has ball - AGGRESSIVE press and support
            ball_controller_pos = ball_controller.player_body.position
            
            if ball_distance < self.pressure_distance * 1.8:  # Very aggressive for attackers
                # Direct pressure on ball carrier
                target_x = ball_controller_pos.x
                target_y = ball_controller_pos.y
            elif ball_distance < self.coordinated_press_distance:
                # Support pressing by cutting off retreat routes
                # Position between ball carrier and their goal
                retreat_direction_x = self.own_goal_x - ball_controller_pos.x
                retreat_direction_y = self.goal_center_y - ball_controller_pos.y
                
                # Normalize direction
                retreat_length = (retreat_direction_x**2 + retreat_direction_y**2)**0.5
                if retreat_length > 0:
                    retreat_direction_x /= retreat_length
                    retreat_direction_y /= retreat_length
                
                # Position to cut off retreat
                target_x = ball_controller_pos.x + retreat_direction_x * 60
                target_y = ball_controller_pos.y + retreat_direction_y * 60
            else:
                # Far away - still move toward ball to provide support
                target_x = ball_pos.x
                target_y = ball_pos.y
            
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
    
    def _update_ball_stealing_coordination(self, team_players: List, opponent_players: List, 
                                         ball, ball_controller, current_time: float):
        """
        Update coordinated ball stealing behavior - multiple players work together to steal ball.
        """
        # Check if opponent has ball control
        if ball_controller not in opponent_players:
            self.ball_steal_mode = False
            self.ball_steal_target = None
            self.ball_steal_participants = []
            return
        
        ball_pos = ball.ball_body.position
        
        # Always find nearby players (more aggressive approach)
        nearby_players = []
        for i, player in enumerate(team_players):
            if i == 0:  # Skip goalkeeper
                continue
            player_pos = player.player_body.position
            distance_to_ball = (ball_pos - player_pos).length
            if distance_to_ball < self.coordinated_press_distance:
                nearby_players.append((player, i, distance_to_ball))
        
        # Sort by distance to ball (closest first)
        nearby_players.sort(key=lambda x: x[2])
        
        # Always activate steal mode if we have ANY nearby players
        if len(nearby_players) >= self.press_commitment_threshold:
            if not self.ball_steal_mode:
                self.ball_steal_mode = True
                self.ball_steal_target = ball_controller
                self.last_ball_steal_time = current_time
            
            # Limit participants to avoid all players clustering
            selected_participants = nearby_players[:self.max_press_participants]
            self.ball_steal_participants = [p[1] for p in selected_participants]
            
        else:
            self.ball_steal_mode = False
            self.ball_steal_target = None
            self.ball_steal_participants = []
            
        # Also exit steal mode if ball carrier has changed or too much time passed
        if (self.ball_steal_target != ball_controller or 
            current_time - self.last_ball_steal_time > 8.0):  # Increased timeout to 8 seconds
            self.ball_steal_mode = False
            self.ball_steal_target = None
            self.ball_steal_participants = []
    
    def _should_participate_in_ball_steal(self, player_index: int, role: PlayerRole) -> bool:
        """
        Determine if a player should participate in coordinated ball stealing.
        """
        # Goalkeeper never participates in field pressing
        if role == PlayerRole.GOALKEEPER:
            return False
        
        # If not in steal mode, use normal behavior
        if not self.ball_steal_mode:
            return False
        
        # Check if this player is designated as a steal participant
        return player_index in self.ball_steal_participants
    
    def _get_coordinated_steal_actions(self, player, player_index: int, ball_controller, 
                                      ball, team_players: List, actions: dict) -> dict:
        """
        Get actions for coordinated ball stealing - multiple players converge and support.
        """
        ball_pos = ball.ball_body.position
        player_pos = player.player_body.position
        controller_pos = ball_controller.player_body.position
        
        # Determine steal strategy based on player index
        steal_participants_count = len(self.ball_steal_participants)
        participant_rank = self.ball_steal_participants.index(player_index)
        
        if participant_rank == 0:  # Primary presser - go directly to ball carrier
            target_x = controller_pos.x
            target_y = controller_pos.y
            
        elif participant_rank == 1 and steal_participants_count > 1:  # Secondary presser - cut off escape route
            # Position to cut off the most likely escape direction
            escape_vector_x = controller_pos.x - ball_pos.x if abs(controller_pos.x - ball_pos.x) > 5 else self.attack_direction * 50
            escape_vector_y = controller_pos.y - ball_pos.y if abs(controller_pos.y - ball_pos.y) > 5 else 0
            
            # Position ahead of ball carrier in their movement direction
            target_x = controller_pos.x + escape_vector_x * 0.8
            target_y = controller_pos.y + escape_vector_y * 0.8
            
        else:  # Support players - position for interceptions and second balls
            # Position to intercept likely pass directions
            goal_direction_x = self.opponent_goal_x - controller_pos.x
            goal_direction_y = self.goal_center_y - controller_pos.y
            
            # Normalize and scale the direction vector
            direction_length = (goal_direction_x**2 + goal_direction_y**2)**0.5
            if direction_length > 0:
                goal_direction_x = (goal_direction_x / direction_length) * self.intercept_distance
                goal_direction_y = (goal_direction_y / direction_length) * self.intercept_distance
            
            # Position between ball carrier and their goal
            target_x = controller_pos.x + goal_direction_x * 0.6
            target_y = controller_pos.y + goal_direction_y * 0.6
        
        # Apply movement toward target
        pos_diff_x = target_x - player_pos.x
        pos_diff_y = target_y - player_pos.y
        
        # More aggressive movement in steal mode
        movement_threshold = 8.0  # Lower threshold for more responsive movement
        
        if abs(pos_diff_x) > movement_threshold:
            if pos_diff_x > 0:
                actions['move_right'] = True
            else:
                actions['move_left'] = True
        
        if abs(pos_diff_y) > movement_threshold:
            if pos_diff_y > 0:
                actions['move_down'] = True
            else:
                actions['move_up'] = True
        
        # Attempt to steal ball if close enough
        distance_to_controller = (controller_pos - player_pos).length
        if distance_to_controller < self.ball_control_distance + 10:  # Slightly larger range for stealing
            actions['shoot'] = True  # Use shoot action to challenge for ball
        
        return actions
    
    def _get_post_steal_attack_actions(self, player, role: PlayerRole, ball, actions: dict) -> dict:
        """
        Enhanced attack actions after successful ball steal - quick transition to goal attempts.
        """
        ball_pos = ball.ball_body.position
        player_pos = player.player_body.position
        
        # Quick decision making for immediate attack
        goal_distance = abs(ball_pos.x - self.opponent_goal_x)
        
        if role == PlayerRole.ATTACKER:
            # Attackers should immediately push toward goal
            if goal_distance < self.shot_distance_threshold:
                # In shooting range - take quick shot
                actions['shoot'] = True
                return actions
            else:
                # Dribble quickly toward goal
                target_x = self.opponent_goal_x
                target_y = self.goal_center_y
        
        elif role == PlayerRole.DEFENDER:
            # Defenders should look for quick forward pass or dribble
            # Find the most advanced attacker
            attackers = [p for i, p in enumerate(self.ball_steal_participants) if i >= 3]  # Get attacker indices
            if attackers:
                # Quick pass to forward player
                actions['pass'] = True
                return actions
            else:
                # No good pass - dribble forward quickly
                target_x = ball_pos.x + (self.attack_direction * 60)
                target_y = self.goal_center_y
        
        # Apply quick movement
        pos_diff_x = target_x - player_pos.x
        pos_diff_y = target_y - player_pos.y
        
        if abs(pos_diff_x) > 8:
            if pos_diff_x > 0:
                actions['move_right'] = True
            else:
                actions['move_left'] = True
        
        if abs(pos_diff_y) > 8:
            if pos_diff_y > 0:
                actions['move_down'] = True
            else:
                actions['move_up'] = True
        
        return actions
    
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