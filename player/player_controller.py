"""
Player Input Control System

This module handles input translation and action coordination for players.
It provides a clean interface between different input sources (keyboard, AI)
and the player's action execution system.

**Responsibility**: Input handling and action coordination
**Dependencies**: pygame (for keyboard), GameConfig
"""

import pygame
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod

from common.Vector import Vector
from config.game_config import GameConfig


class ActionType(Enum):
    """Enumeration of possible player actions"""
    MOVE_UP = "move_up"
    MOVE_DOWN = "move_down"
    MOVE_LEFT = "move_left"
    MOVE_RIGHT = "move_right"
    SHOOT = "shoot"
    PASS = "pass"
    NO_ACTION = "no_action"


@dataclass
class PlayerAction:
    """
    Represents a single player action with context.
    
    **Purpose**: Encapsulate action information for clean processing
    """
    action_type: ActionType
    strength: float = 1.0  # Action strength (0.0 to 1.0)
    direction: Optional[Vector] = None  # For directional actions
    target_position: Optional[Vector] = None  # For targeted actions like passing


class InputSource(ABC):
    """Abstract base class for input sources"""
    
    @abstractmethod
    def get_actions(self, context: dict) -> List[PlayerAction]:
        """Get list of actions from this input source"""
        pass


class KeyboardInputSource(InputSource):
    """
    Handles keyboard input for player control.
    
    **Single Responsibility**: Translate keyboard state to player actions
    """
    
    def __init__(self, config: GameConfig):
        """
        Initialize keyboard input handler.
        
        Args:
            config: Game configuration
        """
        self.config = config
        
        # Key mappings (can be customized)
        self.key_mappings = {
            pygame.K_UP: ActionType.MOVE_UP,
            pygame.K_DOWN: ActionType.MOVE_DOWN,
            pygame.K_LEFT: ActionType.MOVE_LEFT,
            pygame.K_RIGHT: ActionType.MOVE_RIGHT,
            pygame.K_d: ActionType.SHOOT,
            pygame.K_s: ActionType.PASS,
        }
    
    def get_actions(self, context: dict) -> List[PlayerAction]:
        """
        Convert current keyboard state to player actions.
        
        Args:
            context: Dictionary containing 'keys' (pygame key state)
            
        Returns:
            List of PlayerAction objects
        """
        keys = context.get('keys', {})
        actions = []
        
        for key, action_type in self.key_mappings.items():
            if keys.get(key, False):
                actions.append(PlayerAction(action_type=action_type))
        
        return actions
    
    def customize_key_mapping(self, key: int, action_type: ActionType) -> None:
        """Customize key mapping for this input source"""
        self.key_mappings[key] = action_type


class AIInputSource(InputSource):
    """
    Handles AI input for player control.
    
    **Single Responsibility**: Translate AI decisions to player actions
    """
    
    def __init__(self, config: GameConfig):
        """
        Initialize AI input handler.
        
        Args:
            config: Game configuration
        """
        self.config = config
    
    def get_actions(self, context: dict) -> List[PlayerAction]:
        """
        Convert AI action flags to player actions.
        
        Args:
            context: Dictionary containing AI action flags
            
        Returns:
            List of PlayerAction objects
        """
        actions = []
        
        # Extract AI action flags from context
        move_up = context.get('move_up', False)
        move_down = context.get('move_down', False)
        move_left = context.get('move_left', False)
        move_right = context.get('move_right', False)
        shoot = context.get('shoot', False)
        pass_ball = context.get('pass', False)
        
        # Convert flags to actions
        if move_up:
            actions.append(PlayerAction(action_type=ActionType.MOVE_UP))
        if move_down:
            actions.append(PlayerAction(action_type=ActionType.MOVE_DOWN))
        if move_left:
            actions.append(PlayerAction(action_type=ActionType.MOVE_LEFT))
        if move_right:
            actions.append(PlayerAction(action_type=ActionType.MOVE_RIGHT))
        if shoot:
            actions.append(PlayerAction(action_type=ActionType.SHOOT))
        if pass_ball:
            actions.append(PlayerAction(action_type=ActionType.PASS))
        
        return actions


class PlayerController:
    """
    Manages input handling and action coordination for a player.
    
    **Single Responsibility**: Coordinate between input sources and action execution
    
    **Key Features**:
    - Supports multiple input sources (keyboard, AI)
    - Filters and validates actions
    - Provides action priority and conflict resolution
    - Clean interface for different control modes
    
    **Usage**:
    ```python
    controller = PlayerController(config)
    
    # For keyboard control
    actions = controller.get_keyboard_actions(keys, teammates)
    
    # For AI control  
    actions = controller.get_ai_actions(ai_decisions, teammates)
    
    # Execute actions
    controller.execute_actions(actions, player_systems)
    ```
    """
    
    def __init__(self, config: GameConfig):
        """
        Initialize player controller.
        
        Args:
            config: Game configuration
        """
        self.config = config
        
        # Input sources
        self.keyboard_input = KeyboardInputSource(config)
        self.ai_input = AIInputSource(config)
        
        # Action filtering and priority
        self._action_filters = []
        self._action_priority = {
            ActionType.SHOOT: 5,
            ActionType.PASS: 4,
            ActionType.MOVE_UP: 1,
            ActionType.MOVE_DOWN: 1,
            ActionType.MOVE_LEFT: 1,
            ActionType.MOVE_RIGHT: 1,
        }
    
    def get_keyboard_actions(self, keys: Dict, teammates_positions: List[Vector]) -> List[PlayerAction]:
        """
        Get actions from keyboard input.
        
        Args:
            keys: pygame key state dictionary
            teammates_positions: List of teammate positions for context
            
        Returns:
            Filtered and prioritized list of actions
        """
        context = {
            'keys': keys,
            'teammates_positions': teammates_positions
        }
        
        raw_actions = self.keyboard_input.get_actions(context)
        return self._process_actions(raw_actions, context)
    
    def get_ai_actions(self, move_up: bool, move_down: bool, move_left: bool, 
                      move_right: bool, shoot: bool, pass_ball: bool,
                      teammates_positions: List[Vector]) -> List[PlayerAction]:
        """
        Get actions from AI input.
        
        Args:
            move_up, move_down, move_left, move_right: Movement flags
            shoot: Shoot action flag
            pass_ball: Pass action flag
            teammates_positions: List of teammate positions for context
            
        Returns:
            Filtered and prioritized list of actions
        """
        context = {
            'move_up': move_up,
            'move_down': move_down,
            'move_left': move_left,
            'move_right': move_right,
            'shoot': shoot,
            'pass': pass_ball,
            'teammates_positions': teammates_positions
        }
        
        raw_actions = self.ai_input.get_actions(context)
        return self._process_actions(raw_actions, context)
    
    def _process_actions(self, raw_actions: List[PlayerAction], context: dict) -> List[PlayerAction]:
        """
        Process and filter raw actions.
        
        Args:
            raw_actions: Raw actions from input source
            context: Action context information
            
        Returns:
            Processed and validated actions
        """
        # Apply filters
        filtered_actions = raw_actions
        for filter_func in self._action_filters:
            filtered_actions = filter_func(filtered_actions, context)
        
        # Resolve conflicts and apply priority
        final_actions = self._resolve_action_conflicts(filtered_actions)
        
        return final_actions
    
    def _resolve_action_conflicts(self, actions: List[PlayerAction]) -> List[PlayerAction]:
        """
        Resolve conflicting actions using priority system.
        
        Args:
            actions: List of potentially conflicting actions
            
        Returns:
            List of non-conflicting actions
        """
        if not actions:
            return []
        
        # Group actions by type
        action_groups = {}
        for action in actions:
            action_type = action.action_type
            if action_type not in action_groups:
                action_groups[action_type] = []
            action_groups[action_type].append(action)
        
        # Apply conflict resolution rules
        resolved_actions = []
        
        # Rule 1: Only one ball action (shoot or pass) at a time
        ball_actions = []
        if ActionType.SHOOT in action_groups:
            ball_actions.extend(action_groups[ActionType.SHOOT])
        if ActionType.PASS in action_groups:
            ball_actions.extend(action_groups[ActionType.PASS])
        
        if ball_actions:
            # Choose highest priority ball action
            best_ball_action = max(ball_actions, 
                                 key=lambda a: self._action_priority.get(a.action_type, 0))
            resolved_actions.append(best_ball_action)
        
        # Rule 2: Allow all movement actions (they can combine)
        movement_actions = [ActionType.MOVE_UP, ActionType.MOVE_DOWN, 
                          ActionType.MOVE_LEFT, ActionType.MOVE_RIGHT]
        
        for movement_type in movement_actions:
            if movement_type in action_groups:
                # Take first action of this movement type
                resolved_actions.append(action_groups[movement_type][0])
        
        return resolved_actions
    
    def add_action_filter(self, filter_func: Callable) -> None:
        """
        Add custom action filter function.
        
        Args:
            filter_func: Function that takes (actions, context) and returns filtered actions
        """
        self._action_filters.append(filter_func)
    
    def set_action_priority(self, action_type: ActionType, priority: int) -> None:
        """
        Set priority for an action type.
        
        Args:
            action_type: Type of action
            priority: Priority value (higher = more important)
        """
        self._action_priority[action_type] = priority
    
    def customize_keyboard_mapping(self, key: int, action_type: ActionType) -> None:
        """Customize keyboard key mapping"""
        self.keyboard_input.customize_key_mapping(key, action_type)


# Action filter functions that can be applied to PlayerController

def cooldown_filter(actions: List[PlayerAction], context: dict) -> List[PlayerAction]:
    """
    Filter actions based on cooldown periods.
    
    Args:
        actions: Input actions
        context: Must contain 'last_action_time' and 'current_time'
        
    Returns:
        Actions that are not on cooldown
    """
    last_action_time = context.get('last_action_time', 0)
    current_time = context.get('current_time', 0)
    cooldown_period = context.get('cooldown_period', 1.0)
    
    if current_time - last_action_time < cooldown_period:
        # Filter out ball actions during cooldown
        return [action for action in actions 
                if action.action_type not in [ActionType.SHOOT, ActionType.PASS]]
    
    return actions


def ball_proximity_filter(actions: List[PlayerAction], context: dict) -> List[PlayerAction]:
    """
    Filter ball actions based on ball proximity.
    
    Args:
        actions: Input actions
        context: Must contain 'player_position', 'ball_position', 'config'
        
    Returns:
        Actions valid for current ball proximity
    """
    player_pos = context.get('player_position')
    ball_pos = context.get('ball_position')
    config = context.get('config')
    
    if not all([player_pos, ball_pos, config]):
        return actions
    
    distance = (ball_pos - player_pos).length()
    max_distance = config.ball_control.DRIBBLE_DISTANCE
    
    if distance > max_distance:
        # Too far from ball - remove ball actions
        return [action for action in actions 
                if action.action_type not in [ActionType.SHOOT, ActionType.PASS]]
    
    return actions