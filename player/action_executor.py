"""
Player Action Execution System

This module handles the execution of player actions, including:
- Movement force application
- Shooting mechanics
- Passing mechanics with target selection
- Action validation and timing

**Responsibility**: Execute validated player actions on game systems
**Dependencies**: Physics system, Ball controller, GameConfig
"""

import time
import math
from typing import List, Optional, Tuple
from abc import ABC, abstractmethod

from common.Vector import Vector
from config.game_config import GameConfig
from player.player_controller import PlayerAction, ActionType


class ActionResult:
    """
    Represents the result of an action execution.
    
    **Purpose**: Provide feedback about action success/failure
    """
    
    def __init__(self, success: bool, message: str = "", data: dict = None):
        self.success = success
        self.message = message
        self.data = data or {}
    
    @classmethod
    def success(cls, message: str = "", data: dict = None) -> 'ActionResult':
        """Create successful action result"""
        return cls(True, message, data)
    
    @classmethod
    def failure(cls, message: str, data: dict = None) -> 'ActionResult':
        """Create failed action result"""
        return cls(False, message, data)


class ActionHandler(ABC):
    """Abstract base class for action handlers"""
    
    @abstractmethod
    def can_handle(self, action: PlayerAction) -> bool:
        """Check if this handler can execute the given action"""
        pass
    
    @abstractmethod
    def execute(self, action: PlayerAction, context: dict) -> ActionResult:
        """Execute the action with given context"""
        pass


class MovementActionHandler(ActionHandler):
    """
    Handles player movement actions.
    
    **Single Responsibility**: Convert movement actions to physics forces
    """
    
    def __init__(self, config: GameConfig):
        self.config = config
        
        # Movement direction mappings
        self.direction_vectors = {
            ActionType.MOVE_UP: Vector(0, -1),
            ActionType.MOVE_DOWN: Vector(0, 1),
            ActionType.MOVE_LEFT: Vector(-1, 0),
            ActionType.MOVE_RIGHT: Vector(1, 0),
        }
    
    def can_handle(self, action: PlayerAction) -> bool:
        """Check if this is a movement action"""
        return action.action_type in self.direction_vectors
    
    def execute(self, action: PlayerAction, context: dict) -> ActionResult:
        """
        Execute movement action by applying physics force.
        
        Args:
            action: Movement action to execute
            context: Must contain 'physics' (PlayerPhysics instance)
            
        Returns:
            ActionResult indicating success/failure
        """
        physics = context.get('physics')
        if not physics:
            return ActionResult.failure("No physics system provided")
        
        direction = self.direction_vectors.get(action.action_type)
        if not direction:
            return ActionResult.failure(f"Unknown movement action: {action.action_type}")
        
        # Calculate force strength
        base_force = self.config.physics.MOVEMENT_FORCE
        force_strength = base_force * action.strength
        
        # Apply movement force
        force_vector = direction * force_strength
        physics.apply_movement_force(force_vector.x, force_vector.y)
        
        return ActionResult.success(
            f"Applied {action.action_type.value} force",
            {'force': force_vector, 'strength': action.strength}
        )


class ShootActionHandler(ActionHandler):
    """
    Handles shooting actions.
    
    **Single Responsibility**: Execute shooting mechanics
    """
    
    def __init__(self, config: GameConfig):
        self.config = config
    
    def can_handle(self, action: PlayerAction) -> bool:
        """Check if this is a shoot action"""
        return action.action_type == ActionType.SHOOT
    
    def execute(self, action: PlayerAction, context: dict) -> ActionResult:
        """
        Execute shooting action.
        
        Args:
            action: Shoot action to execute
            context: Must contain 'ball_controller', 'ball', 'space', 'physics', 'current_time'
            
        Returns:
            ActionResult indicating success/failure
        """
        ball_controller = context.get('ball_controller')
        ball = context.get('ball')
        space = context.get('space')
        physics = context.get('physics')
        current_time = context.get('current_time', time.time())
        
        # Validate required context
        if not all([ball_controller, ball, space, physics]):
            return ActionResult.failure("Missing required context for shooting")
        
        # Check if player has ball control
        if not ball_controller.has_ball_control():
            return ActionResult.failure("Player must have ball control to shoot")
        
        # Calculate shot direction
        player_pos = physics.get_position()
        ball_pos = Vector(ball.ball_body.position.x, ball.ball_body.position.y)
        
        # Default shot direction (can be overridden by action.direction)
        if action.direction:
            shot_direction = action.direction.normalize()
        else:
            # Shoot in direction from player to ball (simple approach)
            direction_vector = ball_pos - player_pos
            if direction_vector.length() > 0.1:
                shot_direction = direction_vector.normalize()
            else:
                # Ball at player position, shoot forward (based on recent movement)
                player_velocity = physics.get_velocity()
                if player_velocity.length() > 0.1:
                    shot_direction = player_velocity.normalize()
                else:
                    shot_direction = Vector(1, 0)  # Default right direction
        
        # Calculate shot power
        base_power = self.config.actions.SHOT_STRENGTH
        shot_power = base_power * action.strength
        
        # Release ball control and apply shooting impulse
        ball_controller.release_ball_control(space, current_time)
        
        # Apply impulse to ball
        impulse = shot_direction * shot_power
        ball.ball_body.apply_impulse_at_local_point((impulse.x, impulse.y))
        
        return ActionResult.success(
            "Shot executed",
            {
                'direction': shot_direction,
                'power': shot_power,
                'ball_position': ball_pos
            }
        )


class PassActionHandler(ActionHandler):
    """
    Handles passing actions with target selection.
    
    **Single Responsibility**: Execute passing mechanics with intelligent target selection
    """
    
    def __init__(self, config: GameConfig):
        self.config = config
    
    def can_handle(self, action: PlayerAction) -> bool:
        """Check if this is a pass action"""
        return action.action_type == ActionType.PASS
    
    def execute(self, action: PlayerAction, context: dict) -> ActionResult:
        """
        Execute passing action with target selection.
        
        Args:
            action: Pass action to execute
            context: Must contain 'ball_controller', 'ball', 'space', 'physics', 
                    'teammates_positions', 'current_time'
            
        Returns:
            ActionResult indicating success/failure
        """
        ball_controller = context.get('ball_controller')
        ball = context.get('ball')
        space = context.get('space')
        physics = context.get('physics')
        teammates_positions = context.get('teammates_positions', [])
        current_time = context.get('current_time', time.time())
        
        # Validate required context
        if not all([ball_controller, ball, space, physics]):
            return ActionResult.failure("Missing required context for passing")
        
        # Check if player has ball control
        if not ball_controller.has_ball_control():
            return ActionResult.failure("Player must have ball control to pass")
        
        # Find best pass target
        player_pos = physics.get_position()
        target_pos = self._find_best_pass_target(player_pos, teammates_positions, physics)
        
        if not target_pos:
            return ActionResult.failure("No valid pass target found")
        
        # Calculate pass direction and power (match original implementation)
        pass_direction_vector = target_pos - player_pos
        
        # Use original power calculation: direction_vector * MAX_PASS_STRENGTH / 100
        # This gives stronger, more consistent passes like the original
        base_power = self.config.actions.MAX_PASS_STRENGTH
        pass_power_scaling = base_power / 100.0  # Original used /100
        
        # Release ball control and apply passing impulse
        ball_controller.release_ball_control(space, current_time)
        
        # Apply impulse to ball using original method: direction_vector * power_scaling
        impulse_vector = pass_direction_vector * pass_power_scaling
        ball.ball_body.apply_impulse_at_local_point((impulse_vector.x, impulse_vector.y))
        
        return ActionResult.success(
            "Pass executed",
            {
                'target_position': target_pos,
                'direction': pass_direction_vector,
                'power': pass_power_scaling,
                'distance': pass_direction_vector.length()
            }
        )
    
    def _find_best_pass_target(self, player_pos: Vector, teammates_positions: List[Vector],
                             physics) -> Optional[Vector]:
        """
        Find the best teammate to pass to using original algorithm.
        
        Original method: Select teammate most aligned with player's movement direction,
        breaking ties by closest distance.
        
        Args:
            player_pos: Current player position
            teammates_positions: List of teammate positions
            physics: Player physics system for movement context
            
        Returns:
            Best target position or None if no good target found
        """
        if not teammates_positions:
            return None
        
        # Get player's movement direction (equivalent to original _last_velocity)
        player_velocity = physics.get_velocity()
        
        # If not moving, can't determine facing direction (same as original)
        if player_velocity.length() <= 0:
            return None
        
        # Normalize movement direction (equivalent to original f = _last_velocity.normalize())
        facing_direction = player_velocity.normalize()
        
        best_target = None
        best_dot = -float('inf')
        best_dist_squared = float('inf')
        
        for teammate_pos in teammates_positions:
            # Direction to teammate
            to_teammate = teammate_pos - player_pos
            
            # Skip if teammate is at same position
            if to_teammate.length_squared() == 0:
                continue
                
            # Normalize direction to teammate
            to_teammate_direction = to_teammate.normalize()
            
            # Calculate alignment with facing direction (dot product = cosine of angle)
            dot_product = facing_direction.dot(to_teammate_direction)
            
            # Distance squared for tie-breaking
            dist_squared = to_teammate.length_squared()
            
            # Select best: largest dot (smallest angle), break ties by nearest distance
            # This matches the original logic exactly
            if (dot_product > best_dot or 
                (abs(dot_product - best_dot) < 1e-9 and dist_squared < best_dist_squared)):
                best_target = teammate_pos
                best_dot = dot_product
                best_dist_squared = dist_squared
        
        return best_target
    
    def _calculate_pass_score(self, player_pos: Vector, teammate_pos: Vector,
                            player_velocity: Vector, distance: float) -> float:
        """
        Calculate pass score for a potential target.
        
        Higher score = better pass target
        """
        score = 0.0
        
        # Factor 1: Prefer moderate distances
        optimal_distance = 100.0
        distance_score = 1.0 - abs(distance - optimal_distance) / optimal_distance
        score += distance_score * 30.0
        
        # Factor 2: Prefer passes in player's facing direction
        if player_velocity.length() > 0.1:
            player_direction = player_velocity.normalize()
            pass_direction = (teammate_pos - player_pos).normalize()
            direction_alignment = player_direction.dot(pass_direction)
            score += direction_alignment * 40.0
        
        # Factor 3: Prefer forward passes (toward right side of field)
        forward_component = teammate_pos.x - player_pos.x
        score += forward_component * 0.1
        
        return score


class ActionExecutor:
    """
    Coordinates execution of player actions using specialized handlers.
    
    **Single Responsibility**: Coordinate and execute all player actions
    
    **Key Features**:
    - Delegates actions to specialized handlers
    - Validates actions before execution
    - Provides comprehensive action results
    - Supports adding custom action handlers
    
    **Usage**:
    ```python
    executor = ActionExecutor(config)
    
    context = {
        'physics': player_physics,
        'ball_controller': ball_controller,
        'ball': ball,
        'space': space,
        'teammates_positions': teammates,
        'current_time': time.time()
    }
    
    results = executor.execute_actions(actions, context)
    ```
    """
    
    def __init__(self, config: GameConfig):
        """
        Initialize action executor with default handlers.
        
        Args:
            config: Game configuration
        """
        self.config = config
        
        # Initialize action handlers
        self.handlers = [
            MovementActionHandler(config),
            ShootActionHandler(config),
            PassActionHandler(config),
        ]
    
    def execute_actions(self, actions: List[PlayerAction], context: dict) -> List[ActionResult]:
        """
        Execute a list of actions using appropriate handlers.
        
        Args:
            actions: List of actions to execute
            context: Execution context with required systems
            
        Returns:
            List of action results
        """
        results = []
        
        for action in actions:
            result = self.execute_single_action(action, context)
            results.append(result)
        
        return results
    
    def execute_single_action(self, action: PlayerAction, context: dict) -> ActionResult:
        """
        Execute a single action using the appropriate handler.
        
        Args:
            action: Action to execute
            context: Execution context
            
        Returns:
            Action execution result
        """
        # Find handler for this action
        handler = self._find_handler(action)
        
        if not handler:
            return ActionResult.failure(
                f"No handler found for action: {action.action_type}"
            )
        
        # Execute action using handler
        try:
            return handler.execute(action, context)
        except Exception as e:
            return ActionResult.failure(
                f"Action execution failed: {str(e)}",
                {'exception': e, 'action': action}
            )
    
    def _find_handler(self, action: PlayerAction) -> Optional[ActionHandler]:
        """Find the appropriate handler for an action"""
        for handler in self.handlers:
            if handler.can_handle(action):
                return handler
        return None
    
    def add_handler(self, handler: ActionHandler) -> None:
        """
        Add a custom action handler.
        
        Args:
            handler: Custom action handler to add
        """
        self.handlers.append(handler)
    
    def remove_handler(self, handler_type: type) -> bool:
        """
        Remove a handler of specified type.
        
        Args:
            handler_type: Type of handler to remove
            
        Returns:
            True if handler was removed, False if not found
        """
        for i, handler in enumerate(self.handlers):
            if isinstance(handler, handler_type):
                self.handlers.pop(i)
                return True
        return False
    
    def validate_context(self, context: dict, required_keys: List[str]) -> Tuple[bool, str]:
        """
        Validate that context contains required keys.
        
        Args:
            context: Context dictionary to validate
            required_keys: List of required keys
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        missing_keys = [key for key in required_keys if key not in context]
        
        if missing_keys:
            return False, f"Missing required context keys: {missing_keys}"
        
        return True, ""