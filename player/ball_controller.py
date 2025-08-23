"""
Ball Control System

This module handles all ball interaction mechanics for players, including:
- Spring-based dribbling system
- Ball proximity detection
- Spring creation and management
- Ball control state tracking

**Responsibility**: Ball interaction and control mechanics
**Dependencies**: pymunk physics, GameConfig
"""

import time
import pymunk
from typing import Optional, List, Tuple
from common.Vector import Vector
from config.game_config import GameConfig


class SpringOffset:
    """Helper class to calculate spring anchor offsets based on player movement"""
    
    @staticmethod
    def calculate_offsets(velocity: Vector, offset_length: float) -> dict:
        """
        Calculate spring anchor offsets based on player velocity.
        
        Args:
            velocity: Player velocity vector
            offset_length: Length of offset from player center
            
        Returns:
            Dictionary with front, back, left, right offset vectors
        """
        if velocity.length() > 0:
            # Dynamic offsets based on movement direction
            velocity_normalized = velocity.normalize()
            base_offset = velocity_normalized * (offset_length / 2)
            
            return {
                'front': base_offset + Vector(0, offset_length / 2),
                'left': base_offset + Vector(offset_length / 2, 0),
                'right': base_offset + Vector(-offset_length / 2, 0),
                'back': base_offset + Vector(0, -offset_length / 2)
            }
        else:
            # Static offsets when not moving
            half_offset = offset_length / 2
            return {
                'front': Vector(0, half_offset),
                'left': Vector(half_offset, 0),
                'right': Vector(-half_offset, 0),
                'back': Vector(0, -half_offset)
            }


class BallController:
    """
    Handles ball interaction mechanics for a single player.
    
    **Single Responsibility**: Ball control and dribbling physics
    
    **Key Features**:
    - Creates and manages 4-spring dribbling system
    - Detects ball proximity and control state
    - Handles spring creation/removal with proper timing
    - Provides clean interface for ball control queries
    
    **Technical Details**:
    - Uses 4 springs (front, back, left, right) for natural ball control
    - Springs are created when player is close to ball and cooldown expired
    - Spring positions are dynamic based on player movement direction
    - Automatic spring cleanup when ball moves away
    
    **Usage**:
    ```python
    controller = BallController(config)
    controller.update(player_pos, player_vel, ball, space, current_time)
    has_control = controller.has_ball_control()
    controller.release_ball_control()
    ```
    """
    
    def __init__(self, config: GameConfig):
        """
        Initialize ball controller.
        
        Args:
            config: Game configuration with ball control settings
        """
        self.config = config
        
        # Spring references
        self.springs = {
            'front': None,
            'left': None,
            'right': None,
            'back': None
        }
        
        # State tracking
        self.last_action_time = 0.0
        self._last_ball_position = Vector(0, 0)
        self._control_established_time = 0.0
    
    def update(self, player_position: Vector, player_velocity: Vector, 
               ball, space: pymunk.Space, current_time: float) -> None:
        """
        Update ball control state based on current game state.
        
        Args:
            player_position: Current player position
            player_velocity: Current player velocity
            ball: Ball object with physics body
            space: pymunk physics space
            current_time: Current timestamp
            
        This method should be called every frame to maintain proper ball control.
        """
        ball_position = Vector(ball.ball_body.position.x, ball.ball_body.position.y)
        distance_to_ball = (ball_position - player_position).length()
        
        # Check if player is close enough to control ball
        if distance_to_ball < self.config.ball_control.DRIBBLE_DISTANCE:
            self._handle_ball_proximity(
                player_position, player_velocity, ball, space, current_time
            )
        else:
            # Ball too far - remove all springs
            self._remove_all_springs(space)
    
    def _handle_ball_proximity(self, player_position: Vector, player_velocity: Vector,
                              ball, space: pymunk.Space, current_time: float) -> None:
        """Handle logic when player is close to ball"""
        
        # Update existing springs if they exist
        if self.has_ball_control():
            self._update_spring_positions(player_position, player_velocity)
        
        # Create new springs if cooldown has expired and no springs exist
        elif self._can_establish_control(current_time):
            self._create_ball_control_springs(
                player_position, player_velocity, ball, space, current_time
            )
    
    def _can_establish_control(self, current_time: float) -> bool:
        """Check if player can establish ball control (cooldown check)"""
        return (current_time - self.last_action_time > 
                self.config.actions.DRIBBLE_COOLDOWN)
    
    def _create_ball_control_springs(self, player_position: Vector, player_velocity: Vector,
                                   ball, space: pymunk.Space, current_time: float) -> None:
        """Create all four dribbling springs"""
        
        # Calculate spring anchor offsets
        offsets = SpringOffset.calculate_offsets(
            player_velocity, 
            self.config.ball_control.FRONT_OFFSET_LENGTH
        )
        
        # Create each spring
        for direction, offset in offsets.items():
            anchor_position = player_position + offset
            spring = self._create_single_spring(anchor_position, ball, space)
            self.springs[direction] = spring
        
        self._control_established_time = current_time
    
    def _create_single_spring(self, anchor_position: Vector, ball, 
                            space: pymunk.Space) -> pymunk.DampedSpring:
        """Create a single dribbling spring"""
        spring = pymunk.DampedSpring(
            space.static_body,
            ball.ball_body,
            (anchor_position.x, anchor_position.y),
            (0, 0),  # Ball anchor at center
            rest_length=self.config.ball_control.SPRING_LENGTH,
            stiffness=self.config.ball_control.DRIBBLE_FORCE,
            damping=self.config.ball_control.SPRING_DAMPING
        )
        space.add(spring)
        return spring
    
    def _update_spring_positions(self, player_position: Vector, 
                               player_velocity: Vector) -> None:
        """Update existing spring anchor positions based on player movement"""
        if not self.has_ball_control():
            return
            
        # Recalculate offsets based on current movement
        offsets = SpringOffset.calculate_offsets(
            player_velocity,
            self.config.ball_control.FRONT_OFFSET_LENGTH
        )
        
        # Update each spring's anchor position
        for direction, spring in self.springs.items():
            if spring is not None and direction in offsets:
                anchor_position = player_position + offsets[direction]
                spring.anchor_a = (anchor_position.x, anchor_position.y)
    
    def _remove_all_springs(self, space: pymunk.Space) -> None:
        """Remove all dribbling springs from physics space"""
        for direction in self.springs:
            spring = self.springs[direction]
            if spring is not None and spring in space.constraints:
                space.remove(spring)
                self.springs[direction] = None
    
    def has_ball_control(self) -> bool:
        """
        Check if player currently has active ball control.
        
        Returns:
            True if at least one dribbling spring is active
        """
        return any(spring is not None for spring in self.springs.values())
    
    def release_ball_control(self, space: pymunk.Space, current_time: float) -> None:
        """
        Manually release ball control (for shooting, passing).
        
        Args:
            space: pymunk physics space
            current_time: Current timestamp (for cooldown tracking)
        """
        self._remove_all_springs(space)
        self.last_action_time = current_time
    
    def get_control_strength(self) -> float:
        """
        Get strength of current ball control (0.0 to 1.0).
        
        Returns:
            Control strength based on number of active springs
        """
        if not self.has_ball_control():
            return 0.0
            
        active_springs = sum(1 for spring in self.springs.values() 
                           if spring is not None)
        return active_springs / 4.0  # 4 springs = maximum control
    
    def get_control_duration(self, current_time: float) -> float:
        """
        Get how long player has had ball control.
        
        Args:
            current_time: Current timestamp
            
        Returns:
            Duration in seconds, or 0.0 if no control
        """
        if not self.has_ball_control():
            return 0.0
        return current_time - self._control_established_time
    
    def is_control_stable(self, current_time: float, min_duration: float = 0.1) -> bool:
        """
        Check if ball control is stable (held for minimum duration).
        
        Args:
            current_time: Current timestamp
            min_duration: Minimum duration to consider stable
            
        Returns:
            True if control is stable for at least min_duration
        """
        return self.get_control_duration(current_time) >= min_duration
    
    def cleanup(self, space: pymunk.Space) -> None:
        """Clean up all springs (for player removal/reset)"""
        self._remove_all_springs(space)
    
    def reset(self, space: pymunk.Space) -> None:
        """Reset ball controller to initial state"""
        self.cleanup(space)
        self.last_action_time = 0.0
        self._control_established_time = 0.0
        self._last_ball_position = Vector(0, 0)
    
    def __del__(self):
        """Ensure cleanup when object is destroyed"""
        # Note: Cannot access space here, cleanup should be called explicitly