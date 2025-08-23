"""
Player Physics System

This module handles all physics-related functionality for players, including:
- Physics body creation and management
- Velocity control and damping
- Force application and movement
- Collision detection setup

**Responsibility**: Physics simulation and body management
**Dependencies**: pymunk physics engine, GameConfig
"""

import pymunk
from typing import Tuple
from common.Vector import Vector
from config.game_config import GameConfig


class PlayerPhysics:
    """
    Handles physics body management and movement for a player.
    
    **Single Responsibility**: Physics simulation and body properties
    
    **Key Features**:
    - Creates and manages pymunk physics body
    - Handles velocity limits and damping
    - Provides clean interface for movement forces
    - Manages collision properties
    
    **Usage**:
    ```python
    physics = PlayerPhysics(space, Vector(100, 200), config)
    physics.apply_movement_force(0, -1000)  # Move up
    physics.update()  # Apply velocity limits
    position = physics.get_position()
    ```
    """
    
    def __init__(self, space: pymunk.Space, initial_position: Vector, config: GameConfig):
        """
        Initialize player physics body.
        
        Args:
            space: pymunk physics space
            initial_position: Starting position on field
            config: Game configuration with physics constants
        """
        self.space = space
        self.config = config
        
        # Create physics body
        player_moment = pymunk.moment_for_circle(
            1, 0, config.physics.PLAYER_RADIUS
        )
        self.body = pymunk.Body(
            config.physics.PLAYER_MASS, 
            player_moment
        )
        
        # Set initial properties
        self.body.position = (initial_position.x, initial_position.y)
        self.body.damping = config.physics.PLAYER_DAMPING
        self.body.elasticity = config.physics.PLAYER_ELASTICITY
        
        # Set up custom velocity damping function
        self.body.velocity_func = self._create_velocity_damping_func(
            config.physics.VELOCITY_DAMPING
        )
        
        # Create collision shape
        self.shape = pymunk.Circle(self.body, config.physics.PLAYER_RADIUS)
        self.shape.elasticity = config.physics.PLAYER_ELASTICITY
        self.shape.filter = pymunk.ShapeFilter(group=1)
        
        # Add to physics space
        space.add(self.body, self.shape)
    
    def _create_velocity_damping_func(self, damping_factor: float):
        """Create velocity damping function with specified damping factor"""
        def damp_velocity(body, gravity, damping, dt):
            pymunk.Body.update_velocity(body, gravity, damping_factor, dt)
        return damp_velocity
    
    def apply_movement_force(self, force_x: float, force_y: float) -> None:
        """
        Apply movement force to the physics body.
        
        Args:
            force_x: Force in X direction
            force_y: Force in Y direction
        """
        self.body.apply_force_at_local_point((force_x, force_y))
    
    def apply_directional_movement(self, direction: Vector, strength: float = None) -> None:
        """
        Apply movement force in a specific direction.
        
        Args:
            direction: Normalized direction vector
            strength: Force strength (uses default if None)
        """
        if strength is None:
            strength = self.config.physics.MOVEMENT_FORCE
            
        force = direction.normalize() * strength
        self.apply_movement_force(force.x, force.y)
    
    def update(self) -> None:
        """
        Update physics state, applying velocity limits.
        
        Should be called each frame after forces are applied.
        """
        # Apply maximum speed limit
        current_speed = self.body.velocity.length
        if current_speed > self.config.physics.MAX_SPEED:
            # Scale velocity down to maximum speed
            velocity_direction = self.body.velocity.normalized()
            self.body.velocity = velocity_direction * self.config.physics.MAX_SPEED
    
    def get_position(self) -> Vector:
        """Get current position as Vector"""
        return Vector(self.body.position.x, self.body.position.y)
    
    def get_velocity(self) -> Vector:
        """Get current velocity as Vector"""
        return Vector(self.body.velocity.x, self.body.velocity.y)
    
    def get_speed(self) -> float:
        """Get current speed (velocity magnitude)"""
        return self.body.velocity.length
    
    def set_position(self, position: Vector) -> None:
        """Set position directly (for resets, teleportation)"""
        self.body.position = (position.x, position.y)
    
    def set_velocity(self, velocity: Vector) -> None:
        """Set velocity directly"""
        self.body.velocity = (velocity.x, velocity.y)
    
    def stop(self) -> None:
        """Stop all movement immediately"""
        self.body.velocity = (0, 0)
    
    def is_moving(self, threshold: float = 1.0) -> bool:
        """Check if player is moving faster than threshold"""
        return self.get_speed() > threshold
    
    def get_movement_direction(self) -> Vector:
        """Get normalized direction of current movement (or zero vector if not moving)"""
        velocity = self.get_velocity()
        if velocity.length() > 0.1:  # Avoid division by zero
            return velocity.normalize()
        return Vector(0, 0)
    
    def reset(self, position: Vector) -> None:
        """
        Reset physics body to initial state.
        
        Args:
            position: New position to reset to
        """
        self.set_position(position)
        self.stop()
    
    def cleanup(self) -> None:
        """Remove physics body from space (for cleanup)"""
        if self.body in self.space.bodies:
            self.space.remove(self.body, self.shape)
    
    def __del__(self):
        """Ensure cleanup when object is destroyed"""
        try:
            self.cleanup()
        except:
            pass  # Space might already be destroyed