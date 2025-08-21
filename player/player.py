"""
Soccer Player Coordinator System

This module coordinates between specialized player systems to provide a unified
interface for player functionality. The Player class delegates responsibilities
to focused components following the Single Responsibility Principle.

**Architecture**:
1. **PlayerPhysics**: Handles physics body and movement forces
2. **BallController**: Manages ball interaction and dribbling springs
3. **PlayerRenderer**: Handles visual rendering and drawing
4. **ActionExecutor**: Executes player actions (shoot, pass, move)
5. **PlayerController**: Manages input and action coordination

**Key Benefits**:
- Clear separation of concerns
- Easier testing and maintenance
- Flexible component composition
- Reduced coupling between systems
"""

import time
import pygame
from typing import List, Optional

from common.Vector import Vector
from config.game_config import GameConfig
from player.player_physics import PlayerPhysics
from player.ball_controller import BallController
from player.player_renderer import PlayerRenderer
from player.action_executor import ActionExecutor
from player.player_controller import PlayerController


class Player:
    """
    Unified player interface that coordinates specialized subsystems.
    
    **Single Responsibility**: Coordinate player subsystems and provide unified interface
    
    **Architecture**: This class delegates to specialized components:
    - PlayerPhysics: Physics body and movement
    - BallController: Ball interaction and dribbling
    - PlayerRenderer: Visual representation
    - ActionExecutor: Action execution (shoot, pass, move)  
    - PlayerController: Input handling and action coordination
    
    **Usage**:
    ```python
    player = Player(space, Vector(100, 200), "red", 5, config)
    player.set_ball(ball)
    
    # For keyboard control
    player.update_keyboard(keys, teammates)
    
    # For AI control
    player.update_ai(up, down, left, right, shoot, pass_ball, teammates)
    
    # Each frame
    player.simulate()
    player.draw(surface)
    ```
    """
    
    def __init__(self, space, position: Vector, color: str, number: int, 
                 config: Optional[GameConfig] = None):
        """
        Initialize player with all specialized subsystems.
        
        Args:
            space: pymunk physics space
            position: Initial player position
            color: Team color for rendering
            number: Jersey number (1-5)
            config: Game configuration (creates default if None)
        """
        self.config = config or GameConfig.create_default()
        self.space = space
        self._number = number
        self.ball = None
        
        # Initialize specialized subsystems
        self.physics = PlayerPhysics(space, position, self.config)
        self.ball_controller = BallController(self.config)
        self.renderer = PlayerRenderer(color, number, self.config)
        self.action_executor = ActionExecutor(self.config)
        self.controller = PlayerController(self.config)
        
        # Legacy compatibility properties
        self._last_velocity = Vector(0, 0)
        
    @property
    def player_body(self):
        """Legacy compatibility: access physics body"""
        return self.physics.body
        
    @property  
    def radius(self) -> float:
        """Player radius for collision detection"""
        return self.config.physics.PLAYER_RADIUS
        
    @property
    def pos(self) -> pygame.Vector2:
        """Legacy compatibility: position as pygame Vector2"""
        position = self.physics.get_position()
        return pygame.Vector2(position.x, position.y)

    def position(self) -> Vector:
        """Get current player position"""
        return self.physics.get_position()
    
    def has_ball_control(self) -> bool:
        """
        Check if this player currently has control of the ball.
        
        Returns:
            bool: True if player has active ball control
        """
        return self.ball_controller.has_ball_control()
    
    def set_ball(self, ball) -> None:
        """
        Set the ball reference for this player.
        
        Args:
            ball: Ball object for interaction
        """
        self.ball = ball

    def play(self, ball):
        """Legacy method - use set_ball() instead"""
        self.set_ball(ball)

    def simulate(self) -> None:
        """
        Update player state each frame.
        
        This method coordinates all subsystems:
        - Updates physics (velocity limits)
        - Updates ball control (spring management)
        - Tracks velocity for legacy compatibility
        """
        current_time = time.time()
        
        # Update physics system
        self.physics.update()
        
        # Update ball controller if ball is available
        if self.ball is not None:
            player_position = self.physics.get_position()
            player_velocity = self.physics.get_velocity()
            
            self.ball_controller.update(
                player_position, 
                player_velocity, 
                self.ball, 
                self.space, 
                current_time
            )
        
        # Update legacy velocity tracking
        current_velocity = self.physics.get_velocity()
        if current_velocity.length() > 0:
            self._last_velocity = current_velocity

    def control(self, keys, teammates_positions: List[Vector]) -> None:
        """
        Legacy keyboard control method.
        
        Args:
            keys: pygame key state (ScancodeWrapper from pygame.key.get_pressed())
            teammates_positions: List of teammate positions
        """
        # Convert pygame ScancodeWrapper to dictionary format expected by controller
        # Extract the relevant keys we need for player control
        keys_dict = {
            pygame.K_UP: keys[pygame.K_UP],
            pygame.K_DOWN: keys[pygame.K_DOWN],
            pygame.K_LEFT: keys[pygame.K_LEFT],
            pygame.K_RIGHT: keys[pygame.K_RIGHT],
            pygame.K_d: keys[pygame.K_d],
            pygame.K_s: keys[pygame.K_s],
        }
        
        # Get actions from controller
        actions = self.controller.get_keyboard_actions(keys_dict, teammates_positions)
        
        # Execute actions
        self._execute_actions(actions, teammates_positions)
    
    def apply_actions(self, move_up: bool, move_down: bool, move_left: bool, 
                     move_right: bool, shoot: bool, pass_ball: bool, 
                     teammates_positions: List[Vector]) -> None:
        """
        Generic player control method that applies movement and actions.
        
        Args:
            move_up, move_down, move_left, move_right: Boolean movement directions
            shoot: Boolean shoot action
            pass_ball: Boolean pass action  
            teammates_positions: List of teammate positions for passing
        """
        # Get actions from controller
        actions = self.controller.get_ai_actions(
            move_up, move_down, move_left, move_right, shoot, pass_ball, teammates_positions
        )
        
        # Execute actions
        self._execute_actions(actions, teammates_positions)
    
    def _execute_actions(self, actions, teammates_positions: List[Vector]) -> None:
        """Execute list of player actions using action executor"""
        if not actions:
            return
            
        # Build execution context
        context = {
            'physics': self.physics,
            'ball_controller': self.ball_controller,
            'ball': self.ball,
            'space': self.space,
            'teammates_positions': teammates_positions,
            'current_time': time.time()
        }
        
        # Execute all actions
        results = self.action_executor.execute_actions(actions, context)

    def remove_ball_springs(self) -> None:
        """Legacy method - ball spring removal is now handled by BallController"""
        if self.ball_controller:
            self.ball_controller.release_ball_control(self.space, time.time())

    def draw(self, surface) -> None:
        """
        Draw the player on the surface with animated limbs.
        
        Args:
            surface: pygame surface to draw on
        """
        current_position = self.physics.get_position()
        current_velocity = self.physics.get_velocity()
        has_control = self.has_ball_control()
        self.renderer.draw(surface, current_position, velocity=current_velocity, has_ball_control=has_control)

    def reset(self, initial_position: Vector) -> None:
        """
        Reset player to initial state.
        
        Args:
            initial_position: Position to reset to
        """
        # Reset all subsystems
        self.physics.reset(initial_position)
        self.ball_controller.reset(self.space)
        
        # Reset legacy state
        self._last_velocity = Vector(0, 0)
    
    def cleanup(self) -> None:
        """Clean up player resources (for removal from game)"""
        self.physics.cleanup()
        self.ball_controller.cleanup(self.space)
        
    # Legacy compatibility methods (deprecated - use specific subsystem methods)
    
    def _handover(self, teammates_positions):
        """
        Legacy passing method - now handled by ActionExecutor.
        This method is kept for backward compatibility but should not be used directly.
        """
        # This functionality is now in PassActionHandler within ActionExecutor
        pass

    def _shoot(self, diff):
        """
        Legacy shooting method - now handled by ActionExecutor. 
        This method is kept for backward compatibility but should not be used directly.
        """
        # This functionality is now in ShootActionHandler within ActionExecutor
        pass
