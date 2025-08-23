"""
Game Configuration Constants

This module centralizes all game constants and configuration values that were
previously scattered throughout the codebase. This improves maintainability
and makes it easier to tune game balance.

**Categories**:
1. **Physics**: Mass, forces, damping, velocity limits
2. **Ball Control**: Dribble distances, spring properties
3. **Actions**: Shot strength, pass power, cooldowns
4. **Visual**: Rendering sizes, colors, fonts
5. **Field**: Dimensions, goal positions
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class PhysicsConfig:
    """Physics-related constants"""
    PLAYER_MASS: float = 20.0
    PLAYER_RADIUS: float = 15.0
    PLAYER_ELASTICITY: float = 0.7
    PLAYER_DAMPING: float = 0.1
    VELOCITY_DAMPING: float = 0.96
    MAX_SPEED: float = 200.0
    MOVEMENT_FORCE: float = 10000.0


@dataclass(frozen=True)
class BallControlConfig:
    """Ball control and dribbling constants"""
    DRIBBLE_DISTANCE: float = 30.0
    SPRING_LENGTH: float = 10.0
    DRIBBLE_FORCE: float = 500.0
    FRONT_OFFSET_LENGTH: float = 50.0
    CONTROL_RADIUS: float = 40.0
    SPRING_DAMPING: float = 30.0


@dataclass(frozen=True)
class ActionConfig:
    """Action-related constants (shooting, passing, cooldowns)"""
    SHOT_STRENGTH: float = 500.0
    MAX_PASS_STRENGTH: float = 300.0
    DRIBBLE_COOLDOWN: float = 0.3
    
    # Action thresholds
    SHOOT_DISTANCE_THRESHOLD: float = 150.0
    PASS_DISTANCE_MIN: float = 50.0
    PASS_DISTANCE_MAX: float = 200.0


@dataclass(frozen=True)
class VisualConfig:
    """Visual rendering constants"""
    PLAYER_RADIUS: float = 15.0
    FONT_SIZE: int = 24
    TEXT_COLOR: Tuple[int, int, int] = (255, 255, 255)  # White
    PHYSICS_DEBUG_THICKNESS: int = 2


@dataclass(frozen=True)
class FieldConfig:
    """Field dimensions and positions"""
    WIDTH: int = 800
    HEIGHT: int = 600
    
    # Goal positions
    LEFT_GOAL_X: float = 52.0
    RIGHT_GOAL_X: float = 748.0
    GOAL_CENTER_Y: float = 300.0
    GOAL_WIDTH: float = 52.0
    GOAL_HEIGHT: float = 120.0


class GameConfig:
    """
    Central game configuration container.
    
    **Usage**:
    ```python
    from config.game_config import GameConfig
    
    config = GameConfig()
    player_mass = config.physics.PLAYER_MASS
    shot_power = config.actions.SHOT_STRENGTH
    ```
    
    **Benefits**:
    - Single source of truth for all constants
    - Easy to modify game balance
    - Type-safe configuration access
    - Clear categorization of settings
    """
    
    def __init__(self):
        self.physics = PhysicsConfig()
        self.ball_control = BallControlConfig()
        self.actions = ActionConfig()
        self.visual = VisualConfig()
        self.field = FieldConfig()
    
    @classmethod
    def create_default(cls) -> 'GameConfig':
        """Create default game configuration"""
        return cls()
    
    @classmethod
    def create_fast_paced(cls) -> 'GameConfig':
        """Create configuration for fast-paced gameplay"""
        config = cls()
        # Override specific values for faster gameplay
        object.__setattr__(config.physics, 'MAX_SPEED', 300.0)
        object.__setattr__(config.physics, 'MOVEMENT_FORCE', 15000.0)
        object.__setattr__(config.actions, 'DRIBBLE_COOLDOWN', 0.5)
        return config
    
    @classmethod 
    def create_simulation(cls) -> 'GameConfig':
        """Create configuration optimized for training/simulation"""
        config = cls()
        # Override for more predictable physics in training
        object.__setattr__(config.physics, 'VELOCITY_DAMPING', 0.9)
        object.__setattr__(config.ball_control, 'DRIBBLE_DISTANCE', 35.0)
        return config


# Global default instance for backward compatibility
# TODO: Remove this once all code is refactored to use dependency injection
DEFAULT_CONFIG = GameConfig.create_default()