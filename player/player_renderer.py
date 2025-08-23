"""
Player Visual Rendering System

This module handles all visual representation of players, including:
- Drawing player circles with team colors
- Rendering player numbers/jersey numbers
- Visual effects for special states
- Debug visualization support

**Responsibility**: Visual rendering and display
**Dependencies**: pygame, GameConfig
"""

import pygame
import math
import time
from typing import Optional, Tuple
from common.Vector import Vector
from config.game_config import GameConfig


class PlayerRenderer:
    """
    Handles visual rendering of a player.
    
    **Single Responsibility**: Visual representation and drawing
    
    **Key Features**:
    - Draws player circle with team color
    - Renders jersey number on player
    - Supports different visual states (normal, highlighted, etc.)
    - Clean separation from game logic
    
    **Usage**:
    ```python
    renderer = PlayerRenderer("red", 5, config)
    renderer.draw(surface, position, radius=15)
    renderer.draw_highlighted(surface, position)
    ```
    """
    
    def __init__(self, color: str, jersey_number: int, config: GameConfig):
        """
        Initialize player renderer.
        
        Args:
            color: Team color (e.g., "red", "blue", or hex color)
            jersey_number: Player's jersey number (1-5)
            config: Game configuration with visual settings
        """
        self.config = config
        self.jersey_number = jersey_number
        
        # Convert color string to pygame Color
        if isinstance(color, str):
            self.color = pygame.Color(color)
        else:
            self.color = color
            
        # Cache font for performance
        self._font = None
        self._font_size = config.visual.FONT_SIZE
        
        # Visual states
        self.is_highlighted = False
        self.is_selected = False
        
        # Animation state for limb movement
        self._animation_time = 0.0
        self._last_velocity = Vector(0, 0)
        self._movement_speed = 0.0
        
        # Animation parameters
        self.limb_color = pygame.Color("black") if color != "black" else pygame.Color("white")
        self.animation_speed = 8.0  # Limb oscillation frequency
        self.limb_length = 8.0      # Length of arms and legs
        self.limb_thickness = 2     # Thickness of limbs
    
    @property
    def font(self) -> pygame.font.Font:
        """Lazy-loaded font property for performance"""
        if self._font is None:
            self._font = pygame.font.Font(None, self._font_size)
        return self._font
    
    def draw(self, surface: pygame.Surface, position: Vector, 
             radius: Optional[float] = None, velocity: Optional[Vector] = None, 
             has_ball_control: bool = False) -> None:
        """
        Draw the player at the specified position with animated limbs.
        
        Args:
            surface: pygame surface to draw on
            position: Player position
            radius: Player radius (uses config default if None)
            velocity: Player velocity for animation (optional)
            has_ball_control: Whether player has ball control (affects limb animation)
        """
        if radius is None:
            radius = self.config.visual.PLAYER_RADIUS
            
        # Update animation based on velocity
        if velocity is not None:
            self._update_animation(velocity)
        
        # Draw animated limbs first (behind the body)
        self._draw_animated_limbs(surface, position, radius, velocity or Vector(0, 0), has_ball_control)
        
        # Draw player circle (body)
        pygame.draw.circle(
            surface, 
            self.color, 
            (int(position.x), int(position.y)), 
            int(radius)
        )
        
        # Draw small inner circle for selected player
        if self.is_selected:
            self._draw_inner_selection_circle(surface, position, radius)
        
        # Draw jersey number
        self._draw_jersey_number(surface, position)
        
        # Draw highlight effects if needed (selection is now handled by yellow jersey number + inner circle)
        if self.is_highlighted:
            self._draw_highlight_indicator(surface, position, radius)
    
    def draw_highlighted(self, surface: pygame.Surface, position: Vector,
                        radius: Optional[float] = None, velocity: Optional[Vector] = None) -> None:
        """
        Draw the player with highlight effect (temporary highlight).
        
        Args:
            surface: pygame surface to draw on
            position: Player position
            radius: Player radius (uses config default if None)
            velocity: Player velocity for animation (optional)
        """
        old_highlight = self.is_highlighted
        self.is_highlighted = True
        self.draw(surface, position, radius, velocity)
        self.is_highlighted = old_highlight
    
    def draw_selected(self, surface: pygame.Surface, position: Vector,
                     radius: Optional[float] = None, velocity: Optional[Vector] = None) -> None:
        """
        Draw the player with selection indicator (currently controlled player).
        
        Args:
            surface: pygame surface to draw on
            position: Player position  
            radius: Player radius (uses config default if None)
            velocity: Player velocity for animation (optional)
        """
        old_selected = self.is_selected
        self.is_selected = True
        self.draw(surface, position, radius, velocity)
        self.is_selected = old_selected
    
    def _draw_jersey_number(self, surface: pygame.Surface, position: Vector) -> None:
        """Draw the player's jersey number centered on the player"""
        # Use yellow color for selected player, normal color otherwise
        text_color = pygame.Color("yellow") if self.is_selected else self.config.visual.TEXT_COLOR
        
        text_surface = self.font.render(
            str(self.jersey_number), 
            True, 
            text_color
        )
        text_rect = text_surface.get_rect(
            center=(int(position.x), int(position.y))
        )
        surface.blit(text_surface, text_rect)
    
    def _draw_inner_selection_circle(self, surface: pygame.Surface, 
                                   position: Vector, radius: float) -> None:
        """Draw small inner circle for selected player"""
        # Small circle inside the player body
        inner_radius = radius * 0.3  # 30% of player radius
        inner_color = pygame.Color("yellow")  # Matches the jersey number color
        
        pygame.draw.circle(
            surface,
            inner_color,
            (int(position.x), int(position.y)),
            int(inner_radius)
        )
    
    def _draw_selection_indicator(self, surface: pygame.Surface, 
                                position: Vector, radius: float) -> None:
        """Draw selection indicator around the player (deprecated - now uses yellow number + inner circle)"""
        # Selection is now handled by yellow jersey number and inner circle instead of outline
        pass
    
    def _draw_highlight_indicator(self, surface: pygame.Surface,
                                position: Vector, radius: float) -> None:
        """Draw highlight indicator around the player"""
        # Draw yellow outline for highlighted player
        pygame.draw.circle(
            surface,
            pygame.Color("yellow"),
            (int(position.x), int(position.y)),
            int(radius + 2),
            2  # Thickness
        )
    
    def draw_debug_info(self, surface: pygame.Surface, position: Vector,
                       velocity: Vector, debug_info: dict) -> None:
        """
        Draw debug information around the player.
        
        Args:
            surface: pygame surface to draw on
            position: Player position
            velocity: Player velocity for drawing direction
            debug_info: Dictionary with debug information to display
        """
        # Draw velocity vector
        if velocity.length() > 1:
            end_pos = position + velocity.normalize() * 30
            pygame.draw.line(
                surface,
                pygame.Color("green"),
                (int(position.x), int(position.y)),
                (int(end_pos.x), int(end_pos.y)),
                2
            )
            
        # Draw debug text
        y_offset = -40
        for key, value in debug_info.items():
            debug_text = f"{key}: {value}"
            text_surface = pygame.font.Font(None, 16).render(
                debug_text, True, pygame.Color("white")
            )
            surface.blit(
                text_surface, 
                (int(position.x) - 30, int(position.y) + y_offset)
            )
            y_offset += 15
    
    def set_highlighted(self, highlighted: bool) -> None:
        """Set highlight state"""
        self.is_highlighted = highlighted
    
    def set_selected(self, selected: bool) -> None:
        """Set selection state"""
        self.is_selected = selected
    
    def change_color(self, new_color: str) -> None:
        """Change player color (useful for team changes, etc.)"""
        if isinstance(new_color, str):
            self.color = pygame.Color(new_color)
        else:
            self.color = new_color
    
    def change_jersey_number(self, new_number: int) -> None:
        """Change jersey number"""
        self.jersey_number = new_number
    
    def _update_animation(self, velocity: Vector) -> None:
        """
        Update animation state based on player velocity.
        
        Args:
            velocity: Current player velocity
        """
        self._last_velocity = velocity
        self._movement_speed = velocity.length()
        
        # Update animation time based on movement speed
        if self._movement_speed > 5.0:  # Only animate if moving significantly
            dt = 1.0 / 60.0  # Assume 60 FPS
            self._animation_time += dt * self.animation_speed * (self._movement_speed / 100.0)
        else:
            # Slow down animation when not moving much
            self._animation_time += 0.001
    
    def _draw_animated_limbs(self, surface: pygame.Surface, position: Vector, 
                           radius: float, velocity: Vector, has_ball_control: bool = False) -> None:
        """
        Draw animated arms and legs from top-down perspective.
        
        In top-down view:
        - Arms are positioned on left and right sides of body, swing forward/backward
        - Legs show running stride pattern with left/right positioning
        
        Args:
            surface: pygame surface to draw on
            position: Player center position
            radius: Player radius
            velocity: Player velocity for animation
            has_ball_control: Whether player has ball control
        """
        # Calculate movement direction (0 = right, π/2 = down, π = left, 3π/2 = up)
        if velocity.length() > 1.0:
            movement_direction = math.atan2(velocity.y, velocity.x)
        else:
            movement_direction = 0  # Default facing right when stationary
        
        # Animation phases for realistic running motion
        # Arms and legs move in opposite phases (when left leg forward, right arm forward)
        arm_phase = math.sin(self._animation_time * 2)  # -1 to 1
        leg_phase = -arm_phase  # Opposite phase for realistic running
        
        # TOP-DOWN ARM POSITIONS
        # Arms are positioned on left and right sides of the body
        arm_side_offset = radius * 0.8  # Distance from center to shoulder
        arm_length = self.limb_length
        
        # Left arm (player's left side)
        left_arm_base = Vector(
            position.x + arm_side_offset * math.cos(movement_direction - math.pi/2),
            position.y + arm_side_offset * math.sin(movement_direction - math.pi/2)
        )
        
        # Left arm swings forward/backward relative to movement direction
        left_arm_swing = arm_phase * 0.4  # Swing amplitude
        left_arm_end = Vector(
            left_arm_base.x + arm_length * math.cos(movement_direction + left_arm_swing),
            left_arm_base.y + arm_length * math.sin(movement_direction + left_arm_swing)
        )
        
        # Right arm (player's right side) 
        right_arm_base = Vector(
            position.x + arm_side_offset * math.cos(movement_direction + math.pi/2),
            position.y + arm_side_offset * math.sin(movement_direction + math.pi/2)
        )
        
        # Right arm swings opposite to left arm
        right_arm_swing = -arm_phase * 0.4
        right_arm_end = Vector(
            right_arm_base.x + arm_length * math.cos(movement_direction + right_arm_swing),
            right_arm_base.y + arm_length * math.sin(movement_direction + right_arm_swing)
        )
        
        # TOP-DOWN LEG POSITIONS  
        # Legs show running stride from above
        leg_side_offset = radius * 0.4  # Legs closer together than arms
        leg_length = self.limb_length * 1.2  # Legs slightly longer
        
        # Left leg (player's left side)
        left_leg_base = Vector(
            position.x + leg_side_offset * math.cos(movement_direction - math.pi/2),
            position.y + leg_side_offset * math.sin(movement_direction - math.pi/2)
        )
        
        # Left leg stride - swings forward/backward with larger amplitude
        left_leg_swing = leg_phase * 0.6
        left_leg_end = Vector(
            left_leg_base.x + leg_length * math.cos(movement_direction + left_leg_swing),
            left_leg_base.y + leg_length * math.sin(movement_direction + left_leg_swing)
        )
        
        # Right leg (player's right side)
        right_leg_base = Vector(
            position.x + leg_side_offset * math.cos(movement_direction + math.pi/2),
            position.y + leg_side_offset * math.sin(movement_direction + math.pi/2)
        )
        
        # Right leg strides opposite to left leg
        right_leg_swing = -leg_phase * 0.6
        right_leg_end = Vector(
            right_leg_base.x + leg_length * math.cos(movement_direction + right_leg_swing),
            right_leg_base.y + leg_length * math.sin(movement_direction + right_leg_swing)
        )
        
        # Draw limbs (if moving significantly)
        if self._movement_speed > 5.0:
            # Draw arms from shoulders
            pygame.draw.line(surface, self.limb_color,
                           (int(left_arm_base.x), int(left_arm_base.y)),
                           (int(left_arm_end.x), int(left_arm_end.y)), 
                           self.limb_thickness)
            pygame.draw.line(surface, self.limb_color,
                           (int(right_arm_base.x), int(right_arm_base.y)),
                           (int(right_arm_end.x), int(right_arm_end.y)), 
                           self.limb_thickness)
            
            # Draw legs from hips
            pygame.draw.line(surface, self.limb_color,
                           (int(left_leg_base.x), int(left_leg_base.y)),
                           (int(left_leg_end.x), int(left_leg_end.y)), 
                           self.limb_thickness)
            pygame.draw.line(surface, self.limb_color,
                           (int(right_leg_base.x), int(right_leg_base.y)),
                           (int(right_leg_end.x), int(right_leg_end.y)), 
                           self.limb_thickness)
        else:
            # Draw static limbs when not moving
            self._draw_static_limbs_topdown(surface, position, radius)
    
    def _draw_static_limbs_topdown(self, surface: pygame.Surface, position: Vector, radius: float) -> None:
        """
        Draw static limbs when player is not moving (top-down perspective).
        
        Args:
            surface: pygame surface to draw on
            position: Player center position
            radius: Player radius
        """
        # Static limbs in top-down view
        arm_side_offset = radius * 0.8
        leg_side_offset = radius * 0.4
        arm_length = self.limb_length
        leg_length = self.limb_length * 1.2
        
        # Static arms extending horizontally from shoulders (left and right sides)
        left_arm_base = Vector(position.x - arm_side_offset, position.y)
        right_arm_base = Vector(position.x + arm_side_offset, position.y)
        left_arm_end = Vector(left_arm_base.x - arm_length, left_arm_base.y)
        right_arm_end = Vector(right_arm_base.x + arm_length, right_arm_base.y)
        
        # Static legs extending downward from hips (slightly forward stance)
        left_leg_base = Vector(position.x - leg_side_offset, position.y)
        right_leg_base = Vector(position.x + leg_side_offset, position.y)
        left_leg_end = Vector(left_leg_base.x, left_leg_base.y + leg_length)
        right_leg_end = Vector(right_leg_base.x, right_leg_base.y + leg_length)
        
        # Draw static arms
        pygame.draw.line(surface, self.limb_color,
                       (int(left_arm_base.x), int(left_arm_base.y)),
                       (int(left_arm_end.x), int(left_arm_end.y)), 
                       self.limb_thickness)
        pygame.draw.line(surface, self.limb_color,
                       (int(right_arm_base.x), int(right_arm_base.y)),
                       (int(right_arm_end.x), int(right_arm_end.y)), 
                       self.limb_thickness)
        
        # Draw static legs
        pygame.draw.line(surface, self.limb_color,
                       (int(left_leg_base.x), int(left_leg_base.y)),
                       (int(left_leg_end.x), int(left_leg_end.y)), 
                       self.limb_thickness)
        pygame.draw.line(surface, self.limb_color,
                       (int(right_leg_base.x), int(right_leg_base.y)),
                       (int(right_leg_end.x), int(right_leg_end.y)), 
                       self.limb_thickness)


class TeamRenderer:
    """
    Helper class for rendering entire teams with consistent styling.
    
    **Purpose**: Coordinate rendering of multiple players with team-wide effects
    """
    
    def __init__(self, team_color: str, config: GameConfig):
        """
        Initialize team renderer.
        
        Args:
            team_color: Color for all players on this team
            config: Game configuration
        """
        self.team_color = team_color
        self.config = config
        self.player_renderers = []
    
    def create_player_renderer(self, jersey_number: int) -> PlayerRenderer:
        """Create and register a new player renderer for this team"""
        renderer = PlayerRenderer(self.team_color, jersey_number, self.config)
        self.player_renderers.append(renderer)
        return renderer
    
    def draw_all_players(self, surface: pygame.Surface, players_data: list,
                        selected_player_index: Optional[int] = None) -> None:
        """
        Draw all players for this team.
        
        Args:
            surface: pygame surface to draw on
            players_data: List of (position, renderer) tuples
            selected_player_index: Index of currently selected player
        """
        for i, (position, renderer) in enumerate(players_data):
            if i == selected_player_index:
                renderer.draw_selected(surface, position)
            else:
                renderer.draw(surface, position)