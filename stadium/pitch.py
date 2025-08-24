"""
Soccer Field/Pitch Implementation

This module creates the soccer field with physics boundaries, visual rendering,
and goal objects. It handles the field layout, wall physics, and field markings.

**Key Components**:
1. **Physics Boundaries**: Invisible walls around the field for ball/player collision
2. **Visual Field**: Green field with white line markings (boundaries, center line, circle)
3. **Goal Integration**: Manages two Goal objects for score detection
4. **Field Constants**: Defines dimensions and positions for game layout

**Field Layout**:
- 800x600 pixel field with 50-pixel margins
- Center line divides field in half
- Goals positioned at x=52 (left) and x=748 (right)
- Elastic walls prevent objects from leaving the field
"""

import pygame
import pymunk

from stadium.goal import Goal


class Pitch:
    """
    Soccer field with physics boundaries, visual rendering, and goal management.
    
    **Purpose**: Create the playing environment with realistic field physics
    
    **Key Features**:
    1. **Physics Walls**: Invisible barriers around field perimeter for collision
    2. **Field Rendering**: Green field with white soccer field markings
    3. **Goal Management**: Two Goal objects for score detection
    4. **Coordinate System**: Defines field dimensions and important positions
    
    **Field Specifications**:
    - Total area: 800x600 pixels
    - Playing field: 700x500 pixels (50px margins)
    - Center at (400, 300)
    - Goals at x=52 (left, team_2 scores) and x=748 (right, team_1 scores)
    - Elastic walls with full bounce (elasticity = 1.0)
    """

    def __init__(self, space) -> None:
        # Field boundary constants (playable area)
        self.FIELD_TOP = 50
        self.FIELD_BOTTOM = 550
        self.FIELD_LEFT = 50
        self.FIELD_RIGHT = 750
        self.FIELD_CENTER_Y = (self.FIELD_TOP + self.FIELD_BOTTOM) // 2
        
        # Define wall positions (outside the playable area for physics)
        top_wall = pymunk.Segment(space.static_body, (0, -20), (800, -20), 20)
        bottom_wall = pymunk.Segment(space.static_body, (0, 620), (800, 620), 20)
        left_wall = pymunk.Segment(space.static_body, (-20, 0), (-20, 600), 20)
        right_wall = pymunk.Segment(space.static_body, (820, 0), (820, 600), 20)

        # Set wall elasticity (bouncy)
        for wall in [top_wall, bottom_wall, left_wall, right_wall]:
            wall.elasticity = 1.0
            space.add(wall)

        self.GOAL_COLOR = pygame.Color("grey")
        self.GOAL_WIDTH = 30
        self.GOAL_HEIGHT = 120  # Size of goal opening

        # Goals with increased depth to reach screen edges (prevent ball from entering behind)
        self.goal_left = Goal(space = space, position = (52, 300), orientation="right", width=52)  # Extends to left edge (0)
        self.goal_right = Goal(space = space, position = (748, 300), orientation="left", width=52)  # Extends to right edge (800)

    def draw_pitch(self, surface):
        GREEN = (34, 139, 34)

        surface.fill(GREEN)

        self.draw_white_lines(surface)
        self.goal_left.draw(surface)
        self.goal_right.draw(surface)

    def draw_white_lines(self, surface):
        WHITE = (255, 255, 255)
        # Outer boundaries
        pygame.draw.rect(surface, WHITE, pygame.Rect(50, 50, 700, 500), 5)
        # Midfield line
        pygame.draw.line(surface, WHITE, (400, 50), (400, 550), 3)
        # Center circle
        pygame.draw.circle(surface, WHITE, (400, 300), 60, 3)
    
    def get_field_bounds(self) -> dict:
        """
        Get the field boundaries for set-piece detection.
        
        Returns:
            dict: Field boundaries with 'left', 'right', 'top', 'bottom' coordinates
        """
        return {
            'left': self.FIELD_LEFT,
            'right': self.FIELD_RIGHT, 
            'top': self.FIELD_TOP,
            'bottom': self.FIELD_BOTTOM
        }


