"""
Soccer Goal Implementation with Physics and Collision Detection

This module implements soccer goals with realistic physics, collision detection,
and goal scoring mechanics. Each goal consists of posts with physics collision
and a sensor area for detecting when the ball crosses the goal line.

**Key Features**:
1. **Physics Posts**: Solid goal posts that ball and players can collide with
2. **Goal Detection**: Sensor area to detect ball entry for scoring
3. **Directional Goals**: Support for left-facing and right-facing goals
4. **Visual Rendering**: Draw goal posts and sensor area for debugging

**Goal Structure**:
- Two vertical posts forming the goal opening
- Rectangular sensor area behind the posts
- Orientation-aware positioning (left vs right goals)
- Collision-enabled posts with elasticity and friction
"""

import pygame
import pymunk
from pymunk import Vec2d


class Goal:
    """
    Soccer goal with physics posts and goal detection sensor.
    
    **Purpose**: Provide goal structure with collision and scoring detection
    
    **Key Components**:
    1. **Goal Posts**: Two vertical pymunk segments with collision physics
    2. **Sensor Area**: Invisible rectangular sensor behind posts for goal detection
    3. **Orientation System**: Supports left-facing and right-facing goals
    4. **Collision Detection**: Method to check if ball position is inside goal
    
    **Physics Properties**:
    - Posts have elasticity (bouncy) and friction for realistic ball interaction
    - Sensor area is collision-free but detects object entry
    - Orientation determines goal opening direction (left/right)
    
    **Usage**: Created by Pitch class, used by Match class for goal detection
    """
    def __init__(self, space, position, orientation="right", width=30, height=120, post_thickness=5, collision_type=10, name="goal"):
        self.GOAL_COLOR = pygame.Color("grey")
        self.space = space
        self.position = Vec2d(position[0], position[1])
        self.width = width
        self.height = height
        self.name = name
        self.collision_type = collision_type
        self.score = False
        self.post_thickness = post_thickness

        if orientation not in ("left", "right"):
            raise ValueError("Orientation must be 'left' or 'right'")
        self.orientation = orientation

        # Set direction and perpendicular vectors based on orientation
        if orientation == "right":
            self.direction = Vec2d(1, 0)
        else:  # "left"
            self.direction = Vec2d(-1, 0)
        self.perp = self.direction.perpendicular()

        self._create_posts(post_thickness)
        self._create_sensor_area()

    def _create_posts(self, thickness):
        top_offset = self.perp.normalized() * (self.height / 2)

        post1_start = self.position + top_offset
        post1_end = self.position + top_offset - self.direction * self.width

        post2_start = self.position - top_offset
        post2_end = self.position - top_offset - self.direction * self.width

        self.post1 = pymunk.Segment(self.space.static_body, post1_start, post1_end, thickness)
        self.post2 = pymunk.Segment(self.space.static_body, post2_start, post2_end, thickness)

        for post in [self.post1, self.post2]:
            post.elasticity = 1.0
            post.friction = 0.5

        self.space.add(self.post1, self.post2)

    def _create_sensor_area(self):
        top = self.position + self.perp.normalized() * (self.height / 2)
        bottom = self.position - self.perp.normalized() * (self.height / 2)
        back = -self.direction.normalized() * self.width

        corner1 = top
        corner2 = bottom
        corner3 = bottom + back
        corner4 = top + back

        self.goal_shape = pymunk.Poly(self.space.static_body, [corner1, corner2, corner3, corner4])
        self.goal_shape.sensor = True
        self.goal_shape.collision_type = self.collision_type

        self.space.add(self.goal_shape)


    def is_ball_inside_goal(self, ball_pos):
        tolerance = 5

        half_height = self.height / 2

        # Check if within height bounds
        within_height = self.position.y - half_height + tolerance <= ball_pos.y <= self.position.y + half_height - tolerance
        
        # Check depth bounds based on goal orientation
        if self.orientation == "right":
            # Left goal (position 52, opens right) - ball must cross goal line going left
            # Goal area is behind the goal line (x < 52)
            within_depth = ball_pos.x <= self.position.x - tolerance
        else:  # orientation == "left"
            # Right goal (position 748, opens left) - ball must cross goal line going right  
            # Goal area is behind the goal line (x > 748)
            within_depth = ball_pos.x >= self.position.x + tolerance

        return within_height and within_depth


    def draw(self, surface):
        for post in [self.post1, self.post2]:
            start = int(post.a.x), int(post.a.y)
            end = int(post.b.x), int(post.b.y)
            pygame.draw.line(surface, self.GOAL_COLOR, start, end, int(self.post_thickness))

        points = [(int(p.x), int(p.y)) for p in self.goal_shape.get_vertices()]
        pygame.draw.polygon(surface, self.GOAL_COLOR, points, width=1)
