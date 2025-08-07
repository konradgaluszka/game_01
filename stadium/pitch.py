import math

import pygame
import pymunk
from pygame import Rect
from pymunk import Vec2d


class Pitch:

    def __init__(self, space) -> None:
        # Define wall positions
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
        self.FIELD_TOP = 0
        self.FIELD_BOTTOM = 600
        self.FIELD_LEFT = 50
        self.FIELD_RIGHT = 750
        self.FIELD_CENTER_Y = (self.FIELD_TOP + self.FIELD_BOTTOM) // 2


        self.goal_left = Goal(space = space, position = (52, 300))
        self.goal_right = Goal(space = space, position = (778, 300))

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


class Goal:
    def __init__(self, space, position, angle_degrees=0.0, width=30, height=120, post_thickness=5, collision_type=10, name="goal"):
        self.GOAL_COLOR = pygame.Color("grey")
        self.space = space
        self.position = position
        self.angle = math.radians(angle_degrees)
        self.width = width
        self.height = height
        self.name = name

        self.collision_type = collision_type
        self.goal_scored = False

        # Rotation matrix
        self.direction = Vec2d(1, 0).rotated(self.angle)
        self.perp = self.direction.perpendicular()

        # Compute corners and posts
        self.post_thickness = post_thickness
        self._create_posts(post_thickness)
        self._create_sensor_area()

    def _create_posts(self, thickness):
        # Left and right post positions
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
        # Sensor rectangle (goal mouth area)
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

    def setup_collision_handler(self, ball_collision_type):
        def _goal_scored(arbiter, space, data):
            self.goal_scored = True
            print(f"⚽ Goal scored in {self.name}!")
            return False  # Do not affect physics

        handler = self.space.add_collision_handler(ball_collision_type, self.collision_type)
        handler.begin = _goal_scored

    def reset(self):
        self.goal_scored = False

    def draw(self, surface):
        # Draw goal posts
        for post in [self.post1, self.post2]:
            start = int(post.a.x), int(post.a.y)
            end = int(post.b.x), int(post.b.y)
            pygame.draw.line(surface, self.GOAL_COLOR, start, end, int(self.post_thickness))

        # Draw goal sensor area (mouth of the goal)
        points = [(int(p.x), int(p.y)) for p in self.goal_shape.get_vertices()]
        pygame.draw.polygon(surface, self.GOAL_COLOR, points, width=1)
