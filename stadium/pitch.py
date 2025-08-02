import pygame
import pymunk

class Pitch:

    def __init__(self, space) -> None:
        # Define wall positions
        top_wall = pymunk.Segment(space.static_body, (0, 00), (800, 00), 20)
        bottom_wall = pymunk.Segment(space.static_body, (0, 600), (800, 600), 20)
        left_wall = pymunk.Segment(space.static_body, (0, 0), (00, 600), 20)
        right_wall = pymunk.Segment(space.static_body, (800, 0), (800, 600), 20)

        # Set wall elasticity (bouncy)
        for wall in [top_wall, bottom_wall, left_wall, right_wall]:
            wall.elasticity = 1.0
            space.add(wall)

        self.GOAL_WIDTH = 100
        self.GOAL_HEIGHT = 60  # Size of goal opening
        self.FIELD_TOP = 50
        self.FIELD_BOTTOM = 550
        self.FIELD_LEFT = 50
        self.FIELD_RIGHT = 750
        self.FIELD_CENTER_Y = (self.FIELD_TOP + self.FIELD_BOTTOM) // 2

        # Goal on the left
        left_goal_shape = pymunk.Poly.create_box(
            space.static_body,
            size=(5, self.GOAL_HEIGHT)
        )
        left_goal_shape.sensor = True
        left_goal_shape.collision_type = 10  # custom type

        # Goal on the right
        right_goal_shape = pymunk.Poly.create_box(
            space.static_body,
            size=(5, self.GOAL_HEIGHT)
        )
        right_goal_shape.sensor = True
        right_goal_shape.collision_type = 11
        space.add(left_goal_shape, right_goal_shape)

    def draw_pitch(self, surface):
        GREEN = (34, 139, 34)
        WHITE = (255, 255, 255)

        surface.fill(GREEN)

        # Outer boundaries
        pygame.draw.rect(surface, WHITE, pygame.Rect(50, 50, 700, 500), 5)

        # Midfield line
        pygame.draw.line(surface, WHITE, (400, 50), (400, 550), 3)

        # Center circle
        pygame.draw.circle(surface, WHITE, (400, 300), 60, 3)

        # Goals
        pygame.draw.rect(surface, WHITE, pygame.Rect(375, 45, 50, 10))  # Top goal
        pygame.draw.rect(surface, WHITE, pygame.Rect(375, 545, 50, 10))  # Bottom goal

        # Left goal (green)
        pygame.draw.rect(surface, (0, 255, 0),
                         pygame.Rect(self.FIELD_LEFT - 5, self.FIELD_CENTER_Y - self.GOAL_HEIGHT // 2, 5, self.GOAL_HEIGHT))
        # Right goal (red)
        pygame.draw.rect(surface, (255, 0, 0),
                         pygame.Rect(self.FIELD_RIGHT, self.FIELD_CENTER_Y - self.GOAL_HEIGHT // 2, 5, self.GOAL_HEIGHT))


