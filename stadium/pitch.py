import pygame
import pymunk

from stadium.goal import Goal


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


        self.goal_left = Goal(space = space, position = (52, 300), orientation="right")
        self.goal_right = Goal(space = space, position = (748, 300), orientation="left")

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



