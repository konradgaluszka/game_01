import pygame
import pymunk

ball_max_speed = 600

class Ball:
    def __init__(self, x, y, space, color) -> None:
        self.initial_x = x
        self.initial_y = y
        ball_mass = 0.05
        self.ball_body = pymunk.Body(1, pymunk.moment_for_circle(ball_mass, 0, 7))
        self.ball_body.position = (x, y)
        self.radius = 7
        ball_shape = pymunk.Circle(self.ball_body, self.radius)
        ball_shape.elasticity = 0.99
        space.add(self.ball_body, ball_shape)
        ball_shape.filter = pymunk.ShapeFilter(group=1)
        self.color = color


    def draw(self, surface):
        pygame.draw.circle(surface, self.color, (int(self.ball_body.position.x), int(self.ball_body.position.y)), self.radius)

    def simulate(self):
        if self.ball_body.velocity.length > ball_max_speed:
            self.ball_body.velocity = self.ball_body.velocity.normalized() * ball_max_speed

    def position(self):
        return self.ball_body.position

    def reset(self):
        self.ball_body.position = (self.initial_x, self.initial_y)
        self.ball_body.velocity = (0, 0)
