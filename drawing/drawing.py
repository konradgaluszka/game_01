import pygame

def draw_pitch(surface):
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
    pygame.draw.rect(surface, WHITE, pygame.Rect(375, 45, 50, 10))   # Top goal
    pygame.draw.rect(surface, WHITE, pygame.Rect(375, 545, 50, 10))  # Bottom goal


class Player:
    def __init__(self, x, y, color):
        self.pos = pygame.Vector2(x, y)
        self.color = color
        self.radius = 15

    def draw(self, surface):
        pygame.draw.circle(surface, self.color, (int(self.pos.x), int(self.pos.y)), self.radius)

class Ball:
    def __init__(self, x, y):
        self.pos = pygame.Vector2(x, y)
        self.color = (255, 255, 255)
        self.radius = 10

    def draw(self, surface):
        pygame.draw.circle(surface, self.color, (int(self.pos.x), int(self.pos.y)), self.radius)
