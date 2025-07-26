import pygame
import sys
import drawing
from drawing.drawing import draw_pitch, Player, Ball

# Init
pygame.init()
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("5-a-side Football")

clock = pygame.time.Clock()
FPS = 60

players = [
    Player(300, 200, (0, 128, 255)),  # team A
    Player(500, 200, (0, 128, 255)),
    Player(300, 400, (255, 0, 0)),    # team B
    Player(500, 400, (255, 0, 0)),
]

ball = Ball(400, 300)


while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    draw_pitch(screen)

    for player in players:
        player.draw(screen)
    ball.draw(screen)

    pygame.display.flip()
    clock.tick(FPS)

