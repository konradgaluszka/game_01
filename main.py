import pygame
import sys
import pymunk.pygame_util
from player.ball import Ball
from player.player import Player
from stadium.pitch import Pitch
import time

pygame.font.init()
font = pygame.font.SysFont("Arial", 16)

space = pymunk.Space()
space.gravity = (0, 0)
space.damping = 0.6

pitch = Pitch(space)
ball = Ball(400, 300, space, pygame.Color("yellow"))
player1 = Player(space, 200, 300, pygame.Color("blue"))
player2 = Player(space, 600, 300, pygame.Color("red"))
player3 = Player(space, 400, 500, pygame.Color("black"))
controlled_player = None
player1.play(ball)
player2.play(ball)
player3.play(ball)

# Init
pygame.init()
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("5-a-side Football")

# Draw helper
draw_options = pymunk.pygame_util.DrawOptions(screen)

clock = pygame.time.Clock()
FPS = 60

while True:
    # EXIT
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    # DRAWING
    pitch.draw_pitch(screen)
    ball.draw(screen)
    player1.draw(screen)
    player2.draw(screen)
    player3.draw(screen)

    # CONTROL
    space.step(1 / FPS)

    keys = pygame.key.get_pressed()
    now = time.time()

    # SIMULATION
    player1.simulate()
    player2.simulate()
    player3.simulate()
    ball.simulate()

    if keys[pygame.K_p]:
        space.debug_draw(draw_options)

    if keys[pygame.K_1]:
        controlled_player = player1

    if keys[pygame.K_2]:
        controlled_player = player2

    if keys[pygame.K_3]:
        controlled_player = player3

    if keys[pygame.K_0]:
        controlled_player = None

    if controlled_player is not None:
        controlled_player.control(keys)

    pygame.display.flip()
    clock.tick(FPS)
