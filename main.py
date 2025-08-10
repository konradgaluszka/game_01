import pygame
import sys
import pymunk.pygame_util

from common.Vector import Vector
from player.ball import Ball
from player.team import Team, TeamAreaDimensions
from stadium.pitch import Pitch
from game.match import Match
import time

WIDTH, HEIGHT = 800, 600

pygame.font.init()
font = pygame.font.SysFont("Arial", 16)

space = pymunk.Space()
space.gravity = (0, 0)
space.damping = 0.6

pitch = Pitch(space)
ball = Ball(400, 300, space, pygame.Color("yellow"))
team_1 = Team(side="left", team_area_dimensions=TeamAreaDimensions(top_left=Vector(0, 0), bottom_right=Vector(WIDTH/2, HEIGHT)), space=space, color="red", ball=ball)
team_2 = Team(side="right", team_area_dimensions=TeamAreaDimensions(top_left=Vector(WIDTH/2, 0), bottom_right=Vector(WIDTH, HEIGHT)), space=space, color="blue", ball=ball)
all_players = team_1.players() + team_2.players()
controlled_player = all_players[0]
match = Match(pitch.goal_left.is_ball_inside_goal, pitch.goal_right.is_ball_inside_goal, ball,
              resettable_objects=[ball, team_1, team_2])


# Init
pygame.init()

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

        if event.type == pygame.KEYDOWN:
            if pygame.K_0 <= event.key <= pygame.K_9:
                # Convert key to number (0-9)
                active_player_index = event.key - pygame.K_0
                controlled_player = all_players[active_player_index]

    # DRAWING
    pitch.draw_pitch(screen)
    ball.draw(screen)

    # CONTROL
    space.step(1 / FPS)

    keys = pygame.key.get_pressed()
    now = time.time()

    # SIMULATION
    team_1.simulate()
    team_2.simulate()
    ball.simulate()
    match.update(keys)

    if keys[pygame.K_p]:
        space.debug_draw(draw_options)

    controlled_player.control(keys)

    match.draw(screen)
    team_1.draw(screen)
    team_2.draw(screen)

    pygame.display.flip()
    clock.tick(FPS)
