import pygame
import sys
import pymunk.pygame_util
from drawing.drawing import draw_pitch, Player, Ball

space = pymunk.Space()
space.gravity = (0, 0)
space.damping = 0.6


# Ball body + shape
ball_mass = 0.05
ball_body = pymunk.Body(1, pymunk.moment_for_circle(ball_mass, 0, 10))
ball_body.position = (400, 300)
ball_shape = pymunk.Circle(ball_body, 10)
ball_shape.elasticity = 0.99
space.add(ball_body, ball_shape)

# # Wall (top border)
# segment = pymunk.Segment(space.static_body, (50, 50), (750, 50), 5)
# segment.elasticity = 1.0
# space.add(segment)


# Define wall positions
top_wall = pymunk.Segment(space.static_body, (50, 50), (750, 50), 5)
bottom_wall = pymunk.Segment(space.static_body, (50, 550), (750, 550), 5)
left_wall = pymunk.Segment(space.static_body, (50, 50), (50, 550), 5)
right_wall = pymunk.Segment(space.static_body, (750, 50), (750, 550), 5)

# Set wall elasticity (bouncy)
for wall in [top_wall, bottom_wall, left_wall, right_wall]:
    wall.elasticity = 1.0
    space.add(wall)


# Player body + shape
player_mass = 10
player_body = pymunk.Body(player_mass, pymunk.moment_for_circle(1, 0, 15))
player_body.position = (200, 300)  # Starting position
player_body.damping = 0.1 # Value between 0 (no damping) and 1 (no slowdown)
def damp_player_velocity(body, gravity, damping, dt):
    pymunk.Body.update_velocity(body, gravity, 0.96, dt)  # strong slowdown

player_body.velocity_func = damp_player_velocity


player_shape = pymunk.Circle(player_body, 15)
player_shape.elasticity = 0.7
player_shape.color = pygame.Color("blue")  # Only for drawing (optional)

space.add(player_body, player_shape)



# Init
pygame.init()
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("5-a-side Football")


# Draw helper
draw_options = pymunk.pygame_util.DrawOptions(screen)

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

    space.step(1 / FPS)

    space.debug_draw(draw_options)

    keys = pygame.key.get_pressed()
    force = 7000.0  # Tweak this value for speed

    if keys[pygame.K_w]:
        player_body.apply_force_at_local_point((0, -force))
    if keys[pygame.K_s]:
        player_body.apply_force_at_local_point((0, force))
    if keys[pygame.K_a]:
        player_body.apply_force_at_local_point((-force, 0))
    if keys[pygame.K_d]:
        player_body.apply_force_at_local_point((force, 0))

    max_speed = 200
    if player_body.velocity.length > max_speed:
        player_body.velocity = player_body.velocity.normalized() * max_speed
    #for player in players:
     #   player.draw(screen)
    #ball.draw(screen)

    pygame.display.flip()
    clock.tick(FPS)

