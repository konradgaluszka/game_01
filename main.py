import pygame
import sys
import pymunk.pygame_util
from drawing.drawing import draw_pitch, Player, Ball

pygame.font.init()
font = pygame.font.SysFont("Arial", 16)

space = pymunk.Space()
space.gravity = (0, 0)
space.damping = 0.6



# Ball body + shape
ball_mass = 0.05
ball_body = pymunk.Body(1, pymunk.moment_for_circle(ball_mass, 0, 7))
ball_body.position = (400, 300)
ball_shape = pymunk.Circle(ball_body, 7)
ball_shape.elasticity = 0.99
space.add(ball_body, ball_shape)

# # Wall (top border)
# segment = pymunk.Segment(space.static_body, (50, 50), (750, 50), 5)
# segment.elasticity = 1.0
# space.add(segment)


# Define wall positions
top_wall = pymunk.Segment(space.static_body, (50, 50), (750, 50), 20)
bottom_wall = pymunk.Segment(space.static_body, (50, 550), (750, 550), 20)
left_wall = pymunk.Segment(space.static_body, (50, 50), (50, 550), 20)
right_wall = pymunk.Segment(space.static_body, (750, 50), (750, 550), 20)

# Set wall elasticity (bouncy)
for wall in [top_wall, bottom_wall, left_wall, right_wall]:
    wall.elasticity = 1.0
    space.add(wall)


# Player body + shape
player_mass = 20
player_moment = pymunk.moment_for_circle(1, 0, 15)
player_body = pymunk.Body(player_mass, moment=99999)
player_body.position = (200, 300)  # Starting position
player_body.damping = 0.1 # Value between 0 (no damping) and 1 (no slowdown)
player_body.elasticity = 0.1


def damp_player_velocity(body, gravity, damping, dt):
    pymunk.Body.update_velocity(body, gravity, 0.96, dt)  # strong slowdown

player_body.velocity_func = damp_player_velocity


player_shape = pymunk.Circle(player_body, 15)
player_shape.elasticity = 0.7
player_shape.color = pygame.Color("blue")  # Only for drawing (optional)


player_shape.filter = pymunk.ShapeFilter(group=1)
ball_shape.filter = pymunk.ShapeFilter(group=1)




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

DRIBBLE_DISTANCE = 20
SHOT_STRENGTH = 500
DRIBBLE_FORCE = 50
CONTROL_RADIUS = 20

dribble_spring = pymunk.DampedSpring(
    player_body,             # Attach to player
    ball_body,               # Attach to ball
    (0, 0),                  # Anchor point on player (center)
    (0, 0),                  # Anchor on ball
    rest_length=25,          # How far the ball should stay in front
    stiffness=100,           # Higher = stronger pull
    damping=20               # Resistance to motion
)
space.add(dribble_spring)

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    draw_pitch(screen)

    space.step(1 / FPS)

    space.debug_draw(draw_options)

    keys = pygame.key.get_pressed()
    force = 10000.0  # Tweak this value for speed

    # Before adding a new spring, remove the old one (if it exists)
    if dribble_spring in space.constraints:
        space.remove(dribble_spring)

    # Compute dynamic anchor in front of player
    if player_body.velocity.length > 0:
        front_offset = player_body.velocity.normalized() * 20
    else:
        front_offset = (0, 0)

    dribble_spring = pymunk.DampedSpring(
        player_body,
        ball_body,
        front_offset,
        (0, 0),
        rest_length=2,
        stiffness=150,
        damping=50
    )
    slide_limit = pymunk.SlideJoint(
        player_body,
        ball_body,
        (0, 0),
        (0, 0),
        0,
        20
    )
    space.add(dribble_spring, slide_limit)


    if keys[pygame.K_UP]:
        player_body.apply_force_at_local_point((0, -force))
    if keys[pygame.K_DOWN]:
        player_body.apply_force_at_local_point((0, force))
    if keys[pygame.K_LEFT]:
        player_body.apply_force_at_local_point((-force, 0))
    if keys[pygame.K_RIGHT]:
        player_body.apply_force_at_local_point((force, 0))

    if keys[pygame.K_d]:
        # Only kick if near ball
        diff = ball_body.position - player_body.position
        if diff.length < DRIBBLE_DISTANCE:
            direction = diff.normalized()
            ball_body.apply_impulse_at_local_point(direction * SHOT_STRENGTH)  # adjust power

    max_speed = 200
    if player_body.velocity.length > max_speed:
        player_body.velocity = player_body.velocity.normalized() * max_speed

    ball_max_speed = 600
    if ball_body.velocity.length > ball_max_speed:
        player_body.velocity = player_body.velocity.normalized() * ball_max_speed

    # Vector from player to ball
    diff = ball_body.position - player_body.position

    # Only try to dribble if ball is within range and player is moving
    # if diff.length < CONTROL_RADIUS: #and player_body.velocity.length > 5:
    #     desired_position = player_body.position + player_body.velocity.normalized() * DRIBBLE_DISTANCE
    #     displacement = desired_position - ball_body.position
    #     disp_x = round(displacement.x, 1)
    #     disp_y = round(displacement.y, 1)
    #     text = font.render(f"dx: {disp_x}, dy: {disp_y}", True, (255, 255, 255))  # white text
    #     screen.blit(text, (ball_body.position.x + 20, ball_body.position.y - 20))
    #
    #     # Apply small force toward desired dribble spot
    #     ball_body.apply_force_at_world_point(displacement * DRIBBLE_FORCE, ball_body.position)

    #for player in players:
     #   player.draw(screen)
    #ball.draw(screen)



    pygame.display.flip()
    clock.tick(FPS)

