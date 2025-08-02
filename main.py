import pygame
import sys
import pymunk.pygame_util
from drawing.drawing import draw_pitch, Player, Ball
import time

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


# Define wall positions
top_wall = pymunk.Segment(space.static_body, (0, 00), (800, 00), 20)
bottom_wall = pymunk.Segment(space.static_body, (0, 600), (800, 600), 20)
left_wall = pymunk.Segment(space.static_body, (0, 0), (00, 600), 20)
right_wall = pymunk.Segment(space.static_body, (800, 0), (800, 600), 20)

# Set wall elasticity (bouncy)
for wall in [top_wall, bottom_wall, left_wall, right_wall]:
    wall.elasticity = 1.0
    space.add(wall)

GOAL_WIDTH = 100
GOAL_HEIGHT = 60  # Size of goal opening
FIELD_TOP = 50
FIELD_BOTTOM = 550
FIELD_LEFT = 50
FIELD_RIGHT = 750
FIELD_CENTER_Y = (FIELD_TOP + FIELD_BOTTOM) // 2

# Goal on the left
left_goal_shape = pymunk.Poly.create_box(
    space.static_body,
    size=(5, GOAL_HEIGHT)
)
left_goal_shape.sensor = True
left_goal_shape.collision_type = 10  # custom type

# Goal on the right
right_goal_shape = pymunk.Poly.create_box(
    space.static_body,
    size=(5, GOAL_HEIGHT)
)
right_goal_shape.sensor = True
right_goal_shape.collision_type = 11
space.add(left_goal_shape, right_goal_shape)


# Player body + shape
player_mass = 20
player_moment = pymunk.moment_for_circle(1, 0, 15)
player_body = pymunk.Body(player_mass, player_moment)
player_body.position = (200, 300)  # Starting position
player_body.damping = 0.1 # Value between 0 (no damping) and 1 (no slowdown)
player_body.elasticity = 0.1

player_last_shot_time = 0
DRIBBLE_COOLDOWN = 1.0
dribble_spring_front = None
dribble_spring_left = None
dribble_spring_right = None
dribble_spring_back = None

max_speed = 200
ball_max_speed = 600

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

DRIBBLE_DISTANCE = 40
SPRING_LENGTH = 10
SHOT_STRENGTH = 500
DRIBBLE_FORCE = 500
CONTROL_RADIUS = 40
FRONT_OFFSET_LENGTH = 50

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
    now = time.time()

    # Left goal (green)
    pygame.draw.rect(screen, (0, 255, 0),
                     pygame.Rect(FIELD_LEFT - 5, FIELD_CENTER_Y - GOAL_HEIGHT // 2, 5, GOAL_HEIGHT))
    # Right goal (red)
    pygame.draw.rect(screen, (255, 0, 0), pygame.Rect(FIELD_RIGHT, FIELD_CENTER_Y - GOAL_HEIGHT // 2, 5, GOAL_HEIGHT))




    # Compute dynamic anchor in front of player
    if player_body.velocity.length > 0:
        front_offset = player_body.velocity.normalized() * FRONT_OFFSET_LENGTH/2 + (0, FRONT_OFFSET_LENGTH/2)
    else:
        front_offset = (0, FRONT_OFFSET_LENGTH/2)

    # Compute dynamic anchor in left of player
    if player_body.velocity.length > 0:
        left_offset = player_body.velocity.normalized() * FRONT_OFFSET_LENGTH/2 + (FRONT_OFFSET_LENGTH/2, 0)
    else:
        left_offset = (FRONT_OFFSET_LENGTH/2, 0)

    # Right
    if player_body.velocity.length > 0:
        right_offset = player_body.velocity.normalized() * (FRONT_OFFSET_LENGTH / 2) + (-FRONT_OFFSET_LENGTH / 2, 0)
    else:
        right_offset = (-FRONT_OFFSET_LENGTH / 2, 0)

    # Back
    if player_body.velocity.length > 0:
        back_offset = player_body.velocity.normalized() * (FRONT_OFFSET_LENGTH / 2) + (0, -FRONT_OFFSET_LENGTH / 2)
    else:
        back_offset = (0, -FRONT_OFFSET_LENGTH / 2)


    diff = ball_body.position - player_body.position
    if diff.length < DRIBBLE_DISTANCE:
        if dribble_spring_front is not None:
            dribble_spring_front.anchor_a = player_body.position + front_offset
            dribble_spring_left.anchor_a = player_body.position + left_offset
            dribble_spring_back.anchor_a = player_body.position + back_offset
            dribble_spring_right.anchor_a = player_body.position + right_offset
        if now - player_last_shot_time > DRIBBLE_COOLDOWN \
                and dribble_spring_front not in space.constraints:
            dribble_spring_front = pymunk.DampedSpring(
                space.static_body,
                ball_body,
                player_body.position + front_offset,
                (0, 0),
                rest_length=SPRING_LENGTH,
                stiffness=DRIBBLE_FORCE,
                damping=30
            )
            space.add(dribble_spring_front)

            dribble_spring_left = pymunk.DampedSpring(
                space.static_body,
                ball_body,
                player_body.position + left_offset,
                (0, 0),
                rest_length=SPRING_LENGTH,
                stiffness=DRIBBLE_FORCE,
                damping=30
            )
            space.add(dribble_spring_left)

            dribble_spring_back = pymunk.DampedSpring(
                space.static_body,
                ball_body,
                player_body.position + back_offset,
                (0, 0),
                rest_length=SPRING_LENGTH,
                stiffness=DRIBBLE_FORCE,
                damping=30
            )
            space.add(dribble_spring_back)

            dribble_spring_right = pymunk.DampedSpring(
                space.static_body,
                ball_body,
                player_body.position + right_offset,
                (0, 0),
                rest_length=SPRING_LENGTH,
                stiffness=DRIBBLE_FORCE,
                damping=30
            )
            space.add(dribble_spring_right)


        if keys[pygame.K_d]:
            direction = diff.normalized()
            if dribble_spring_front in space.constraints:
                space.remove(dribble_spring_front)
                space.remove(dribble_spring_left)
                space.remove(dribble_spring_back)
                space.remove(dribble_spring_right)
            ball_body.apply_impulse_at_local_point(direction * SHOT_STRENGTH)  # adjust power
            player_last_shot_time = time.time()

    if keys[pygame.K_UP]:
        player_body.apply_force_at_local_point((0, -force))
    if keys[pygame.K_DOWN]:
        player_body.apply_force_at_local_point((0, force))
    if keys[pygame.K_LEFT]:
        player_body.apply_force_at_local_point((-force, 0))
    if keys[pygame.K_RIGHT]:
        player_body.apply_force_at_local_point((force, 0))





    if player_body.velocity.length > max_speed:
        player_body.velocity = player_body.velocity.normalized() * max_speed

    if ball_body.velocity.length > ball_max_speed:
        ball_body.velocity = player_body.velocity.normalized() * ball_max_speed

    # Vector from player to ball
    diff = ball_body.position - player_body.position


    #for player in players:
     #   player.draw(screen)
    #ball.draw(screen)



    pygame.display.flip()
    clock.tick(FPS)

