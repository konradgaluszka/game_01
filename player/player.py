import pygame
import pymunk
import time


class Player:
    def __init__(self, space, x, y, color):
        self.space = space
        self.pos = pygame.Vector2(x, y)
        self.color = color
        self.radius = 15
        # Player body + shape
        player_mass = 20
        player_moment = pymunk.moment_for_circle(1, 0, 15)
        self.player_body = pymunk.Body(player_mass, player_moment)
        self.player_body.position = (x, y)  # Starting position
        self.player_body.damping = 0.1  # Value between 0 (no damping) and 1 (no slowdown)
        self.player_body.elasticity = 0.1

        self.player_last_shot_time = 0
        self.DRIBBLE_COOLDOWN = 1.0
        self.dribble_spring_front = None
        self.dribble_spring_left = None
        self.dribble_spring_right = None
        self.dribble_spring_back = None

        self.max_speed = 200

        self.DRIBBLE_DISTANCE = 40
        self.SPRING_LENGTH = 10
        self.SHOT_STRENGTH = 500
        self.DRIBBLE_FORCE = 500
        self.CONTROL_RADIUS = 40
        self.FRONT_OFFSET_LENGTH = 50

        self.force = 10000.0  # Tweak this value for speed
        self.ball = None

        def damp_player_velocity(body, gravity, damping, dt):
            pymunk.Body.update_velocity(body, gravity, 0.96, dt)  # strong slowdown

        self.player_body.velocity_func = damp_player_velocity

        player_shape = pymunk.Circle(self.player_body, 15)
        player_shape.elasticity = 0.7
        player_shape.color = pygame.Color("blue")  # Only for drawing (optional)

        player_shape.filter = pymunk.ShapeFilter(group=1)

        space.add(self.player_body, player_shape)

    def play(self, ball):
        self.ball = ball

    def simulate(self):
        now = time.time()

        # Compute dynamic anchor in front of player
        if self.player_body.velocity.length > 0:
            front_offset = self.player_body.velocity.normalized() * self.FRONT_OFFSET_LENGTH / 2 + (0, self.FRONT_OFFSET_LENGTH / 2)
        else:
            front_offset = (0, self.FRONT_OFFSET_LENGTH / 2)

        # Compute dynamic anchor in left of player
        if self.player_body.velocity.length > 0:
            left_offset = self.player_body.velocity.normalized() * self.FRONT_OFFSET_LENGTH / 2 + (self.FRONT_OFFSET_LENGTH / 2, 0)
        else:
            left_offset = (self.FRONT_OFFSET_LENGTH / 2, 0)

        # Right
        if self.player_body.velocity.length > 0:
            right_offset = self.player_body.velocity.normalized() * (self.FRONT_OFFSET_LENGTH / 2) + (-self.FRONT_OFFSET_LENGTH / 2, 0)
        else:
            right_offset = (-self.FRONT_OFFSET_LENGTH / 2, 0)

        # Back
        if self.player_body.velocity.length > 0:
            back_offset = self.player_body.velocity.normalized() * (self.FRONT_OFFSET_LENGTH / 2) + (0, -self.FRONT_OFFSET_LENGTH / 2)
        else:
            back_offset = (0, -self.FRONT_OFFSET_LENGTH / 2)

        if self.ball is not None:
            self.simulate_ball_interaction(back_offset, front_offset, left_offset, now, right_offset)


        if self.player_body.velocity.length > self.max_speed:
            self.player_body.velocity = self.player_body.velocity.normalized() * self.max_speed

    def control(self, keys):
        now = time.time()
        diff = self.ball.ball_body.position - self.player_body.position
        if keys[pygame.K_UP]:
            self.player_body.apply_force_at_local_point((0, -self.force))
        if keys[pygame.K_DOWN]:
            self.player_body.apply_force_at_local_point((0, self.force))
        if keys[pygame.K_LEFT]:
            self.player_body.apply_force_at_local_point((-self.force, 0))
        if keys[pygame.K_RIGHT]:
            self.player_body.apply_force_at_local_point((self.force, 0))

        if diff.length < self.DRIBBLE_DISTANCE:
            if now - self.player_last_shot_time > self.DRIBBLE_COOLDOWN \
                    and self.dribble_spring_front not in self.space.constraints:
                if keys[pygame.K_d]:
                    direction = diff.normalized()
                    if self.dribble_spring_front in self.space.constraints:
                        self.space.remove(self.dribble_spring_front)
                        self.space.remove(self.dribble_spring_left)
                        self.space.remove(self.dribble_spring_back)
                        self.space.remove(self.dribble_spring_right)
                    self.ball.ball_body.apply_impulse_at_local_point(direction * self.SHOT_STRENGTH)  # adjust power
                    self.player_last_shot_time = time.time()
    def simulate_ball_interaction(self, back_offset, front_offset, left_offset, now, right_offset):
        diff = self.ball.ball_body.position - self.player_body.position
        if diff.length < self.DRIBBLE_DISTANCE:
            if self.dribble_spring_front is not None:
                self.dribble_spring_front.anchor_a = self.player_body.position + front_offset
                self.dribble_spring_left.anchor_a = self.player_body.position + left_offset
                self.dribble_spring_back.anchor_a = self.player_body.position + back_offset
                self.dribble_spring_right.anchor_a = self.player_body.position + right_offset
            if now - self.player_last_shot_time > self.DRIBBLE_COOLDOWN \
                    and self.dribble_spring_front not in self.space.constraints:
                self.dribble_spring_front = pymunk.DampedSpring(
                    self.space.static_body,
                    self.ball.ball_body,
                    self.player_body.position + front_offset,
                    (0, 0),
                    rest_length=self.SPRING_LENGTH,
                    stiffness=self.DRIBBLE_FORCE,
                    damping=30
                )
                self.space.add(self.dribble_spring_front)

                self.dribble_spring_left = pymunk.DampedSpring(
                    self.space.static_body,
                    self.ball.ball_body,
                    self.player_body.position + left_offset,
                    (0, 0),
                    rest_length=self.SPRING_LENGTH,
                    stiffness=self.DRIBBLE_FORCE,
                    damping=30
                )
                self.space.add(self.dribble_spring_left)

                self.dribble_spring_back = pymunk.DampedSpring(
                    self.space.static_body,
                    self.ball.ball_body,
                    self.player_body.position + back_offset,
                    (0, 0),
                    rest_length=self.SPRING_LENGTH,
                    stiffness=self.DRIBBLE_FORCE,
                    damping=30
                )
                self.space.add(self.dribble_spring_back)

                self.dribble_spring_right = pymunk.DampedSpring(
                    self.space.static_body,
                    self.ball.ball_body,
                    self.player_body.position + right_offset,
                    (0, 0),
                    rest_length=self.SPRING_LENGTH,
                    stiffness=self.DRIBBLE_FORCE,
                    damping=30
                )
                self.space.add(self.dribble_spring_right)



    def draw(self, surface):
        pygame.draw.circle(surface, self.color, (int(self.player_body.position.x), int(self.player_body.position.y)), self.radius)