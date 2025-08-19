"""
Soccer Player Physics and Behavior System

This module implements individual player physics, movement, ball interactions,
and advanced soccer mechanics like dribbling, shooting, and passing.

**Core Features**:
1. **Physics Integration**: Uses pymunk for realistic movement and collisions
2. **Ball Interaction**: Sophisticated dribbling system with spring-based ball control  
3. **Shooting & Passing**: Direction-based kicks with power control
4. **Movement Control**: Keyboard input handling with velocity limits
5. **Visual Rendering**: Player drawing with jersey numbers

**Technical Details**:
- Players are circular physics bodies (radius 15) with mass and elasticity
- Ball dribbling uses 4 spring constraints for natural ball control
- Shooting system uses impulse-based ball physics
- Movement forces are applied each frame based on input
- Cooldowns prevent spam shooting/passing
"""

import math
import pygame
import pymunk
import time
from pymunk import Vec2d

from common.Vector import Vector


class Player:
    """
    Individual soccer player with physics, controls, and ball interaction.
    
    **Purpose**: Represent a single player with realistic soccer mechanics
    
    **Key Systems**:
    1. **Physics Body**: Circular collision body with mass, velocity, damping
    2. **Ball Control**: Spring-based dribbling system for natural ball handling
    3. **Action System**: Shooting, passing with directional control and cooldowns
    4. **Movement**: Force-based movement with velocity limits and damping
    5. **Visual**: Rendering with team colors and player numbers
    
    **Ball Interaction**:
    - Automatic dribbling when close to ball (< 30 units)
    - 4-spring system (front, back, left, right) for smooth ball control
    - Shooting releases springs and applies impulse to ball
    - Cooldown system prevents rapid-fire actions
    """
    def __init__(self, space, position: Vector, color, number: int):
        self.space = space
        self.pos = pygame.Vector2(position.x, position.y)
        self.color = color
        self.radius = 15
        self._number = number
        # Player body + shape
        player_mass = 20
        player_moment = pymunk.moment_for_circle(1, 0, 15)
        self.player_body = pymunk.Body(player_mass, player_moment)
        self.player_body.position = (position.x, position.y)  # Starting position
        self.player_body.damping = 0.1  # Value between 0 (no damping) and 1 (no slowdown)
        self.player_body.elasticity = 0.1

        self.player_last_shot_time = 0
        self.DRIBBLE_COOLDOWN = 1.0
        self.dribble_spring_front = None
        self.dribble_spring_left = None
        self.dribble_spring_right = None
        self.dribble_spring_back = None

        self.max_speed = 200

        self.DRIBBLE_DISTANCE = 30
        self.SPRING_LENGTH = 10
        self.SHOT_STRENGTH = 500
        self.MAX_PASS_STRENGTH = 300
        self.DRIBBLE_FORCE = 500
        self.CONTROL_RADIUS = 40
        self.FRONT_OFFSET_LENGTH = 50

        self.force = 10000.0  # Tweak this value for speed
        self.ball = None
        self._last_velocity = Vector(0, 0)

        def damp_player_velocity(body, gravity, damping, dt):
            pymunk.Body.update_velocity(body, gravity, 0.96, dt)  # strong slowdown

        self.player_body.velocity_func = damp_player_velocity

        player_shape = pymunk.Circle(self.player_body, 15)
        player_shape.elasticity = 0.7
        player_shape.color = pygame.Color("blue")  # Only for drawing (optional)

        player_shape.filter = pymunk.ShapeFilter(group=1)

        space.add(self.player_body, player_shape)

    def position(self) -> Vector:
        return Vector(self.player_body.position.x, self.player_body.position.y)
    
    def has_ball_control(self) -> bool:
        """
        Check if this player currently has control of the ball based on active dribble springs.
        
        Returns:
            bool: True if player has active ball control springs
        """
        return (self.dribble_spring_front is not None and 
                self.dribble_spring_front in self.space.constraints)

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

        if self.player_body.velocity.length > 0:
            self._last_velocity = Vector(self.player_body.velocity.x, self.player_body.velocity.y)

    def control(self, keys, teammates_positions):
        """Legacy keyboard control method - translates pygame keys to actions"""
        # Translate keyboard input to generic actions
        move_up = keys[pygame.K_UP]
        move_down = keys[pygame.K_DOWN] 
        move_left = keys[pygame.K_LEFT]
        move_right = keys[pygame.K_RIGHT]
        shoot = keys[pygame.K_d]
        pass_ball = keys[pygame.K_s]
        
        # Use generic control method
        self.apply_actions(move_up, move_down, move_left, move_right, shoot, pass_ball, teammates_positions)
    
    def apply_actions(self, move_up, move_down, move_left, move_right, shoot, pass_ball, teammates_positions):
        """
        Generic player control method that applies movement and actions.
        
        Args:
            move_up, move_down, move_left, move_right: Boolean movement directions
            shoot: Boolean shoot action
            pass_ball: Boolean pass action  
            teammates_positions: List of teammate positions for passing
        """
        # Apply movement forces
        if move_up:
            self.player_body.apply_force_at_local_point((0, -self.force))
        if move_down:
            self.player_body.apply_force_at_local_point((0, self.force))
        if move_left:
            self.player_body.apply_force_at_local_point((-self.force, 0))
        if move_right:
            self.player_body.apply_force_at_local_point((self.force, 0))

        # Apply ball actions if player is close to ball
        diff = self.ball.ball_body.position - self.player_body.position
        if diff.length < self.DRIBBLE_DISTANCE:
            now = time.time()
            if now - self.player_last_shot_time > self.DRIBBLE_COOLDOWN:
                if shoot:
                    self._shoot(diff)
                elif pass_ball:
                    self._handover(teammates_positions=teammates_positions)

    def remove_ball_springs(self):
        if self.dribble_spring_front is not None and self.dribble_spring_front in self.space.constraints:
            self.space.remove(self.dribble_spring_front)
        if self.dribble_spring_left is not None and self.dribble_spring_left in self.space.constraints:
            self.space.remove(self.dribble_spring_left)
        if self.dribble_spring_back is not None and self.dribble_spring_back in self.space.constraints:
            self.space.remove(self.dribble_spring_back)
        if self.dribble_spring_right is not None and self.dribble_spring_right in self.space.constraints:
            self.space.remove(self.dribble_spring_right)

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
        else:
            self.remove_ball_springs()


    def draw(self, surface):
        pygame.draw.circle(surface, self.color, (int(self.player_body.position.x), int(self.player_body.position.y)), self.radius)
        font = pygame.font.Font(None, 24)  # default font, size 24
        text_surface = font.render(str(self._number), True, (255, 255, 255))
        text_rect = text_surface.get_rect(center=(int(self.player_body.position.x), int(self.player_body.position.y)))
        surface.blit(text_surface, text_rect)

    def reset(self, initial_position: Vector):
        self.player_body.position = (initial_position.x, initial_position.y)
        self.player_body.velocity = (0, 0)
        self.remove_ball_springs()

    def _handover(self, teammates_positions):
        if self._last_velocity.length() <= 0:
            return None  # undefined orientation

        f = self._last_velocity.normalize()

        best = None
        best_dot = -float("inf")
        best_dist2 = float("inf")


        for p in teammates_positions:
            d = p - self.position()
            if d.length_squared() == 0:
                continue
            v = d.normalize()
            dot = f.dot(v)  # cosine of angle between facing and direction to p

            # pick the largest dot (smallest angle). break ties by nearest distance
            dist2 = d.length_squared()
            if (dot > best_dot) or (math.isclose(dot, best_dot, rel_tol=1e-9) and dist2 < best_dist2):
                best, best_dot, best_dist2 = Vec2d(p.x, p.y), dot, dist2

        if best is None:
            return None

        d = best - self.player_body.position

        self.remove_ball_springs()
        self.ball.ball_body.apply_impulse_at_local_point(d * self.MAX_PASS_STRENGTH/100)  # adjust power
        self.player_last_shot_time = time.time()



    def _shoot(self, diff):
        direction = diff.normalized()
        self.remove_ball_springs()
        self.ball.ball_body.apply_impulse_at_local_point(direction * self.SHOT_STRENGTH)  # adjust power
        self.player_last_shot_time = time.time()
