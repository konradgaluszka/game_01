import pygame
import time
import threading

GOAL_COOLDOWN_SECONDS = 3
class Match:
    def __init__(self, goal1_check_fn, goal2_check_fn, ball_position_supplier, resettable_objects=[]):
        self.goal1_check_fn = goal1_check_fn  # e.g., check if ball is in left goal
        self.goal2_check_fn = goal2_check_fn  # e.g., check if ball is in right goal

        self.goal1_score = 0
        self.goal2_score = 0

        self.match_start_time = time.time()
        self.last_goal_time = 0

        self._font = pygame.font.Font(None, 36)
        self.resettable_objects = resettable_objects
        self.ball_position_supplier = ball_position_supplier

    def reset(self):
        self.goal1_score = 0
        self.goal2_score = 0
        self.match_start_time = time.time()
        self._reset_positions_only()

    def _reset_positions_only(self):
        for resettable_object in self.resettable_objects:
            if hasattr(resettable_object, "reset") and callable(getattr(resettable_object, "reset")):
                resettable_object.reset()
            else:
                print(f"got object that's not resettable!")

    def restart(self):
        yield

    def update(self, keys):
        # Handle reset on backspace
        if keys[pygame.K_BACKSPACE]:
            self.reset()

        # Check goals (once per frame)
        if time.time() - self.last_goal_time < GOAL_COOLDOWN_SECONDS:
            return

        ball_position = self.ball_position_supplier.position()
        if self.goal1_check_fn(ball_position):
            self._update_goal("goal1_score")
        elif self.goal2_check_fn(ball_position):
            self._update_goal("goal2_score")

    def _update_goal(self, goal_score_field):
        setattr(self,goal_score_field, getattr(self, goal_score_field) + 1)
        self.last_goal_time = time.time()
        threading.Timer(2.0, self._reset_positions_only).start()

    def get_elapsed_time_str(self):
        elapsed = int(time.time() - self.match_start_time)
        minutes = elapsed // 60
        seconds = elapsed % 60
        return f"{minutes:02d}:{seconds:02d}"

    def draw(self, surface):
        # Display time and score centered at the top
        time_str = self.get_elapsed_time_str()
        score_str = f"{self.goal1_score} : {self.goal2_score}"
        display_str = f"{time_str}   {score_str}"

        text_surface = self._font.render(display_str, True, (255, 255, 255))
        text_rect = text_surface.get_rect(center=(surface.get_width() // 2, 20))
        surface.blit(text_surface, text_rect)
