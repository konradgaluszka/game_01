"""
Match Manager - Core Game State and Scoring System

This module handles the high-level match logic including:
- Goal detection and scoring
- Game timer and elapsed time tracking  
- Automatic position resets after goals
- Score display and UI rendering
- Manual reset via keyboard input
- Set-piece management (side kicks, corner kicks)

**Key Features**:
- Thread-safe goal detection with cooldown periods
- Automatic player/ball position reset after goals
- Set-piece detection and management
- Player restriction enforcement during set pieces
- Clean separation between game logic and physics
- Real-time score and timer display
"""

import pygame
import time
import threading
from typing import List, Optional

# Prevent rapid goal detection (ball bouncing in goal area)
GOAL_COOLDOWN_SECONDS = 3

class Match:
    """
    Central match manager that handles scoring, timing, and game state.
    
    **Purpose**: Manage high-level game flow separate from physics simulation
    
    **Responsibilities**:
    1. **Goal Detection**: Check ball position against goal areas each frame
    2. **Score Tracking**: Maintain goal counters for both teams
    3. **Game Timer**: Track elapsed match time for display
    4. **Reset Logic**: Restore positions after goals or manual reset
    5. **UI Display**: Render score and time information
    
    **Integration**: Called from main game loop to update match state
    and handle goal events with automatic position resets.
    """
    def __init__(self, goal1_check_fn, goal2_check_fn, ball_position_supplier, resettable_objects=[], 
                 pitch=None, all_players: Optional[List] = None):
        """
        Initialize match with goal detection functions and resettable objects.
        
        Args:
            goal1_check_fn: Function to check if ball is in left goal (team_2 scores)
            goal2_check_fn: Function to check if ball is in right goal (team_1 scores)  
            ball_position_supplier: Object with position() method returning ball location
            resettable_objects: List of objects with reset() method (ball, teams)
            pitch: Pitch object for field boundary information
            all_players: List of all players for set-piece management
        """
        # Goal detection functions (provided by pitch/goal objects)
        self.goal1_check_fn = goal1_check_fn  # Left goal - team_2 scores here
        self.goal2_check_fn = goal2_check_fn  # Right goal - team_1 scores here

        # Score tracking
        self.goal1_score = 0  # Goals scored by team_2 (in left goal)
        self.goal2_score = 0  # Goals scored by team_1 (in right goal)

        # Timing
        self.match_start_time = time.time()
        self.last_goal_time = 0  # Prevents rapid goal detection

        # UI rendering
        self._font = pygame.font.Font(None, 36)
        
        # Objects that can be reset to starting positions
        self.resettable_objects = resettable_objects
        self.ball_position_supplier = ball_position_supplier
        
        # Set-piece system
        self.set_piece_manager = None
        if pitch and all_players and ball_position_supplier:
            try:
                from game.set_piece import SetPieceManager
                field_bounds = pitch.get_field_bounds()
                self.set_piece_manager = SetPieceManager(field_bounds, ball_position_supplier, all_players)
            except ImportError:
                print("Set-piece system not available")
                self.set_piece_manager = None

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

        # Update set-piece system
        if self.set_piece_manager:
            self.set_piece_manager.update(time.time())

        # Only check goals during normal play (not during set pieces)
        if self.set_piece_manager and self.set_piece_manager.is_set_piece_active():
            return  # Skip goal detection during set pieces
            
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
        
        # Draw set-piece information
        if self.set_piece_manager:
            self.set_piece_manager.draw_set_piece_info(surface)
    
    def is_set_piece_active(self) -> bool:
        """Check if a set piece is currently active"""
        if self.set_piece_manager:
            return self.set_piece_manager.is_set_piece_active()
        return False
    
    def get_restricted_players(self) -> List:
        """Get list of players who should be restricted during set pieces"""
        if self.set_piece_manager:
            return self.set_piece_manager.get_restricted_players()
        return []
