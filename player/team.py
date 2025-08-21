"""
Team Management System

This module handles team creation, player organization, and team-level behaviors.
It manages the 5 players per team, their formation, and provides the interface
for controlling multiple players as a cohesive unit.

**Key Features**:
- Automatic player positioning in realistic soccer formations
- Team-wide simulation and rendering
- Player switching and control management
- Starting position management and resets
- Side-specific formations (left vs right team layouts)
"""

from dataclasses import dataclass
import pygame

from common.Vector import Vector
from player.ball import Ball
from player.player import Player


@dataclass
class TeamAreaDimensions:
    """
    Defines the rectangular area assigned to a team on the field.
    
    Used to calculate player starting positions and constrain team movement
    to their half of the field during setup.
    """
    top_left: Vector      # Upper-left corner of team's area
    bottom_right: Vector  # Lower-right corner of team's area


class Team:
    """
    Manages a team of 5 soccer players with formation, control, and coordination.
    
    **Purpose**: Organize players into a cohesive team unit with proper formation
    
    **Key Responsibilities**:
    1. **Formation Management**: Calculate and maintain realistic starting positions
    2. **Player Creation**: Instantiate 5 Player objects with team properties
    3. **Control Interface**: Handle keyboard input and distribute to active player
    4. **Team Simulation**: Update all players each frame
    5. **Reset Functionality**: Restore players to starting positions
    
    **Team Layouts**:
    - Left team (red): Defensive formation facing right
    - Right team (blue): Defensive formation facing left
    - Both use 1-2-2 formation (goalkeeper, 2 defenders, 2 forwards)
    """
    def __init__(self, side: str, team_area_dimensions: TeamAreaDimensions, space, color: str, ball: Ball) -> None:
        self._side = side
        self._team_area_dimensions = team_area_dimensions
        self._ball = ball
        self._space = space
        self._color = color
        self._opening_layout = self._calculate_opening_layout()
        self._players = self._create_players()
        self._controlled_player = self._players[0]
        
        # Initialize visual selection indicators
        self._update_player_selection_visuals()

    def reset(self):
        for i in range(0, 5):
            self._players[i].reset(self._opening_layout[i])

    def control(self, keys):
        # Manual player switching with number keys (fix key detection)
        for i, key in enumerate([pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4, pygame.K_5]):
            if keys[key]:
                old_controlled = self._controlled_player
                self._controlled_player = self._players[i]
                # Update visual selection state
                if old_controlled != self._controlled_player:
                    self._update_player_selection_visuals()
        
        # Automatic player switching - switch to player who has ball control
        player_with_ball = self._get_player_with_ball_control()
        if player_with_ball is not None and player_with_ball != self._controlled_player:
            self._controlled_player = player_with_ball
            self._update_player_selection_visuals()

        # Control the selected player with all teammates positions
        teammates_positions = [pl.position() for pl in self._players if pl != self._controlled_player]
        self._controlled_player.control(keys, teammates_positions)
    
    def _get_player_with_ball_control(self):
        """
        Find which player currently has ball control based on dribble springs.
        
        Returns:
            Player or None: The player with ball control, or None if no player has control
        """
        for player in self._players:
            if player.has_ball_control():
                return player
        return None
    
    def _update_player_selection_visuals(self):
        """
        Update visual indicators for which player is currently selected.
        """
        # Clear all player selection states
        for player in self._players:
            if hasattr(player, 'renderer') and player.renderer:
                player.renderer.set_selected(False)
                player.renderer.set_highlighted(False)
        
        # Set current controlled player as selected
        if self._controlled_player and hasattr(self._controlled_player, 'renderer'):
            self._controlled_player.renderer.set_selected(True)
            
        # Highlight player with ball control if different from controlled player
        player_with_ball = self._get_player_with_ball_control()
        if (player_with_ball and player_with_ball != self._controlled_player and 
            hasattr(player_with_ball, 'renderer')):
            player_with_ball.renderer.set_highlighted(True)

    def players(self):
        return self._players

    def draw(self, surface):
        for player in self._players:
            player.draw(surface)

    def _calculate_opening_layout(self):
        area_dimensions = self._team_area_dimensions.bottom_right - self._team_area_dimensions.top_left
        if self._side == "left":
            return [
                Vector(self._team_area_dimensions.top_left.x + area_dimensions.x * 0.2,
                       self._team_area_dimensions.top_left.y + area_dimensions.y * 0.5),
                Vector(self._team_area_dimensions.top_left.x + area_dimensions.x * 0.4,
                       self._team_area_dimensions.top_left.y + area_dimensions.y * 0.33),
                Vector(self._team_area_dimensions.top_left.x + area_dimensions.x * 0.4,
                       self._team_area_dimensions.top_left.y + area_dimensions.y * 0.67),
                Vector(self._team_area_dimensions.top_left.x + area_dimensions.x * 0.7,
                       self._team_area_dimensions.top_left.y + area_dimensions.y * 0.33),
                Vector(self._team_area_dimensions.top_left.x + area_dimensions.x * 0.7,
                       self._team_area_dimensions.top_left.y + area_dimensions.y * 0.67)
            ]
        elif self._side == "right":
            return [
                Vector(self._team_area_dimensions.top_left.x + area_dimensions.x * 0.8,
                       self._team_area_dimensions.top_left.y + area_dimensions.y * 0.5),
                Vector(self._team_area_dimensions.top_left.x + area_dimensions.x * 0.6,
                       self._team_area_dimensions.top_left.y + area_dimensions.y * 0.33),
                Vector(self._team_area_dimensions.top_left.x + area_dimensions.x * 0.6,
                       self._team_area_dimensions.top_left.y + area_dimensions.y * 0.67),
                Vector(self._team_area_dimensions.top_left.x + area_dimensions.x * 0.3,
                       self._team_area_dimensions.top_left.y + area_dimensions.y * 0.33),
                Vector(self._team_area_dimensions.top_left.x + area_dimensions.x * 0.3,
                       self._team_area_dimensions.top_left.y + area_dimensions.y * 0.67)
            ]
        else:
            raise Exception("Only right and left sides allowed during calculation of opening team layout")

    def simulate(self):
        for player in self._players:
            player.simulate()
        
        # Update visual selection indicators each frame in case ball control changes
        self._update_player_selection_visuals()

    def _create_players(self):
        players = [Player(space=self._space, position=self._opening_layout[i], color=self._color, number=i+1) for i in
                   range(5)]
        for player in players:
            player.play(self._ball)
        return players
