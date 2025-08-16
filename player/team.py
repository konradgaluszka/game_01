from dataclasses import dataclass

from common.Vector import Vector
from player.ball import Ball
from player.player import Player


@dataclass
class TeamAreaDimensions:
    top_left: Vector
    bottom_right: Vector


class Team:
    def __init__(self, side: str, team_area_dimensions: TeamAreaDimensions, space, color: str, ball: Ball) -> None:
        self._side = side
        self._team_area_dimensions = team_area_dimensions
        self._ball = ball
        self._space = space
        self._color = color
        self._opening_layout = self._calculate_opening_layout()
        self._players = self._create_players()

    def reset(self):
        for i in range(0, 5):
            self._players[i].reset(self._opening_layout[i])

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

    def _create_players(self):
        players = [Player(space=self._space, position=self._opening_layout[i], color=self._color, number=i) for i in
                   range(5)]
        for player in players:
            player.play(self._ball)
        return players
