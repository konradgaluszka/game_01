"""
Main game file for 5-a-side Soccer Game with AI Support

This is the entry point for the soccer game that features:
- Human-controlled team_1 (red, left side) via keyboard input
- AI-controlled team_2 (blue, right side) when trained model is available
- Real-time physics simulation using pymunk
- Goal detection and scoring system
- Optional AI training integration

The game runs at 60 FPS and handles both human input and AI decision making
in a single game loop, providing a hybrid human vs AI soccer experience.
"""

import pygame
import sys
import pymunk.pygame_util
import os
import time
import argparse

# Core game components
from common.Vector import Vector
from player.ball import Ball
from player.team import Team, TeamAreaDimensions
from stadium.pitch import Pitch
from game.match import Match

# AI system integration with graceful fallback
# This allows the game to run without AI dependencies installed
try:
    from ai.ai_controller import AIController
    AI_AVAILABLE = True
except ImportError:
    AIController = None
    AI_AVAILABLE = False
    print("AI dependencies not installed. Team_2 will run without AI control.")

# Opponent AI system
try:
    from ai.opponent_ai import OpponentAI
    OPPONENT_AI_AVAILABLE = True
except ImportError:
    OpponentAI = None
    OPPONENT_AI_AVAILABLE = False
    print("OpponentAI not available.")

# === COMMAND LINE ARGUMENTS ===
parser = argparse.ArgumentParser(description="5-a-side Soccer Game with AI Support")
parser.add_argument("--team1-control", choices=["human", "model", "opponent_ai"], default="human",
                    help="Control method for team_1 (left/red): human=keyboard, model=trained_model, opponent_ai=rule_based_AI")
parser.add_argument("--team2-control", choices=["human", "model", "opponent_ai", "none"], default="opponent_ai",
                    help="Control method for team_2 (right/blue): human=keyboard, model=trained_model, opponent_ai=rule_based_AI, none=no_control")
parser.add_argument("--team1-model", type=str, default="ai/models/quick_iterative_final.zip",
                    help="Path to trained model for team_1 (if using model control)")
parser.add_argument("--team2-model", type=str, default="ai/models/quick_iterative_final.zip",
                    help="Path to trained model for team_2 (if using model control)")
args = parser.parse_args()

# === GAME CONFIGURATION ===
# Screen dimensions for the soccer field
WIDTH, HEIGHT = 800, 600

# === PYGAME INITIALIZATION ===
pygame.font.init()
font = pygame.font.SysFont("Arial", 16)

# === PHYSICS WORLD SETUP ===
# Create pymunk physics space with custom properties for soccer gameplay
space = pymunk.Space()
space.gravity = (0, 0)      # No gravity - top-down 2D soccer view
space.damping = 0.6         # Air resistance to prevent infinite sliding

# === GAME OBJECTS CREATION ===
# Create the soccer field with goals and boundaries
pitch = Pitch(space)

# Create the ball at center field
ball = Ball(400, 300, space, pygame.Color("yellow"))

# Create team_1 (human-controlled, red, left side)
team_1 = Team(
    side="left", 
    team_area_dimensions=TeamAreaDimensions(
        top_left=Vector(0, 0), 
        bottom_right=Vector(WIDTH/2, HEIGHT)
    ), 
    space=space, 
    color="red", 
    ball=ball
)

# Create team_2 (AI-controlled, blue, right side)
team_2 = Team(
    side="right", 
    team_area_dimensions=TeamAreaDimensions(
        top_left=Vector(WIDTH/2, 0), 
        bottom_right=Vector(WIDTH, HEIGHT)
    ), 
    space=space, 
    color="blue", 
    ball=ball
)

# Combine all players for easy access
all_players = team_1.players() + team_2.players()

# Set which team the human controls based on arguments
controlled_team = None
if args.team1_control == "human":
    controlled_team = team_1
elif args.team2_control == "human":
    controlled_team = team_2

# Create match manager to handle scoring and game state
match = Match(
    pitch.goal_left.is_ball_inside_goal,    # Team_1's goal (team_2 scores here)
    pitch.goal_right.is_ball_inside_goal,   # Team_2's goal (team_1 scores here)
    ball,
    resettable_objects=[ball, team_1, team_2]
)

# === AI CONTROLLER SETUP ===
# Initialize AI controllers based on command line arguments
team_1_controller = None
team_2_controller = None
team_1_opponent_ai = None
team_2_opponent_ai = None

# Setup team_1 control
if args.team1_control == "model" and AI_AVAILABLE:
    model_path = args.team1_model if os.path.exists(args.team1_model) else None
    team_1_controller = AIController(model_path)
    if model_path:
        print(f"Team_1 using trained model: {model_path}")
    else:
        print("Team_1 model not found, team_1 will have no AI control")
elif args.team1_control == "opponent_ai" and OPPONENT_AI_AVAILABLE:
    team_1_opponent_ai = OpponentAI(team_side="left", field_width=WIDTH, field_height=HEIGHT)
    print("Team_1 using rule-based OpponentAI")
elif args.team1_control == "human":
    print("Team_1 controlled by human (keyboard)")

# Setup team_2 control  
if args.team2_control == "model" and AI_AVAILABLE:
    model_path = args.team2_model if os.path.exists(args.team2_model) else None
    team_2_controller = AIController(model_path)
    if model_path:
        print(f"Team_2 using trained model: {model_path}")
    else:
        print("Team_2 model not found, team_2 will have no AI control")
elif args.team2_control == "opponent_ai" and OPPONENT_AI_AVAILABLE:
    team_2_opponent_ai = OpponentAI(team_side="right", field_width=WIDTH, field_height=HEIGHT)
    print("Team_2 using rule-based OpponentAI")
elif args.team2_control == "human":
    print("Team_2 controlled by human (keyboard)")
else:
    print("Team_2 has no control (stationary)")


# === PYGAME WINDOW SETUP ===
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("5-a-side Football")

# Physics debug drawing helper (press P to toggle)
draw_options = pymunk.pygame_util.DrawOptions(screen)

# Game timing control
clock = pygame.time.Clock()
FPS = 60

# === MAIN GAME LOOP ===
"""
Main game loop that runs at 60 FPS and handles:
1. Event processing (quit, keyboard input)
2. Physics simulation (pymunk space step)
3. Game object updates (players, ball, match state)
4. Human player control (keyboard input to team_1)
5. AI player control (neural network decisions for team_2)
6. Rendering (draw all game objects to screen)
"""
while True:
    # === EVENT HANDLING ===
    # Check for quit events and window closing
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    # === PHYSICS SIMULATION ===
    # Advance physics simulation by one frame (1/60th second)
    space.step(1 / FPS)

    # Get current keyboard state for player control
    keys = pygame.key.get_pressed()
    now = time.time()

    # === GAME OBJECT UPDATES ===
    # Update all game objects with physics and AI behaviors
    team_1.simulate()           # Update team_1 player physics and behaviors
    team_2.simulate()           # Update team_2 player physics and behaviors  
    ball.simulate()             # Update ball physics and interactions
    match.update(keys)          # Update match state (scoring, resets, timer)

    # === PLAYER CONTROL ===
    # Control teams based on configuration
    
    # Human control (if any team is human-controlled)
    if controlled_team:
        controlled_team.control(keys)
    
    # Team_1 AI control
    if team_1_controller:
        # Trained model control for team_1
        team_1_controller.control_team(team_1, ball, all_players, match)
    elif team_1_opponent_ai:
        # Rule-based AI control for team_1
        team_1_opponent_ai.control_team(team_1.players(), ball, team_2.players())
    
    # Team_2 AI control
    if team_2_controller:
        # Trained model control for team_2
        team_2_controller.control_team(team_2, ball, all_players, match)
    elif team_2_opponent_ai:
        # Rule-based AI control for team_2
        team_2_opponent_ai.control_team(team_2.players(), ball, team_1.players())

    # === RENDERING ===
    # Draw all game elements to the screen in correct order
    pitch.draw_pitch(screen)    # Draw field, goals, and boundaries (background)
    ball.draw(screen)           # Draw the ball
    team_1.draw(screen)         # Draw red team players
    team_2.draw(screen)         # Draw blue team players
    match.draw(screen)          # Draw UI (score, timer)

    # Optional physics debug visualization (press P to toggle)
    if keys[pygame.K_p]:
        space.debug_draw(draw_options)

    # Update display and maintain 60 FPS
    pygame.display.flip()
    clock.tick(FPS)
