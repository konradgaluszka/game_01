# Soccer AI Training Module

This module provides reinforcement learning capabilities for training AI players in the soccer game.

## Structure

- `soccer_env.py`: Gymnasium environment wrapper for the soccer game
- `train.py`: Training script using PPO algorithm from stable-baselines3
- `ai_controller.py`: AI controller that loads trained models and controls team_2
- `models/`: Directory where trained models are saved
- `logs/`: Directory for tensorboard logs

## Usage

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train a Model
```bash
# Train for 100,000 timesteps (default)
python ai/train.py train

# Train for custom timesteps with custom name
python ai/train.py train 200000 my_soccer_ai
```

### 3. Evaluate a Model
```bash
# Evaluate the default model
python ai/train.py evaluate

# Evaluate a specific model
python ai/train.py evaluate ai/models/my_soccer_ai_final.zip 5
```

### 4. Use Trained Model in Game
Simply run the main game:
```bash
python main.py
```

The game will automatically load the trained model from `ai/models/soccer_ai_final.zip` if it exists. Team_2 (blue, right side) will be controlled by AI, while team_1 (red, left side) remains keyboard-controlled.

## Model Details

- **Algorithm**: PPO (Proximal Policy Optimization)
- **Observation Space**: 50-dimensional vector including:
  - Ball position and velocity (4 values)
  - All players' positions and velocities (40 values)
  - Goal positions and match state (6 values)
- **Action Space**: 5 discrete actions per player × 5 players
  - 0: Do nothing
  - 1: Move up
  - 2: Move down
  - 3: Move left
  - 4: Move right
  - 5: Shoot/dribble

## Reward Structure

- +100 for scoring a goal
- -50 for opponent scoring
- +0.1 for being close to the ball
- +0.05 for moving ball toward opponent goal
- -0.01 time penalty to encourage faster play