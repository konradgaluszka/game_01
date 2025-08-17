#!/usr/bin/env python3
"""
Setup script for AI training dependencies and initial model training.
"""

import subprocess
import sys
import os


def install_dependencies():
    """Install AI training dependencies"""
    print("Installing AI training dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install dependencies: {e}")
        return False


def train_initial_model():
    """Train an initial AI model"""
    print("Training initial AI model (this may take a while)...")
    try:
        # Create directories
        os.makedirs("ai/models", exist_ok=True)
        os.makedirs("ai/logs", exist_ok=True)
        
        # Train a quick model (10k timesteps for demo)
        subprocess.check_call([
            sys.executable, "ai/train.py", "train", "10000", "soccer_ai"
        ])
        print("✓ Initial model trained successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to train model: {e}")
        return False


def main():
    print("=== Soccer AI Setup ===")
    print("This script will install dependencies and train an initial AI model.")
    
    # Install dependencies
    if not install_dependencies():
        return 1
    
    # Ask user if they want to train now
    train_now = input("\nWould you like to train an initial AI model now? (y/n): ").lower().strip()
    
    if train_now in ['y', 'yes']:
        if not train_initial_model():
            return 1
        print("\n✓ Setup complete! You can now run 'python main.py' to play against the AI.")
    else:
        print("\n✓ Dependencies installed! You can train a model later with 'python ai/train.py train'")
    
    print("\nUsage:")
    print("  python main.py                     # Play the game (team_1 = you, team_2 = AI)")
    print("  python ai/train.py train [steps]   # Train AI model")
    print("  python ai/train.py evaluate [model] # Evaluate AI model")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())