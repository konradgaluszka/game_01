@echo off
echo Activating virtual environment...
call venv\Scripts\activate.bat
echo Virtual environment activated.
echo.
echo Available commands:
echo   python main.py                       - Run the soccer game
echo   python ai/train.py train [steps]     - Train AI model  
echo   python ai/train.py evaluate [model]  - Evaluate AI model
echo   python setup_ai.py                   - Setup and train initial model
echo.
cmd /k