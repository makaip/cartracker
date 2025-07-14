@echo off
:: Car Tracker Setup and Run Script for Windows
:: This script helps set up and run the car tracking system

title Car Tracker Setup

echo ================================
echo Car Tracker Setup Script
echo ================================
echo.

:: Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)

echo Python found: 
python --version
echo.

:: Check if we're in the right directory
if not exist "car_tracker.py" (
    echo Error: car_tracker.py not found
    echo Please run this script from the cartracker directory
    pause
    exit /b 1
)

:: Menu system
:menu
echo What would you like to do?
echo.
echo 1. Install dependencies
echo 2. Set up sample car data
echo 3. Run car tracker (4 cameras)
echo 4. Run car tracker (custom)
echo 5. Validate car data
echo 6. Exit
echo.
set /p choice="Enter your choice (1-6): "

if "%choice%"=="1" goto install
if "%choice%"=="2" goto setup_data
if "%choice%"=="3" goto run_default
if "%choice%"=="4" goto run_custom
if "%choice%"=="5" goto validate
if "%choice%"=="6" goto exit
echo Invalid choice, please try again.
echo.
goto menu

:install
echo.
echo Installing dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo Error installing dependencies
    pause
    goto menu
)
echo Dependencies installed successfully!
echo.
pause
goto menu

:setup_data
echo.
echo Setting up sample car data...
set /p num_cars="Enter number of sample cars to create (default 3): "
if "%num_cars%"=="" set num_cars=3
python setup_cars.py sample data %num_cars%
if errorlevel 1 (
    echo Error setting up car data
    pause
    goto menu
)
echo Sample car data created successfully!
echo.
pause
goto menu

:run_default
echo.
echo Running car tracker with 4 cameras...
echo Press Ctrl+C to stop the tracker
echo.
python track.py
pause
goto menu

:run_custom
echo.
set /p cameras="Enter number of cameras (1-9, default 4): "
if "%cameras%"=="" set cameras=4
set /p datadir="Enter data directory (default 'data'): "
if "%datadir%"=="" set datadir=data
echo.
echo Running car tracker with %cameras% cameras, data directory: %datadir%
echo Press Ctrl+C to stop the tracker
echo.
python track.py %cameras% %datadir%
pause
goto menu

:validate
echo.
echo Validating car data...
python setup_cars.py validate
pause
goto menu

:exit
echo.
echo Thanks for using Car Tracker!
pause
exit /b 0
