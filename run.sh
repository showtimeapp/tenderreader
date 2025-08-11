#!/bin/bash
# run.sh - Quick start script for macOS/Linux

echo "================================"
echo "Tender Compliance Checker"
echo "================================"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Check if requirements are installed
echo "Checking dependencies..."
pip show streamlit > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
fi

# Create required directories
mkdir -p chroma_db temp_uploads

# Run tests
echo "Running setup tests..."
python test_setup.py

echo ""
echo "Starting application..."
echo "================================"
streamlit run app.py

# Note: For Windows users, create run.bat with:
# @echo off
# echo ================================
# echo Tender Compliance Checker
# echo ================================
# 
# REM Check if virtual environment exists
# if not exist "venv" (
#     echo Creating virtual environment...
#     python -m venv venv
# )
# 
# REM Activate virtual environment
# echo Activating virtual environment...
# call venv\Scripts\activate
# 
# REM Install requirements if needed
# echo Checking dependencies...
# pip show streamlit >nul 2>&1
# if errorlevel 1 (
#     echo Installing dependencies...
#     pip install -r requirements.txt
# )
# 
# REM Create required directories
# if not exist "chroma_db" mkdir chroma_db
# if not exist "temp_uploads" mkdir temp_uploads
# 
# REM Run tests
# echo Running setup tests...
# python test_setup.py
# 
# echo.
# echo Starting application...
# echo ================================
# streamlit run app.py