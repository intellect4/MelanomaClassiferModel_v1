@echo off
title Melanoma AI Clinical Decision Support Dashboard
echo ======================================================================
echo    Starting Melanoma AI Clinical Decision Support Dashboard
echo ======================================================================
echo Hardware: Intel Core i5-13500H CPU Engine
echo Dataset : Linked to local DataSet/ (7,677 Histology Patches)
echo.
echo Launching Streamlit Web Application at http://localhost:8501 ...
echo Press Ctrl+C in this terminal window to stop the application.
echo.
python -m streamlit run frontend/app.py
pause
