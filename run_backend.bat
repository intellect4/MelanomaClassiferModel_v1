@echo off
title Melanoma AI Decision Support - FastAPI REST Server
echo ======================================================================
echo    Starting Melanoma AI REST API Backend (FastAPI / Uvicorn)
echo ======================================================================
echo Swagger OpenAPI Docs : http://127.0.0.1:8000/docs
echo Health Check Endpoint: http://127.0.0.1:8000/health
echo.
echo Launching Uvicorn Server at http://127.0.0.1:8000 ...
echo Press Ctrl+C in this terminal window to stop the server.
echo.
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
pause
