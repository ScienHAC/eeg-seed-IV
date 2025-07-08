@echo off
echo Starting local web server for SEED-IV Dataset Explorer...
echo.
echo Open your browser and go to: http://localhost:8000
echo.
echo Press Ctrl+C to stop the server
echo.
python -m http.server 8000
pause
