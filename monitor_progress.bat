@echo off
echo Monitoring SEED-IV System Progress...
echo =====================================
echo.

:loop
python check_progress.py
echo.
echo Checking again in 60 seconds...
timeout /t 60 >nul
goto loop
