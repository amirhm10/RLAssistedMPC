@echo off
setlocal
cd /d "%~dp0"
call "C:\ProgramData\anaconda3\Scripts\activate.bat" rl-assisted-env
if errorlevel 1 exit /b 1
python -B -u run_polymer_review_recovery.py --run %*
set "REVIEW_RECOVERY_EXIT=%ERRORLEVEL%"
echo.
echo Recovery command finished with exit code %REVIEW_RECOVERY_EXIT%.
pause
exit /b %REVIEW_RECOVERY_EXIT%
