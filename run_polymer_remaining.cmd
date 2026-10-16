@echo off
setlocal
cd /d "%~dp0"
call "C:\ProgramData\anaconda3\Scripts\activate.bat" rl-assisted-env
python -B -u run_polymer_remaining.py --launch
pause
