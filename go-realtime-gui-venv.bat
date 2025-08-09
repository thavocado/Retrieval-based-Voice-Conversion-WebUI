@echo off
if exist .venv\Scripts\python.exe (
    .venv\Scripts\python.exe gui_v1.py
) else (
    python gui_v1.py
)
pause