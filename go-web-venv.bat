@echo off
if exist .venv\Scripts\python.exe (
    .venv\Scripts\python.exe infer-web.py --pycmd .venv\Scripts\python.exe --port 7897
) else (
    python infer-web.py --pycmd python --port 7897
)
pause