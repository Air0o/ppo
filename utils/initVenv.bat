@echo off
cd utils
cd ..
@echo on
python3 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\activate.bat