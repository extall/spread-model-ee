@echo off

#SET pyuicpath="C:\Anaconda\Library\bin\pyuic5"
SET pyuicpath="C:\Users\Alex\.conda\envs\movement\Library\bin\pyuic5"

echo Running pyuic5...

%PYUICPATH% ui\vspread_model_v1.ui -o ui\vspread_model_v1_ui.py

echo Done.
pause