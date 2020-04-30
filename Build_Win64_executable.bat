@echo off

pyinstaller --add-data "res/A.ico;res/" --add-data "configs/Area_OD_matrix.pkl;configs" spread_ee_tool.py

echo Done.