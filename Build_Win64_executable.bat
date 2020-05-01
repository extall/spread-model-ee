@echo off

pyinstaller --add-data "res/A.ico;res/" --add-data "data/Area_OD_matrix.pkl;data" spread_ee_model_gui_v1.py

echo Done.