# Generic imports
import sys
import traceback
import os
import time
from datetime import timedelta, datetime, date
import subprocess
import numpy as np

# Specific UI features
from PyQt5 import QtGui, QtWidgets, QtCore
from PyQt5.QtWidgets import QGraphicsView
from PyQt5.QtWidgets import QSplashScreen, QMessageBox, QGraphicsScene, QFileDialog

import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
plt.ion()
import matplotlib.patches as patches

# The library
from lib.spread_ee import *
from ui import vspread_model_v1_ui

# Overall constants
PUBLISHER = "AlphaControlLab"
APP_TITLE = "EE Virus Spread Model"
APP_VERSION = "1.0-beta"


# Main UI class with all methods
class SpreadEEGui(QtWidgets.QMainWindow, vspread_model_v1_ui.Ui_SpreadEEGui):
    # Applications states in status bar
    APP_STATUS_STATES = {"ready": "Ready.",
                         "processing": "Processing "}

    DATA_PROCESSED_STATE = {0: "Data processed: <span style='color: red'>NO</span>",
                            1: "Data processed: <span style='color: green'>YES</span>"}

    # Internal vars
    initializing = False
    app = None
    sim = None  # The simulation object
    model_initialized = False  # Whether model is initialized

    data_processed = False

    def __init__(self, parent=None):

        self.initializing = True

        # Setting up the base UI
        super(SpreadEEGui, self).__init__(parent)
        self.setupUi(self)

        # Update button states
        self.update_button_states()

        # Log this anyway
        self.log("Application started")

        # Initialization completed
        self.initializing = False

        # Some configs
        self.update_data_processed_status()

        # Set up the status bar
        self.status_bar_message("ready")

    # Data processed status
    def update_data_processed_status(self):
        txt = self.DATA_PROCESSED_STATE[0]
        if self.data_processed:
            txt = self.DATA_PROCESSED_STATE[1]
        self.lblDataProcesses.setText(txt)

    def setup_simulation(self):
        # TODO: implement everything here

        self.model_initialized = True

    def create_new_simulation(self):

        # Create a new simulation
        try:
            self.sim = Covid19SimulationEEV1(data_url = self.sanitize_url(self.txtCovid19TestSrc.text()),
                                             log_redirect = self.log,
                                             days_back = int(self.txtNDaysAgo.text()))
            self.setup_simulation()
        except:
            self.log("Could not initialize model. Check your network connection and whether the data file is accessible.")

        # If plotting is enabled, show active case dynamics
        if self.actionDisplayActiveCaseGraph.isChecked() and self.model_initialized:
            
            datas = self.sim.initial_pool_aux_data

            # Now we create a dict which holds days relative to the first date
            # and for every day we will collect information on how many infected there are
            # and how many recovered. Thus we will have a curve of active cases according
            # to the COVID19 statistics

            first_date = datas["first_date"]
            the_log = datas["logdata"]

            days_active_cases = {}

            for row in the_log:
                rel_time = row[1] - first_date
                rel_day = rel_time.days
                diff = 1 if row[3] is "I" else -1
                if rel_day in days_active_cases.keys():
                    days_active_cases[rel_day] += diff
                else:
                    days_active_cases[rel_day] = diff

            # Create a numpy array such that will be filled with available data. Will count
            # active cases daily. This is needed to plot the final graph.
            num_pts = max(days_active_cases.keys()) + 1

            t = np.arange(0, num_pts)
            y = np.zeros((num_pts,))

            # Start filling in the Y (active cases)
            actsum = 0
            for i in t:
                if int(i) in days_active_cases:
                    actsum += days_active_cases[int(i)]
                y[int(i)] = actsum

            h = plt.figure()
            plt.plot(t, y)
            plt.xlabel("Days since " + str(datas["first_date"]))
            plt.ylabel("Number of active cases in Estonia [statistical model]")
            plt.grid(True)

    # Set up those UI elements that depend on config
    def config_ui(self):

        # TODO: TEMP: For buttons, use .clicked.connect(self.*), for menu actions .triggered.connect(self.*),
        # TODO: TEMP: for checkboxes use .stateChanged, and for spinners .valueChanged
        self.btnFetchDataAndProcess.clicked.connect(self.create_new_simulation)

    # Helper for QMessageBox
    @staticmethod
    def show_info_box(title, text):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setText(text)
        msg.setWindowTitle(title)
        msg.setModal(True)
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec()

    @staticmethod
    def sanitize_url(url):
        return url.replace(" ", "")

    # In-GUI console log
    def log(self, line):
        # Get the time stamp
        ts = datetime.fromtimestamp(time.time()).strftime('[%Y-%m-%d %H:%M:%S] ')
        self.txtConsole.moveCursor(QtGui.QTextCursor.End)
        self.txtConsole.insertPlainText(ts + line + os.linesep)

        # Only do this if app is already referenced in the GUI (TODO: a more elegant solution?)
        if self.app is not None:
            self.app.processEvents()

    @staticmethod
    def open_file_in_os(fn):
        if sys.platform == "win32":
            os.startfile(fn)
        else:
            opener = "open" if sys.platform == "darwin" else "xdg-open"
            subprocess.call([opener, fn])

    # Path related functions
    @staticmethod
    def fix_path(p):
        # Only if string is nonempty
        if len(p) > 0:
            p = p.replace("/", os.sep).replace("\\", os.sep)
            p = p + os.sep if p[-1:] != os.sep else p
            return p

    @staticmethod
    def fix_file_path(p):
        # Only if string is nonempty
        if len(p) > 0:
            p = p.replace("/", os.sep).replace("\\", os.sep)
            return p

    def update_button_states(self):

        state = False
        if self.sim is not None:
            state = True

        self.btnSingleSim.setEnabled(state)
        self.btnMCSim.setEnabled(state)
        self.btnSaveConfig.setEnabled(state)
        self.btnConfigSaveAs.setEnabled(state)


    # The following methods deal with config files
    def config_load(self):
        print("Not implemented")

    def config_save(self):
        print("Not implemented")

    def check_paths(self):
        # Use this to check the paths
        print("Not implemented")
        # self.txtImageDir.setText(self.fix_path(self.txtImageDir.text())) # example

    # Show different messages in status bar
    def status_bar_message(self, msgid):
        self.statusbar.showMessage(self.APP_STATUS_STATES[msgid])
        if self.app is not None:
            self.app.processEvents()

    # Locate the shapefile directory
    def load_config_file(self):
        # dir = QtWidgets.QFileDialog.getExistingDirectory(self, "Choose directory containing the defect shapefiles")
        print("Not implemented")

    def save_as_config_file(self):
        print("Not implemented")

def main():
    # Prepare and launch the GUI
    app = QtWidgets.QApplication(sys.argv)
    app.setWindowIcon(QtGui.QIcon('res/A.ico'))
    dialog = SpreadEEGui()
    dialog.setWindowTitle(APP_TITLE + " - v." + APP_VERSION) # Window title
    dialog.app = app  # Store the reference
    dialog.show()

    # Now we have to load the app configuration file
    dialog.config_load()

    # After loading the config file, we need to set up relevant UI elements
    dialog.config_ui()
    dialog.app.processEvents()

    # Now we also save the config file
    dialog.config_save()

    # And proceed with execution
    app.exec_()


# Run main loop
if __name__ == '__main__':
    # Set the exception hook
    sys.excepthook = traceback.print_exception
    main()
