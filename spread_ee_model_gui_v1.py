# Generic imports
import sys
import traceback
import os
import time
from datetime import timedelta, datetime, date
import subprocess
import numpy as np
from pathlib import Path
import pickle

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

# Some additional ones
CONFIG_DIR_NAME = "configs"
OUTPUT_DIR_NAME = "results"

# Main UI class with all methods
class SpreadEEGui(QtWidgets.QMainWindow, vspread_model_v1_ui.Ui_SpreadEEGui):
    # Applications states in status bar
    APP_STATUS_STATES = {"ready": "Ready.",
                         "processing": "Processing "}

    DATA_PROCESSED_STATE = {0: "Model init: <span style='color: red'>NO</span>",
                            1: "Model init: <span style='color: green'>YES</span>"}

    initializing = False
    app = None
    sim = None  # The simulation object
    model_initialized = False  # Whether model is initialized

    proposed_fn = None

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
        if self.model_initialized:
            txt = self.DATA_PROCESSED_STATE[1]
        self.lblDataProcesses.setText(txt)

    # Generate file name for the config file based on now
    @staticmethod
    def generate_file_name():
        ts = datetime.fromtimestamp(time.time()).strftime('%Y_%m_%d_%H%M%S')
        return "ee_spr_config_" + ts + ".cfg"  # Internally we use Pickle, but ext is .cfg

    # Get and check generation data
    def get_and_check_generation_data(self):

        # If there are some issues with the values, will inform the user.
        generation_param_error = False
        generation_param_error_texts = []

        # N days back
        try:
            days_back = int(self.txtNDaysAgo.text())

            # Check numeric values
            if days_back < 0 or days_back > 80:
                generation_param_error = True
                days_back = 10
                generation_param_error_texts.append("Stop initialization N days back parameter " +
                                                    "must be in the range [0, 80]. Setting to default = 10.")
        except:
            generation_param_error = True
            generation_param_error_texts.append("Cannot understand the stop initialization N days back parameter."
                                                + " Setting to default = 10.")
            days_back = 10

        self.txtNDaysAgo.setText(str(days_back))  # Set parameter after all checks are completed

        # Fraction of generation interval
        try:
            gen_interval_factor = float(self.txtFractionGenerationInt.text())
        except:
            generation_param_error = True
            generation_param_error_texts.append("Cannot understand the generation interval factor parameter."
                                                + " Setting to default = 0.8.")
            gen_interval_factor = 0.8

        self.txtFractionGenerationInt.setText(str(gen_interval_factor))

        # Infection probability
        try:
            prob_will_infect = float(self.txtProbInitialInfect.text())

            # This is a probability, need to check that it is within correct the correct range
            if prob_will_infect < 0 or prob_will_infect > 1:
                generation_param_error = True
                prob_will_infect = 0.1
                generation_param_error_texts.append("Initial pool infection probability " +
                                                    "must be in the range [0, 1]. Setting to default = 0.1.")
        except:
            generation_param_error = True
            generation_param_error_texts.append("Cannot understand the initial pool infection probability "
                                                "parameter. Setting to default = 0.1.")
            prob_will_infect = 0.1

        self.txtProbInitialInfect.setText(str(prob_will_infect))

        if generation_param_error:
            self.show_info_box("There were issues with generation parameters",
                               "\n\n".join(generation_param_error_texts))
            self.log("Some generation parameters were erroneous. Reverted to some defaults.")

        gen_params = {
            "days_back": days_back,
            "gen_interval_factor": gen_interval_factor,
            "prob_will_infect": prob_will_infect
        }

        return gen_params

    # There are many parameters that need to be checked here. So a semi-automatic check will be implemented.
    # Also, you should only call this if a simulation structure already exists and is attached to the UI
    def get_and_check_simulation_data(self):

        # The following is a list of lists for parameters to check and/or set.
        # The order of elements in a parameter list is as follows:
        # [paramName, widgetName, defaultVal, (min, max), boundType, dtype]
        # where
        # [0] paramKey - dict key for the parameter
        # [1] paramName - human-readable name of the parameter
        # [2] widgetName - UI widget name. Only line edit widgets supported now
        # [3] defaultVal - default value to which the parameter is reverted if erroneous
        # [4] (min, max) - min and max bounds of the parameter, can be None, see also boundType
        # [5] boundType - "lb", "ub", "both", or None. If former three, apply data from (min, max)
        # [6] dtype - type of the parameter. Either "int" or "float".

        all_params = [
            ["frac_stay_active", "Fraction staying active in region", "txtSimParamStayingActive",
            0.005, (0, 1), "both", "float"],
            ["prob_mob_regions", "Probability of mobility between regions is unrestricted",
             "txtSimParamProbMobBetweenReg", 1.0, (0, 1), "both", "float"],
            ["prob_mob_saaremaa", "Probability of mobility between Saaremaa and mainland is unrestricted",
             "txtSimParamProbMobilityBetweenSaareUnrestr", 1.0, (0, 1), "both", "float"],
            ["r0r1", "R0 for Region 1", "txtR0R1", 0.9, (0,), "lb", "float"],
            ["r0r2", "R0 for Region 2", "txtR0R2", 0.8, (0,), "lb", "float"],
            ["r0r3", "R0 for Region 3", "txtR0R3", 0.8, (0,), "lb", "float"],
            ["r0r4", "R0 for Region 4", "txtR0R4", 0.8, (0,), "lb", "float"],
            ["r0r5", "R0 for Region 5", "txtR0R5", 0.8, (0,), "lb", "float"],
            ["r0r6", "R0 for Region 6", "txtR0R6", 0.6, (0,), "lb", "float"],
            ["r0r7", "R0 for Region 7", "txtR0R7", 0.8, (0,), "lb", "float"],
            ["r0r8", "R0 for Region 8", "txtR0R8", 0.8, (0,), "lb", "float"],
            ["t_stop", "Simulation stops after N days", "txtTStopNDays", 90, (0,), "lb", "int"]
        ]

        # Lambda for converting to int or float
        convif = lambda val, dtype: int(val) if dtype == "int" else float(val)

        sim_params = {}
        sim_params_error = False
        sim_params_error_texts = []

        for p in all_params:

            try:
                this_param = convif(getattr(self, p[2]).text(), p[6])

                # Check parameter range
                if p[5] is not None:
                    # TODO: Only implemented lower bound and both bounds in the interest of time
                    if p[5] == "lb":
                        if this_param < p[4][0]:
                            sim_params_error = True
                            sim_params_error_texts.append(
                                "Parameter \"" + p[1] + "\" is below the lower bound. Will use default value = "
                                + str(p[3]))
                            this_param = convif(p[3], p[6])

                    if p[5] == "both":
                        if this_param < p[4][0] or this_param > p[4][1]:
                            sim_params_error = True
                            sim_params_error_texts.append(
                                "Parameter \"" + p[1] + "\" is outside the permitted bounds. Will use default value = "
                                + str(p[3]))
                            this_param = convif(p[3], p[6])
            except:
                sim_params_error = True
                sim_params_error_texts.append("Error parsing \"" + p[1] + "\" parameter. " +
                                              "Will use default value = " + str(p[3]))
                this_param = convif(p[3], p[6])

            # Put this param back to the line edit
            getattr(self, p[2]).setText(str(this_param))

            # Save parameter in dict
            sim_params[p[0]] = this_param

        # Check for errors
        if sim_params_error:
            self.show_info_box("There were issues with simulation parameters",
                               "\n\n".join(sim_params_error_texts))
            self.log("Some simulation parameters were erroneous. Reverted to some defaults.")

        return sim_params

    def setup_created_simulation(self):

        # We may need this later
        self.proposed_fn = self.generate_file_name()

        # Check if save config file path is specified. If not, create a configs folder
        # and specify the file. Note that it is not saved automatically. Warn the user.
        if self.txtConfigFileLoc.text() == "":
            # Check if config folder exists, if not, create it
            p = Path(CONFIG_DIR_NAME)
            p.resolve()
            if not (p.exists() and p.is_dir()):
                os.mkdir(CONFIG_DIR_NAME)

            # Generate a new file name and put it to the textbox with path
            self.txtConfigFileLoc.setText(CONFIG_DIR_NAME + os.sep + self.proposed_fn)

        # Check if the output folder is empty
        if self.txtOutputFolder.text() == "":
            p = Path(OUTPUT_DIR_NAME)
            p.resolve()
            if not (p.exists() and p.is_dir()):
                os.mkdir(OUTPUT_DIR_NAME)
            self.txtOutputFolder.setText(self.fix_path(OUTPUT_DIR_NAME))

        # Read the simulation parameters and set them in the simulation structure
        simp = self.get_and_check_simulation_data()

        # Mobility data settings
        self.sim.fraction_stay_active = simp["frac_stay_active"]
        self.sim.prob_mob_regions_unrestricted = simp["prob_mob_regions"]
        self.sim.prob_mob_saare_reg_unrestricted = simp["prob_mob_saaremaa"]

        # Construct and set R0 per region dict
        covid19_r0 = {
            1: simp["r0r1"],
            2: simp["r0r2"],
            3: simp["r0r3"],
            4: simp["r0r4"],
            5: simp["r0r5"],
            6: simp["r0r6"],
            7: simp["r0r7"],
            8: simp["r0r8"],
        }
        self.sim.r0_per_region = covid19_r0

        # Days to simulate
        self.sim.stop_time = simp["t_stop"]

        self.show_info_box("Model initialized",
                           "The model has been initialized, but the configuration " +
                           "file has not yet been written to disk.")

        self.model_initialized = True

        self.update_button_states()
        self.update_data_processed_status()

    def create_new_simulation(self):

        # Temoporary disable the button
        self.btnFetchDataAndProcess.setEnabled(False)

        # Create a new simulation
        try:
            gen_params = self.get_and_check_generation_data()
            self.sim = Covid19SimulationEEV1(data_url = self.sanitize_url(self.txtCovid19TestSrc.text()),
                                             log_redirect = self.log,
                                             gen_params=gen_params)
            self.setup_created_simulation()
        except:
            self.show_info_box("Cannot initialize the model", "Could not initialize the model. " +
                     "Check your network connection and whether the data file is accessible.")

            self.log("Error during model initialization")

        self.btnFetchDataAndProcess.setEnabled(True)

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
        self.btnConfigLoad.clicked.connect(self.config_load)
        self.btnBrowseOutputFolder.clicked.connect(self.browse_output_folder)

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
        if self.model_initialized:
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

    def config_save_as(self):
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

    def browse_output_folder(self):
        dir = QtWidgets.QFileDialog.getExistingDirectory(self, "Choose output directory")
        if dir:
            self.txtOutputFolder.setText(self.fix_path(dir))

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
