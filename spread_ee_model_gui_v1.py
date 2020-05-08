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
import copy

# Specific UI features
from PyQt5 import QtGui, QtWidgets, QtCore
from PyQt5.QtWidgets import QGraphicsView
from PyQt5.QtWidgets import QSplashScreen, QMessageBox, QGraphicsScene, QFileDialog

import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
plt.ion()

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
                         "processing": "Processing data",
                         "running_single": "Running single simulation...",
                         "running_mc": "Running Monte-Carlo simulations...",
                         "stopping_single": "Stopping single simulation...",
                         "stopping_mc": "Stopping Monte-Carlo simulations..."}

    DATA_PROCESSED_STATE = {0: "Model init: <span style='color: red'>NO</span>",
                            1: "Model init: <span style='color: green'>YES</span>"}

    SINGLE_SIM_BUTTON_STATES = ["Run single simulation with logging",
                                "Stop single simulation"]

    MC_SIM_BUTTON_STATES = ["Run Monte-Carlo simulations",
                            "Stop Monte-Carlo simulations"]

    initializing = False
    app = None
    sim = None  # The simulation object
    model_initialized = False  # Whether model is initialized

    fig_initial_handle = None
    fig_later_handle = None

    proposed_fn = None

    initial_active_case_dynamics = None

    mc_simulation_running = False
    mc_simulation_abort = False

    def __init__(self, parent=None):

        self.initializing = True

        # Setting up the base UI
        super(SpreadEEGui, self).__init__(parent)
        self.setupUi(self)

        # Update button states
        self.update_button_states()

        # Log this anyway
        self.log("Application started")

        # Hide progress bar
        self.init_progress_bar()

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

        self.update_simulation_params()

        self.show_info_box("Model initialized",
                           "The model has been initialized, but the configuration " +
                           "file has not yet been written to disk.")

        self.model_initialized = True

        self.update_button_states()
        self.update_data_processed_status()

    def update_simulation_params(self):

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

    def create_new_simulation(self):

        self.status_bar_message("processing")

        # Temoporary disable the button
        self.btnFetchDataAndProcess.setEnabled(False)

        # Create a new simulation
        try:
            gen_params = self.get_and_check_generation_data()
            self.sim = Covid19SimulationEEV1(data_url = self.sanitize_url(self.txtCovid19TestSrc.text()),
                                             log_redirect = self.log,
                                             gen_params=gen_params)
            self.setup_created_simulation()
        except Exception as ex:
            self.show_info_box("Cannot initialize the model", "Could not initialize the model. " +
                     "Check your network connection and whether the data file is accessible.")

            self.log("Error during model initialization: " + str(ex))

        self.btnFetchDataAndProcess.setEnabled(True)

        # If plotting is enabled, show active case dynamics
        if self.actionDisplayActiveCaseGraph.isChecked() and self.model_initialized:
            self.do_initial_plot()

        self.status_bar_message("ready")

    def do_initial_plot(self):

        if self.sim is not None and self.model_initialized:
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

            # Store this information since it may be necessary later
            self.initial_active_case_dynamics = (t, y)

            # Clear previous plot
            if self.fig_initial_handle is not None:
                self.fig_initial_handle.clear()
                self.fig_initial_handle = None

            self.fig_initial_handle = plt.figure(num="EE Virus Active Cases")
            plt.plot(t, y)
            plt.xlabel("Days since " + str(datas["first_date"]))
            plt.ylabel("Number of active cases in Estonia [statistical model]")
            plt.title("Active case dynamics in Estonia (with predicted recovery)")
            plt.grid(True)

    # Set up those UI elements that depend on config
    def config_ui(self):

        # TODO: TEMP: For buttons, use .clicked.connect(self.*), for menu actions .triggered.connect(self.*),
        # TODO: TEMP: for checkboxes use .stateChanged, and for spinners .valueChanged
        self.btnFetchDataAndProcess.clicked.connect(self.create_new_simulation)
        self.btnConfigLoad.clicked.connect(self.config_load)
        self.btnBrowseOutputFolder.clicked.connect(self.browse_output_folder)
        self.btnSaveConfig.clicked.connect(self.config_save)
        self.btnConfigSaveAs.clicked.connect(self.config_save_as)
        self.btnSingleSim.clicked.connect(self.run_single_simulation)
        self.btnMCSim.clicked.connect(self.run_mc_simulations)

        # Menu actions
        self.actionLoad_config_file.triggered.connect(self.config_load)
        self.actionSave_config_file_as.triggered.connect(self.config_save_as)
        self.actionClear_log.triggered.connect(self.clear_log)

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

    def init_progress_bar(self):
        self.progressBar.hide()

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

        p = Path(self.txtConfigFileLoc.text())
        dir = p.parent

        fn = QtWidgets.QFileDialog.getOpenFileName(self, "Load config file", str(dir), "Config file (*.cfg)")

        # If a file is chosen, process it
        if fn[0] != "":
            try:
                with open(fn[0], "rb") as f:
                    my_sim = pickle.load(f)

                # Parse the simulation structure and put all parameters where they need to go
                self.sim = my_sim  # Set up the simulation structure

                # Populate the parameter fields based on the parameters stored in the simulation object
                success = self.populate_fields_from_sim()

                if success:

                    # Set the file name
                    self.txtConfigFileLoc.setText(fn[0])

                    # Check if the output folder is empty
                    if self.txtOutputFolder.text() == "":
                        p = Path(OUTPUT_DIR_NAME)
                        p.resolve()
                        if not (p.exists() and p.is_dir()):
                            os.mkdir(OUTPUT_DIR_NAME)
                        self.txtOutputFolder.setText(self.fix_path(OUTPUT_DIR_NAME))

                    # Finally, show that the model is initialized
                    self.model_initialized = True
                    self.update_data_processed_status()

                    # Update also buttons
                    self.update_button_states()

            except Exception as ex:
                self.show_info_box("Could not load config file", "Could not load or parse the config file. It either" +
                                   " has an unsupported format, or it is a wrong type of" +
                                   " file.")

                self.log("Could not load the config file: " + str(ex))

    def populate_fields_from_sim(self):
        if self.sim is not None:
            try:
                # Simulation parameters
                params_to_set = [[self.sim.fraction_stay_active, "txtSimParamStayingActive"],
                                 [self.sim.r0_per_region[1], "txtR0R1"],
                                 [self.sim.r0_per_region[2], "txtR0R2"],
                                 [self.sim.r0_per_region[3], "txtR0R3"],
                                 [self.sim.r0_per_region[4], "txtR0R4"],
                                 [self.sim.r0_per_region[5], "txtR0R5"],
                                 [self.sim.r0_per_region[6], "txtR0R6"],
                                 [self.sim.r0_per_region[7], "txtR0R7"],
                                 [self.sim.r0_per_region[8], "txtR0R8"],
                                 [self.sim.prob_mob_regions_unrestricted, "txtSimParamProbMobBetweenReg"],
                                 [self.sim.prob_mob_saare_reg_unrestricted, "txtSimParamProbMobilityBetweenSaareUnrestr"],
                                 [self.sim.stop_time, "txtTStopNDays"]
                                 ]

                for p in params_to_set:
                    getattr(self, p[1]).setText(str(p[0]))

                # Initial state generation parameters
                gene_to_set = [
                    [self.sim.gen_params["days_back"], "txtNDaysAgo"],
                    [self.sim.gen_params["gen_interval_factor"], "txtFractionGenerationInt"],
                    [self.sim.gen_params["prob_will_infect"], "txtProbInitialInfect"]
                ]

                for p in gene_to_set:
                    getattr(self, p[1]).setText(str(p[0]))

                return True

            except Exception as ex:
                self.show_info_box("Could not load configuration", "An error occured during loading of parameters " +
                                   "from the configuration file. Please make sure that the file is valid and is " +
                                   "compatible with this version of the tool."
                                   )
                self.log("Could not load parameters from configuration file: " + str(ex))
                return False
        else:
            self.show_info_box("No simulation loaded", "Cannot load the configuration parameters since no there is " +
                               "no simulation data loaded.")
            return False

    def config_save(self):

        # First, update the parameters of simulation.
        # Note that generation parameters ARE NOT overwritten in the save even if changed in the interface
        self.update_simulation_params()

        # Get config file name
        fs = self.txtConfigFileLoc.text()

        # Try to save
        try:
            sim = copy.deepcopy(self.sim)  # Create a deep copy so that everything is stored
            with open(fs, "wb") as f:
                pickle.dump(sim, f)
            self.log("Successfully saved the configuration file.")
        except Exception as ex:
            self.show_info_box("Config file write error", "Could not save configuration file")
            self.log("Could not save configuration file: " + str(ex))

    def config_save_as(self):

        # Get current directory
        p = Path(self.txtConfigFileLoc.text())
        dir = p.parent
        prefer_save = str(dir) + os.sep + self.proposed_fn

        fs = QtWidgets.QFileDialog.getSaveFileName(self, "Save config file as", prefer_save, "*.cfg")
        if fs:
            self.txtConfigFileLoc.setText(fs[0])
            self.config_save()

    def clear_log(self):
        self.txtConsole.setText("")

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

    # Simulations #

    # This is a helper function for fetching simulation data in (param, param_value) pairs
    def get_simulation_metadata(self):
        return [("[Sim] Simulation performed on", self.sim.created_timestamp),
                        ("[Sim] Simulation start date", self.sim.date_start),
                        ("[Sim] Number of days simulated", self.sim.stop_time),
                        ("[Sim] Initial number of infected across all regions", self.sim.initial_infected_number),
                        ("[Sim] Fraction of people staying active in their area", self.sim.fraction_stay_active),
                        ("[Sim] R_0 for area R1", self.sim.r0_per_region[1]),
                        ("[Sim] R_0 for area R2", self.sim.r0_per_region[2]),
                        ("[Sim] R_0 for area R3", self.sim.r0_per_region[3]),
                        ("[Sim] R_0 for area R4", self.sim.r0_per_region[4]),
                        ("[Sim] R_0 for area R5", self.sim.r0_per_region[5]),
                        ("[Sim] R_0 for area R6", self.sim.r0_per_region[6]),
                        ("[Sim] R_0 for area R7", self.sim.r0_per_region[7]),
                        ("[Sim] R_0 for area R8", self.sim.r0_per_region[8]),
                        ("[Sim] Prob. that movement is not restricted between regions",
                         self.sim.prob_mob_regions_unrestricted),
                        ("[Sim] Prob. that movement is not restricted between Saaremaa and mainland",
                         self.sim.prob_mob_saare_reg_unrestricted),
                        ]

    # Single simulation
    def run_single_simulation(self):

        # Need to read off latest parameters
        self.update_simulation_params()

        # Perform actions depending on the simulation state
        if self.sim.simulation_running:
            # Abort simulation
            self.status_bar_message("stopping_single")
            self.sim.simulation_abort = True
            self.btnSingleSim.setEnabled(False)
            self.btnMCSim.setEnabled(False)

        else:
            # Start the simulation
            self.status_bar_message("running_single")
            self.btnSingleSim.setText(self.SINGLE_SIM_BUTTON_STATES[1])

            tim = time.time()
            self.log("Running single simulation with spread simulation starting date "
                     + str(self.sim.date_start) + "...")

            out = self.sim.do_simulation(logfn=self.log)

            if out:

                # Read the data
                act, new = out

                # Reset button state
                self.btnSingleSim.setText(self.SINGLE_SIM_BUTTON_STATES[0])

                self.log("Single simulation completed in " + str(time.time()-tim) + " seconds.")

                column_names = ["1", "2", "3", "4", "5", "6", "7", "8", "Total"]

                # Filename
                fn = "single_sim_results_" + \
                     datetime.fromtimestamp(time.time()).strftime('%Y_%m_%d_%H%M%S') + ".xlsx"

                pdata_act = pd.DataFrame(data=act, columns=column_names)
                pdata_new = pd.DataFrame(data=new, columns=column_names)

                # Also need to add useful metadata
                meta = self.get_simulation_metadata()
                pmeta = pd.DataFrame(data=meta, columns=['Parameter', 'Value'])

                excelwr = pd.ExcelWriter(self.txtOutputFolder.text() + fn, engine="xlsxwriter")

                pdata_act.to_excel(excelwr, sheet_name="exp+inf in areas")
                pdata_new.to_excel(excelwr, sheet_name="new in areas")
                pmeta.to_excel(excelwr, sheet_name="sim meta")

                excelwr.save()

                self.log("Saved simulation result as " + fn)

            else:
                self.btnSingleSim.setText(self.SINGLE_SIM_BUTTON_STATES[0])
                self.btnSingleSim.setEnabled(True)
                self.btnMCSim.setEnabled(True)
                self.log("Single simulation aborted.")

            self.status_bar_message("ready")

    # Monte-Carlo simulations
    def run_mc_simulations(self):

        # Reset abort state
        self.mc_simulation_abort = False

        # Need to read off latest parameters
        self.update_simulation_params()

        # Perform actions depending on the simulation state
        if self.mc_simulation_running:
            # Abort simulation
            self.status_bar_message("stopping_mc")

            self.sim.simulation_abort = True
            self.mc_simulation_abort = True

            self.btnSingleSim.setEnabled(False)
            self.btnMCSim.setEnabled(False)

        else:
            # Start the simulation
            self.mc_simulation_running = True
            self.status_bar_message("running_mc")
            self.btnMCSim.setText(self.MC_SIM_BUTTON_STATES[1])

            tim = time.time()
            self.log("Running Monte-Carlo simulations with spread simulation starting date "
                     + str(self.sim.date_start) + "...")

            try:
                N = int(self.txtMcCount.text())
            except:
                self.show_info_box("Error reading MC run count", "Could not understand the MC run count. " +
                                   "Resetting to default = 1000")
                N = 1000
                self.txtMcCount.setText(str(N))

            # Start performing N simulations...
            self.progressBar.show()

            # We need to store all simulation output
            outs_active = []
            outs_new = []

            for i in range(N):

                # Bad way to update GUI. Should consider separate thread
                self.app.processEvents()

                out = self.sim.do_simulation(do_log=False)

                if out and not self.mc_simulation_abort:

                    # Read the data
                    act, new = out

                    # Store the data
                    outs_active.append(act)
                    outs_new.append(new)

                    # Continue
                    self.progressBar.setValue(int(100*i/N))

                else:
                    self.btnMCSim.setText(self.MC_SIM_BUTTON_STATES[0])
                    self.btnSingleSim.setEnabled(True)
                    self.btnMCSim.setEnabled(True)
                    self.log("Monte-Carlo simulations aborted.")
                    self.progressBar.hide()
                    break

            # If the simulation wasn't aborted, process the statistical data as needed

            if not self.mc_simulation_abort:

                self.btnMCSim.setText(self.MC_SIM_BUTTON_STATES[0])
                self.btnSingleSim.setEnabled(True)
                self.btnMCSim.setEnabled(True)
                self.progressBar.hide()

                self.log("Crunching statistics...")

                # Active cases
                outs_act_avg = sum(outs_active) / N
                outs_act_std = np.sqrt(sum(np.power(outs_active - outs_act_avg, 2) / (N - 1)))

                outs_act_all = np.concatenate([outs_act_avg, outs_act_std], axis=1)

                # New cases
                outs_new_avg = sum(outs_new) / N
                outs_new_std = np.sqrt(sum(np.power(outs_new - outs_new_avg, 2) / (N - 1)))

                outs_new_all = np.concatenate([outs_new_avg, outs_new_std], axis=1)

                column_names = ["1_mean", "2_mean", "3_mean", "4_mean",
                           "5_mean", "6_mean", "7_mean", "8_mean",
                           "Total_mean",
                           "1_std", "2_std", "3_std", "4_std",
                           "5_std", "6_std", "7_std", "8_std",
                           "Total_std"
                           ]

                # Pandas dataframes
                pdata_act = pd.DataFrame(data=outs_act_all, columns=column_names)
                pdata_new = pd.DataFrame(data=outs_new_all, columns=column_names)

                # Metadata
                meta = self.get_simulation_metadata()

                # Insert number of MC runs into metadata
                meta.insert(3, ("[Sim] Number of Monte-Carlo runs performed", int(self.txtMcCount.text())))

                pmeta = pd.DataFrame(data=meta, columns=['Parameter', 'Value'])

                # Filename
                fn = "mc_sim_results_" + \
                     datetime.fromtimestamp(time.time()).strftime('%Y_%m_%d_%H%M%S') + ".xlsx"

                excelwr = pd.ExcelWriter(self.txtOutputFolder.text() + fn, engine="xlsxwriter")

                pdata_act.to_excel(excelwr, sheet_name="exp+inf in areas")
                pdata_new.to_excel(excelwr, sheet_name="new in areas")
                pmeta.to_excel(excelwr, sheet_name="sim meta")

                excelwr.save()

                self.log("Saved simulation result as " + fn)

            self.mc_simulation_running = False

            self.status_bar_message("ready")


def main():
    # Prepare and launch the GUI
    app = QtWidgets.QApplication(sys.argv)
    app.setWindowIcon(QtGui.QIcon('res/A.ico'))
    dialog = SpreadEEGui()
    dialog.setWindowTitle(APP_TITLE + " - v." + APP_VERSION) # Window title
    dialog.app = app  # Store the reference
    dialog.show()

    # After loading the config file, we need to set up relevant UI elements
    dialog.config_ui()
    dialog.app.processEvents()

    # And proceed with execution
    app.exec_()


# Run main loop
if __name__ == '__main__':
    # Set the exception hook
    sys.excepthook = traceback.print_exception
    main()
