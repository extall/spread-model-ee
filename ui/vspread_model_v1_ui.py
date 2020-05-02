# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ui\vspread_model_v1.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_SpreadEEGui(object):
    def setupUi(self, SpreadEEGui):
        SpreadEEGui.setObjectName("SpreadEEGui")
        SpreadEEGui.resize(920, 800)
        SpreadEEGui.setMinimumSize(QtCore.QSize(900, 800))
        self.centralwidget = QtWidgets.QWidget(SpreadEEGui)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.gbModelInitialization = QtWidgets.QGroupBox(self.centralwidget)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.gbModelInitialization.setFont(font)
        self.gbModelInitialization.setObjectName("gbModelInitialization")
        self.verticalLayout_8 = QtWidgets.QVBoxLayout(self.gbModelInitialization)
        self.verticalLayout_8.setObjectName("verticalLayout_8")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label = QtWidgets.QLabel(self.gbModelInitialization)
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.horizontalLayout.addWidget(self.label)
        self.txtCovid19TestSrc = QtWidgets.QLineEdit(self.gbModelInitialization)
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self.txtCovid19TestSrc.setFont(font)
        self.txtCovid19TestSrc.setObjectName("txtCovid19TestSrc")
        self.horizontalLayout.addWidget(self.txtCovid19TestSrc)
        self.btnFetchDataAndProcess = QtWidgets.QPushButton(self.gbModelInitialization)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.btnFetchDataAndProcess.setFont(font)
        self.btnFetchDataAndProcess.setObjectName("btnFetchDataAndProcess")
        self.horizontalLayout.addWidget(self.btnFetchDataAndProcess)
        self.lblDataProcesses = QtWidgets.QLabel(self.gbModelInitialization)
        self.lblDataProcesses.setObjectName("lblDataProcesses")
        self.horizontalLayout.addWidget(self.lblDataProcesses)
        self.verticalLayout_8.addLayout(self.horizontalLayout)
        self.line_2 = QtWidgets.QFrame(self.gbModelInitialization)
        self.line_2.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_2.setObjectName("line_2")
        self.verticalLayout_8.addWidget(self.line_2)
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.label_3 = QtWidgets.QLabel(self.gbModelInitialization)
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self.label_3.setFont(font)
        self.label_3.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_3.setObjectName("label_3")
        self.gridLayout.addWidget(self.label_3, 1, 0, 1, 1)
        self.label_2 = QtWidgets.QLabel(self.gbModelInitialization)
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self.label_2.setFont(font)
        self.label_2.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 0, 0, 1, 1)
        self.txtFractionGenerationInt = QtWidgets.QLineEdit(self.gbModelInitialization)
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self.txtFractionGenerationInt.setFont(font)
        self.txtFractionGenerationInt.setObjectName("txtFractionGenerationInt")
        self.gridLayout.addWidget(self.txtFractionGenerationInt, 1, 1, 1, 1)
        self.txtNDaysAgo = QtWidgets.QLineEdit(self.gbModelInitialization)
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self.txtNDaysAgo.setFont(font)
        self.txtNDaysAgo.setObjectName("txtNDaysAgo")
        self.gridLayout.addWidget(self.txtNDaysAgo, 0, 1, 1, 1)
        self.label_5 = QtWidgets.QLabel(self.gbModelInitialization)
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self.label_5.setFont(font)
        self.label_5.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_5.setObjectName("label_5")
        self.gridLayout.addWidget(self.label_5, 2, 0, 1, 1)
        self.txtProbInitialInfect = QtWidgets.QLineEdit(self.gbModelInitialization)
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self.txtProbInitialInfect.setFont(font)
        self.txtProbInitialInfect.setObjectName("txtProbInitialInfect")
        self.gridLayout.addWidget(self.txtProbInitialInfect, 2, 1, 1, 1)
        self.gridLayout.setRowMinimumHeight(0, 25)
        self.gridLayout.setRowMinimumHeight(1, 25)
        self.gridLayout.setRowMinimumHeight(2, 25)
        self.verticalLayout_8.addLayout(self.gridLayout)
        self.verticalLayout_2.addWidget(self.gbModelInitialization)
        self.gbSimulation = QtWidgets.QGroupBox(self.centralwidget)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.gbSimulation.setFont(font)
        self.gbSimulation.setObjectName("gbSimulation")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.gbSimulation)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.gridLayout_3 = QtWidgets.QGridLayout()
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.label_18 = QtWidgets.QLabel(self.gbSimulation)
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self.label_18.setFont(font)
        self.label_18.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_18.setObjectName("label_18")
        self.gridLayout_3.addWidget(self.label_18, 1, 0, 1, 1)
        self.txtSimParamProbMobBetweenReg = QtWidgets.QLineEdit(self.gbSimulation)
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self.txtSimParamProbMobBetweenReg.setFont(font)
        self.txtSimParamProbMobBetweenReg.setObjectName("txtSimParamProbMobBetweenReg")
        self.gridLayout_3.addWidget(self.txtSimParamProbMobBetweenReg, 1, 1, 1, 1)
        self.txtSimParamStayingActive = QtWidgets.QLineEdit(self.gbSimulation)
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self.txtSimParamStayingActive.setFont(font)
        self.txtSimParamStayingActive.setObjectName("txtSimParamStayingActive")
        self.gridLayout_3.addWidget(self.txtSimParamStayingActive, 0, 1, 1, 1)
        self.label_17 = QtWidgets.QLabel(self.gbSimulation)
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self.label_17.setFont(font)
        self.label_17.setObjectName("label_17")
        self.gridLayout_3.addWidget(self.label_17, 0, 0, 1, 1)
        self.label_19 = QtWidgets.QLabel(self.gbSimulation)
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self.label_19.setFont(font)
        self.label_19.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_19.setObjectName("label_19")
        self.gridLayout_3.addWidget(self.label_19, 2, 0, 1, 1)
        self.txtSimParamProbMobilityBetweenSaareUnrestr = QtWidgets.QLineEdit(self.gbSimulation)
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self.txtSimParamProbMobilityBetweenSaareUnrestr.setFont(font)
        self.txtSimParamProbMobilityBetweenSaareUnrestr.setObjectName("txtSimParamProbMobilityBetweenSaareUnrestr")
        self.gridLayout_3.addWidget(self.txtSimParamProbMobilityBetweenSaareUnrestr, 2, 1, 1, 1)
        self.horizontalLayout_6.addLayout(self.gridLayout_3)
        self.verticalLayout_4.addLayout(self.horizontalLayout_6)
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.gridLayout_2 = QtWidgets.QGridLayout()
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.label_6 = QtWidgets.QLabel(self.gbSimulation)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_6.setFont(font)
        self.label_6.setObjectName("label_6")
        self.gridLayout_2.addWidget(self.label_6, 0, 0, 1, 1)
        self.txtR0R5 = QtWidgets.QLineEdit(self.gbSimulation)
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self.txtR0R5.setFont(font)
        self.txtR0R5.setObjectName("txtR0R5")
        self.gridLayout_2.addWidget(self.txtR0R5, 0, 10, 1, 1)
        self.txtR0R6 = QtWidgets.QLineEdit(self.gbSimulation)
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self.txtR0R6.setFont(font)
        self.txtR0R6.setObjectName("txtR0R6")
        self.gridLayout_2.addWidget(self.txtR0R6, 0, 12, 1, 1)
        self.txtR0R1 = QtWidgets.QLineEdit(self.gbSimulation)
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self.txtR0R1.setFont(font)
        self.txtR0R1.setObjectName("txtR0R1")
        self.gridLayout_2.addWidget(self.txtR0R1, 0, 2, 1, 1)
        self.label_13 = QtWidgets.QLabel(self.gbSimulation)
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self.label_13.setFont(font)
        self.label_13.setObjectName("label_13")
        self.gridLayout_2.addWidget(self.label_13, 0, 13, 1, 1)
        self.txtR0R4 = QtWidgets.QLineEdit(self.gbSimulation)
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self.txtR0R4.setFont(font)
        self.txtR0R4.setObjectName("txtR0R4")
        self.gridLayout_2.addWidget(self.txtR0R4, 0, 8, 1, 1)
        self.label_8 = QtWidgets.QLabel(self.gbSimulation)
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self.label_8.setFont(font)
        self.label_8.setObjectName("label_8")
        self.gridLayout_2.addWidget(self.label_8, 0, 3, 1, 1)
        self.label_7 = QtWidgets.QLabel(self.gbSimulation)
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self.label_7.setFont(font)
        self.label_7.setObjectName("label_7")
        self.gridLayout_2.addWidget(self.label_7, 0, 1, 1, 1)
        self.txtR0R7 = QtWidgets.QLineEdit(self.gbSimulation)
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self.txtR0R7.setFont(font)
        self.txtR0R7.setObjectName("txtR0R7")
        self.gridLayout_2.addWidget(self.txtR0R7, 0, 14, 1, 1)
        self.label_10 = QtWidgets.QLabel(self.gbSimulation)
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self.label_10.setFont(font)
        self.label_10.setObjectName("label_10")
        self.gridLayout_2.addWidget(self.label_10, 0, 7, 1, 1)
        self.txtR0R3 = QtWidgets.QLineEdit(self.gbSimulation)
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self.txtR0R3.setFont(font)
        self.txtR0R3.setObjectName("txtR0R3")
        self.gridLayout_2.addWidget(self.txtR0R3, 0, 6, 1, 1)
        self.label_11 = QtWidgets.QLabel(self.gbSimulation)
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self.label_11.setFont(font)
        self.label_11.setObjectName("label_11")
        self.gridLayout_2.addWidget(self.label_11, 0, 9, 1, 1)
        self.label_9 = QtWidgets.QLabel(self.gbSimulation)
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self.label_9.setFont(font)
        self.label_9.setObjectName("label_9")
        self.gridLayout_2.addWidget(self.label_9, 0, 5, 1, 1)
        self.label_12 = QtWidgets.QLabel(self.gbSimulation)
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self.label_12.setFont(font)
        self.label_12.setObjectName("label_12")
        self.gridLayout_2.addWidget(self.label_12, 0, 11, 1, 1)
        self.label_14 = QtWidgets.QLabel(self.gbSimulation)
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self.label_14.setFont(font)
        self.label_14.setObjectName("label_14")
        self.gridLayout_2.addWidget(self.label_14, 0, 15, 1, 1)
        self.txtR0R2 = QtWidgets.QLineEdit(self.gbSimulation)
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self.txtR0R2.setFont(font)
        self.txtR0R2.setObjectName("txtR0R2")
        self.gridLayout_2.addWidget(self.txtR0R2, 0, 4, 1, 1)
        self.txtR0R8 = QtWidgets.QLineEdit(self.gbSimulation)
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self.txtR0R8.setFont(font)
        self.txtR0R8.setObjectName("txtR0R8")
        self.gridLayout_2.addWidget(self.txtR0R8, 0, 16, 1, 1)
        self.horizontalLayout_4.addLayout(self.gridLayout_2)
        self.verticalLayout_4.addLayout(self.horizontalLayout_4)
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.label_16 = QtWidgets.QLabel(self.gbSimulation)
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self.label_16.setFont(font)
        self.label_16.setObjectName("label_16")
        self.horizontalLayout_5.addWidget(self.label_16)
        self.txtTStopNDays = QtWidgets.QLineEdit(self.gbSimulation)
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self.txtTStopNDays.setFont(font)
        self.txtTStopNDays.setObjectName("txtTStopNDays")
        self.horizontalLayout_5.addWidget(self.txtTStopNDays)
        self.btnSingleSim = QtWidgets.QPushButton(self.gbSimulation)
        self.btnSingleSim.setEnabled(False)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.btnSingleSim.setFont(font)
        self.btnSingleSim.setObjectName("btnSingleSim")
        self.horizontalLayout_5.addWidget(self.btnSingleSim)
        self.label_15 = QtWidgets.QLabel(self.gbSimulation)
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self.label_15.setFont(font)
        self.label_15.setObjectName("label_15")
        self.horizontalLayout_5.addWidget(self.label_15)
        self.lineEdit_2 = QtWidgets.QLineEdit(self.gbSimulation)
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self.lineEdit_2.setFont(font)
        self.lineEdit_2.setObjectName("lineEdit_2")
        self.horizontalLayout_5.addWidget(self.lineEdit_2)
        self.btnMCSim = QtWidgets.QPushButton(self.gbSimulation)
        self.btnMCSim.setEnabled(False)
        self.btnMCSim.setObjectName("btnMCSim")
        self.horizontalLayout_5.addWidget(self.btnMCSim)
        self.verticalLayout_4.addLayout(self.horizontalLayout_5)
        self.verticalLayout_2.addWidget(self.gbSimulation)
        self.gbOutputFolder = QtWidgets.QGroupBox(self.centralwidget)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.gbOutputFolder.setFont(font)
        self.gbOutputFolder.setObjectName("gbOutputFolder")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.gbOutputFolder)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.txtOutputFolder = QtWidgets.QLineEdit(self.gbOutputFolder)
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self.txtOutputFolder.setFont(font)
        self.txtOutputFolder.setObjectName("txtOutputFolder")
        self.horizontalLayout_3.addWidget(self.txtOutputFolder)
        self.btnBrowseOutputFolder = QtWidgets.QPushButton(self.gbOutputFolder)
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self.btnBrowseOutputFolder.setFont(font)
        self.btnBrowseOutputFolder.setObjectName("btnBrowseOutputFolder")
        self.horizontalLayout_3.addWidget(self.btnBrowseOutputFolder)
        self.verticalLayout_2.addWidget(self.gbOutputFolder)
        self.gbConfigurationFile = QtWidgets.QGroupBox(self.centralwidget)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.gbConfigurationFile.setFont(font)
        self.gbConfigurationFile.setObjectName("gbConfigurationFile")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.gbConfigurationFile)
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.label_4 = QtWidgets.QLabel(self.gbConfigurationFile)
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")
        self.horizontalLayout_2.addWidget(self.label_4)
        self.txtConfigFileLoc = QtWidgets.QLineEdit(self.gbConfigurationFile)
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self.txtConfigFileLoc.setFont(font)
        self.txtConfigFileLoc.setObjectName("txtConfigFileLoc")
        self.horizontalLayout_2.addWidget(self.txtConfigFileLoc)
        self.btnConfigLoad = QtWidgets.QPushButton(self.gbConfigurationFile)
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self.btnConfigLoad.setFont(font)
        self.btnConfigLoad.setObjectName("btnConfigLoad")
        self.horizontalLayout_2.addWidget(self.btnConfigLoad)
        self.btnConfigSaveAs = QtWidgets.QPushButton(self.gbConfigurationFile)
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self.btnConfigSaveAs.setFont(font)
        self.btnConfigSaveAs.setObjectName("btnConfigSaveAs")
        self.horizontalLayout_2.addWidget(self.btnConfigSaveAs)
        self.btnSaveConfig = QtWidgets.QPushButton(self.gbConfigurationFile)
        self.btnSaveConfig.setEnabled(False)
        self.btnSaveConfig.setObjectName("btnSaveConfig")
        self.horizontalLayout_2.addWidget(self.btnSaveConfig)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        self.verticalLayout_2.addWidget(self.gbConfigurationFile)
        self.gbApplicationLog = QtWidgets.QGroupBox(self.centralwidget)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.gbApplicationLog.setFont(font)
        self.gbApplicationLog.setObjectName("gbApplicationLog")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.gbApplicationLog)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.txtConsole = QtWidgets.QTextEdit(self.gbApplicationLog)
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self.txtConsole.setFont(font)
        self.txtConsole.setObjectName("txtConsole")
        self.verticalLayout_3.addWidget(self.txtConsole)
        self.progressBar = QtWidgets.QProgressBar(self.gbApplicationLog)
        self.progressBar.setEnabled(True)
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self.progressBar.setFont(font)
        self.progressBar.setProperty("value", 0)
        self.progressBar.setTextVisible(True)
        self.progressBar.setObjectName("progressBar")
        self.verticalLayout_3.addWidget(self.progressBar)
        self.verticalLayout_2.addWidget(self.gbApplicationLog)
        SpreadEEGui.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(SpreadEEGui)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 920, 21))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        self.menuView = QtWidgets.QMenu(self.menubar)
        self.menuView.setObjectName("menuView")
        SpreadEEGui.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(SpreadEEGui)
        self.statusbar.setObjectName("statusbar")
        SpreadEEGui.setStatusBar(self.statusbar)
        self.actionDisplayActiveCaseGraph = QtWidgets.QAction(SpreadEEGui)
        self.actionDisplayActiveCaseGraph.setCheckable(True)
        self.actionDisplayActiveCaseGraph.setChecked(True)
        self.actionDisplayActiveCaseGraph.setObjectName("actionDisplayActiveCaseGraph")
        self.actionLoad_config_file = QtWidgets.QAction(SpreadEEGui)
        self.actionLoad_config_file.setObjectName("actionLoad_config_file")
        self.actionSave_config_file_as = QtWidgets.QAction(SpreadEEGui)
        self.actionSave_config_file_as.setObjectName("actionSave_config_file_as")
        self.menuFile.addAction(self.actionLoad_config_file)
        self.menuFile.addAction(self.actionSave_config_file_as)
        self.menuView.addAction(self.actionDisplayActiveCaseGraph)
        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuView.menuAction())

        self.retranslateUi(SpreadEEGui)
        QtCore.QMetaObject.connectSlotsByName(SpreadEEGui)

    def retranslateUi(self, SpreadEEGui):
        _translate = QtCore.QCoreApplication.translate
        SpreadEEGui.setWindowTitle(_translate("SpreadEEGui", "[Virus Spread in EE] Analysis Tool"))
        self.gbModelInitialization.setTitle(_translate("SpreadEEGui", "Step 1: Model initialization: Parameters below cannot be changed unless a new simulation is created"))
        self.label.setText(_translate("SpreadEEGui", "Data source file URL:"))
        self.txtCovid19TestSrc.setText(_translate("SpreadEEGui", "https://opendata.digilugu.ee/opendata_covid19_test_results.csv"))
        self.btnFetchDataAndProcess.setText(_translate("SpreadEEGui", "Create new simulation"))
        self.lblDataProcesses.setText(_translate("SpreadEEGui", "Data processed: <span style=\"color:red\">NO</span>"))
        self.label_3.setText(_translate("SpreadEEGui", "Fraction of generation interval for new infections ="))
        self.label_2.setText(_translate("SpreadEEGui", "Stop initialization N days ago from current date ="))
        self.txtFractionGenerationInt.setText(_translate("SpreadEEGui", "0.8"))
        self.txtNDaysAgo.setText(_translate("SpreadEEGui", "10"))
        self.label_5.setText(_translate("SpreadEEGui", "Prob (someone infected infects someone new at the start of simulation) ="))
        self.txtProbInitialInfect.setText(_translate("SpreadEEGui", "0.1"))
        self.gbSimulation.setTitle(_translate("SpreadEEGui", "Step 2: Run simulation: Parameters below can be changed to analyze different simulation outcomes"))
        self.label_18.setText(_translate("SpreadEEGui", "Prob (mobility between regions is unrestricted) [0 = completely restricted, 1 = unrestricted, affects all regions] ="))
        self.txtSimParamProbMobBetweenReg.setText(_translate("SpreadEEGui", "1.0"))
        self.txtSimParamStayingActive.setText(_translate("SpreadEEGui", "0.005"))
        self.label_17.setText(_translate("SpreadEEGui", "Fraction of those staying active in their regions [affects mobility, larger number → less mobility between regions] ="))
        self.label_19.setText(_translate("SpreadEEGui", "Prob (mobility between Saaremaa and mainland is unrestricted) [0 = completely restricted, 1 = unrestricted] ="))
        self.txtSimParamProbMobilityBetweenSaareUnrestr.setText(_translate("SpreadEEGui", "1.0"))
        self.label_6.setText(_translate("SpreadEEGui", "R0 for regions"))
        self.txtR0R5.setText(_translate("SpreadEEGui", "0.8"))
        self.txtR0R6.setText(_translate("SpreadEEGui", "0.6"))
        self.txtR0R1.setText(_translate("SpreadEEGui", "0.9"))
        self.label_13.setText(_translate("SpreadEEGui", "R7="))
        self.txtR0R4.setText(_translate("SpreadEEGui", "0.8"))
        self.label_8.setText(_translate("SpreadEEGui", "R2="))
        self.label_7.setText(_translate("SpreadEEGui", "R1="))
        self.txtR0R7.setText(_translate("SpreadEEGui", "0.8"))
        self.label_10.setText(_translate("SpreadEEGui", "R4="))
        self.txtR0R3.setText(_translate("SpreadEEGui", "0.8"))
        self.label_11.setText(_translate("SpreadEEGui", "R5="))
        self.label_9.setText(_translate("SpreadEEGui", "R3="))
        self.label_12.setText(_translate("SpreadEEGui", "R6="))
        self.label_14.setText(_translate("SpreadEEGui", "R8="))
        self.txtR0R2.setText(_translate("SpreadEEGui", "0.8"))
        self.txtR0R8.setText(_translate("SpreadEEGui", "0.8"))
        self.label_16.setText(_translate("SpreadEEGui", "T_stop [days] ="))
        self.txtTStopNDays.setText(_translate("SpreadEEGui", "90"))
        self.btnSingleSim.setText(_translate("SpreadEEGui", "Run single simulation with logging"))
        self.label_15.setText(_translate("SpreadEEGui", "Monte-Carlo simulation count ="))
        self.lineEdit_2.setText(_translate("SpreadEEGui", "1000"))
        self.btnMCSim.setText(_translate("SpreadEEGui", "Run Monte-Carlo simulations"))
        self.gbOutputFolder.setTitle(_translate("SpreadEEGui", "Output folder: Results from simulation runs will be saved to this folder"))
        self.btnBrowseOutputFolder.setText(_translate("SpreadEEGui", "Browse..."))
        self.gbConfigurationFile.setTitle(_translate("SpreadEEGui", "Simulation configuration file: Contains the initialized model and configurable simulation parameters, can be shared with others"))
        self.label_4.setText(_translate("SpreadEEGui", "Location on disk"))
        self.btnConfigLoad.setText(_translate("SpreadEEGui", "Load..."))
        self.btnConfigSaveAs.setText(_translate("SpreadEEGui", "Save as..."))
        self.btnSaveConfig.setText(_translate("SpreadEEGui", "Save all config"))
        self.gbApplicationLog.setTitle(_translate("SpreadEEGui", "Application log"))
        self.menuFile.setTitle(_translate("SpreadEEGui", "File"))
        self.menuView.setTitle(_translate("SpreadEEGui", "View"))
        self.actionDisplayActiveCaseGraph.setText(_translate("SpreadEEGui", "Display total active cases at init"))
        self.actionLoad_config_file.setText(_translate("SpreadEEGui", "Load config file..."))
        self.actionSave_config_file_as.setText(_translate("SpreadEEGui", "Save config file as..."))

