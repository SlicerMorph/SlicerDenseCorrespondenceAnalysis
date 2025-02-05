import os
import unittest
import vtk, qt, ctk, slicer
from slicer.ScriptedLoadableModule import *
import logging
import fnmatch
import  numpy as np
import random
import math
from datetime import datetime
import re
import csv
import vtk.util.numpy_support as vtk_np
from pathlib import Path
import shutil

#
# DeCA
#

class DeCA(ScriptedLoadableModule):
  """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
    """

  def __init__(self, parent):
    ScriptedLoadableModule.__init__(self, parent)
    self.parent.title = "DeCA" # TODO make this more human readable by adding spaces
    self.parent.categories = ["DeCA Toolbox"]
    self.parent.dependencies = []
    self.parent.contributors = ["Sara Rolfe (UW)"] # replace with "Firstname Lastname (Organization)"
    self.parent.helpText = """
      This module provides several flexible workflows for finding and analyzing dense correspondence points between models.
      """
    self.parent.helpText += self.getDefaultModuleDocumentationLink()
    self.parent.acknowledgementText = """

      """ # replace with organization, grant and thanks.

#
# DeCAWidget
#

class DeCAWidget(ScriptedLoadableModuleWidget):
  """Uses ScriptedLoadableModuleWidget base class, available at:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
    """

  def setup(self):
    ScriptedLoadableModuleWidget.setup(self)

    # Set up tabs to split workflow
    tabsWidget = qt.QTabWidget()
    DeCATab = qt.QWidget()
    DeCATabLayout = qt.QFormLayout(DeCATab)
    DeCALTab = qt.QWidget()
    DeCALTabLayout = qt.QFormLayout(DeCALTab)
    visualizeTab = qt.QWidget()
    visualizeTabLayout = qt.QFormLayout(visualizeTab)

    #tabsWidget.addTab(DeCATab, "DeCA")
    tabsWidget.addTab(DeCALTab, "DeCAL")
    #tabsWidget.addTab(visualizeTab, "Visualize Results")

    self.layout.addWidget(tabsWidget)

    ################################### DeCA Tab ###################################
    # Layout within the DeCA tab
    DeCAWidget=ctk.ctkCollapsibleButton()
    DeCAWidgetLayout = qt.QFormLayout(DeCAWidget)
    DeCAWidget.text = "Dense Correspondence I/0"
    DeCATabLayout.addRow(DeCAWidget)

    #
    # Select Atlas Type
    #
    self.calculateAtlasOptionDC=qt.QRadioButton()
    self.calculateAtlasOptionDC.setChecked(True)
    self.loadAtlasOptionDC=qt.QRadioButton()
    DCAtlasButtonGroup = qt.QButtonGroup(DeCAWidget)
    DCAtlasButtonGroup.addButton(self.calculateAtlasOptionDC)
    DCAtlasButtonGroup.addButton(self.loadAtlasOptionDC)
    DeCAWidgetLayout.addRow("Create atlas: ", self.calculateAtlasOptionDC)
    DeCAWidgetLayout.addRow("Load atlas: ", self.loadAtlasOptionDC)

    #
    # Hidden atlas options
    self.atlasCollapsibleButton = ctk.ctkCollapsibleButton()
    self.atlasCollapsibleButton.text = "Atlas Options"
    self.atlasCollapsibleButton.collapsed = True
    self.atlasCollapsibleButton.enabled = False
    DeCAWidgetLayout.addRow(self.atlasCollapsibleButton)
    atlasOptionLayout = qt.QFormLayout(self.atlasCollapsibleButton)

    #
    # Select base mesh
    #
    self.DCBaseModelSelector = ctk.ctkPathLineEdit()
    self.DCBaseModelSelector.filters  = ctk.ctkPathLineEdit().Files
    self.DCBaseModelSelector.nameFilters=["Model (*.ply *.stl *.obj *.vtk *.vtp)"]
    atlasOptionLayout.addRow("Atlas model: ", self.DCBaseModelSelector)

    #
    # Select base landmarks
    #
    self.DCBaseLMSelector = ctk.ctkPathLineEdit()
    self.DCBaseLMSelector.filters  = ctk.ctkPathLineEdit().Files
    self.DCBaseLMSelector.nameFilters=["Point set (*.fcsv *.json *.mrk.json"]
    atlasOptionLayout.addRow("Atlas landmarks: ", self.DCBaseLMSelector)

    #
    # Select Analysis Type
    #
    self.analysisTypeShape=qt.QRadioButton()
    self.analysisTypeShape.setChecked(True)
    self.analysisTypeSymmetry=qt.QRadioButton()
    DCAnalysisButtonGroup = qt.QButtonGroup(DeCAWidget)
    DCAnalysisButtonGroup.addButton(self.analysisTypeShape)
    DCAnalysisButtonGroup.addButton(self.analysisTypeSymmetry)
    DeCAWidgetLayout.addRow("Shape analysis: ", self.analysisTypeShape)
    DeCAWidgetLayout.addRow("Symmetry analysis: ", self.analysisTypeSymmetry)

    #
    # Hidden symmetry options
    #
    self.symmetryCollapsibleButton = ctk.ctkCollapsibleButton()
    self.symmetryCollapsibleButton.text = "Symmetry Options"
    self.symmetryCollapsibleButton.collapsed = True
    self.symmetryCollapsibleButton.enabled = False
    DeCAWidgetLayout.addRow(self.symmetryCollapsibleButton)
    symmetryOptionLayout = qt.QFormLayout(self.symmetryCollapsibleButton)
    self.landmarkIndexText=qt.QLineEdit()
    self.landmarkIndexText.setToolTip("No spaces. Seperate numbers by commas.  Example:  2,1,3,5,4")
    symmetryOptionLayout.addRow('Mirror landmark index', self.landmarkIndexText)

    #
    # Select model directory
    #
    self.meshDirectoryDC=ctk.ctkPathLineEdit()
    self.meshDirectoryDC.filters = ctk.ctkPathLineEdit.Dirs
    self.meshDirectoryDC.setToolTip( "Select directory containing models" )
    DeCAWidgetLayout.addRow("Model directory: ", self.meshDirectoryDC)

    #
    # Select landmark directory
    #
    self.landmarkDirectoryDC=ctk.ctkPathLineEdit()
    self.landmarkDirectoryDC.filters = ctk.ctkPathLineEdit.Dirs
    self.landmarkDirectoryDC.setToolTip( "Select directory containing landmarks" )
    DeCAWidgetLayout.addRow("Landmark directory: ", self.landmarkDirectoryDC)

    #
    # Select DeCA output directory
    #
    self.outputDirectoryDC=ctk.ctkPathLineEdit()
    self.outputDirectoryDC.filters = ctk.ctkPathLineEdit.Dirs
    self.outputDirectoryDC.setToolTip( "Select directory for DeCA output" )
    DeCAWidgetLayout.addRow("DeCA output directory: ", self.outputDirectoryDC)

    #
    # Remove scale option
    #
    self.removeScaleCheckBoxDC = qt.QCheckBox()
    self.removeScaleCheckBoxDC.checked = False
    self.removeScaleCheckBoxDC.setToolTip("If checked, DeCA alignment will include isotropic scaling.")
    DeCAWidgetLayout.addRow("Remove scale: ", self.removeScaleCheckBoxDC)

    #
    # Error checking directory option
    #
    self.writeErrorCheckBox = qt.QCheckBox()
    self.writeErrorCheckBox.checked = False
    self.writeErrorCheckBox.setToolTip("If checked, DeCA will create a directory of results for use in estimating point correspondence error.")
    DeCAWidgetLayout.addRow("Create output for error checking: ", self.writeErrorCheckBox)

    #
    # Run DeCA Button
    #
    self.ApplyButtonDC = qt.QPushButton("Run DeCA")
    self.ApplyButtonDC.toolTip = "Run non-rigid alignment"
    self.ApplyButtonDC.enabled = True
    DeCAWidgetLayout.addRow(self.ApplyButtonDC)

    #
    # Log Information
    #
    self.logInfoDC = qt.QPlainTextEdit()
    self.logInfoDC.setPlaceholderText("DeCA log information")
    self.logInfoDC.setReadOnly(True)
    DeCAWidgetLayout.addRow(self.logInfoDC)

    # Connections
#    self.baseModelSelectorDC.connect('validInputChanged(bool)', self.onSelectDC)
#    self.baseLMSelectorCD.connect('validInputChanged(bool)', self.onSelectDC)
#    self.meshDirectoryDC.connect('validInputChanged(bool)', self.onSelectDC)
#    self.landmarkDirectoryDC.connect('validInputChanged(bool)', self.onSelectDC)
#    self.outputDirectoryDC.connect('validInputChanged(bool)', self.onSelectDC)
    self.analysisTypeShape.connect('toggled(bool)', self.onToggleAnalysis)
    self.analysisTypeSymmetry.connect('toggled(bool)', self.onToggleAnalysis)
    self.calculateAtlasOptionDC.connect('toggled(bool)', self.onToggleAtlas)
    self.loadAtlasOptionDC.connect('toggled(bool)', self.onToggleAtlas)
    self.ApplyButtonDC.connect('clicked(bool)', self.onDCApplyButton)

    # Connections
#    self.meshSelect.connect("currentNodeChanged(vtkMRMLNode*)", self.onVisualizeMeshSelect)
#    self.subjectIDBox.connect("currentIndexChanged(int)", self.onSubjectIDSelect)

    ################################### DeCAL Tab ###################################
    # Layout within the DeCA tab
    DeCALWidget=ctk.ctkCollapsibleButton()
    DeCALWidgetLayout = qt.QFormLayout(DeCALWidget)
    DeCALWidget.text = "Dense Correspondence Landmarking"
    DeCALTabLayout.addRow(DeCALWidget)

    #
    # Select Atlas Type
    #
    DCLAtlasButtonGroup = qt.QButtonGroup(DeCALWidget)
    self.calculateAtlasOptionDCL=qt.QRadioButton()
    self.calculateAtlasOptionDCL.setChecked(True)
    DCLAtlasButtonGroup.addButton(self.calculateAtlasOptionDCL)
    self.loadAtlasOptionDCL=qt.QRadioButton()
    self.loadAtlasOptionDCL.setChecked(False)
    DCLAtlasButtonGroup.addButton(self.loadAtlasOptionDCL)
    DeCALWidgetLayout.addRow("Create atlas: ", self.calculateAtlasOptionDCL)
    DeCALWidgetLayout.addRow("Load atlas: ", self.loadAtlasOptionDCL)

    #
    # Hidden atlas options
    #
    self.atlasCollapsibleButtonDCL = ctk.ctkCollapsibleButton()
    self.atlasCollapsibleButtonDCL.text = "Load Atlas"
    self.atlasCollapsibleButtonDCL.collapsed = True
    self.atlasCollapsibleButtonDCL.enabled = False
    DeCALWidgetLayout.addRow(self.atlasCollapsibleButtonDCL)
    DeCALAtlasOptionLayout = qt.QFormLayout(self.atlasCollapsibleButtonDCL)

    #
    # Select base mesh
    #
    self.DCLBaseModelSelector = ctk.ctkPathLineEdit()
    self.DCLBaseModelSelector.filters  = ctk.ctkPathLineEdit().Files
    self.DCLBaseModelSelector.nameFilters=["Model (*.ply *.stl *.obj *.vtk *.vtp)"]
    DeCALAtlasOptionLayout.addRow("Atlas model: ", self.DCLBaseModelSelector)

    #
    # Select base landmarks
    #
    self.DCLBaseLMSelector = ctk.ctkPathLineEdit()
    self.DCLBaseLMSelector.filters  = ctk.ctkPathLineEdit().Files
    self.DCLBaseLMSelector.nameFilters=["Point set (*.fcsv *.json *.mrk.json"]
    DeCALAtlasOptionLayout.addRow("Atlas landmarks: ", self.DCLBaseLMSelector)

    #
    # Select meshes directory
    #
    self.meshDirectoryDCL=ctk.ctkPathLineEdit()
    self.meshDirectoryDCL.filters = ctk.ctkPathLineEdit.Dirs
    self.meshDirectoryDCL.setToolTip( "Select directory containing models" )
    DeCALWidgetLayout.addRow("Model directory: ", self.meshDirectoryDCL)

    #
    # Select landmarks directory
    #
    self.landmarkDirectoryDCL=ctk.ctkPathLineEdit()
    self.landmarkDirectoryDCL.filters = ctk.ctkPathLineEdit.Dirs
    self.landmarkDirectoryDCL.setToolTip( "Select directory containing landmarks" )
    DeCALWidgetLayout.addRow("Landmark directory: ", self.landmarkDirectoryDCL)

    #
    # Select DeCA output directory
    #
    self.DCLOutputDirectory=ctk.ctkPathLineEdit()
    self.DCLOutputDirectory.filters = ctk.ctkPathLineEdit.Dirs
    self.DCLOutputDirectory.setToolTip( "Select directory for DeCAL output" )
    DeCALWidgetLayout.addRow("DeCAL output directory: ", self.DCLOutputDirectory)

    #
    # Set spacing tolerance
    #
    self.spacingTolerance = ctk.ctkSliderWidget()
    self.spacingTolerance.singleStep = .1
    self.spacingTolerance.minimum = 0
    self.spacingTolerance.maximum = 10
    self.spacingTolerance.value = 4
    self.spacingTolerance.setToolTip("Set tolerance of spacing as a percentage of the image diagonal")
    DeCALWidgetLayout.addRow("Point density adjustment: ", self.spacingTolerance)

    #
    # Generate Atlas Button
    #
    self.getAtlasButton = qt.QPushButton("Create\Load atlas")
    self.getAtlasButton.toolTip = "Generate a new atlas model and landmark set from data"
    self.getAtlasButton.enabled = False
    DeCALWidgetLayout.addRow(self.getAtlasButton)

    #
    # Get Subsample Rate Button
    #
    self.getPointNumberButton = qt.QPushButton("Run subsampling")
    self.getPointNumberButton.toolTip = "Get the number of output points that will be generated"
    self.getPointNumberButton.enabled = False
    DeCALWidgetLayout.addRow(self.getPointNumberButton)

    #
    # Apply Button
    #
    self.DCLApplyButton = qt.QPushButton("Run DeCAL")
    self.DCLApplyButton.toolTip = "Generate a set of corresponding landmarks"
    self.DCLApplyButton.enabled = False
    DeCALWidgetLayout.addRow(self.DCLApplyButton)

    #
    # Log Information
    #
    self.logInfoDCL = qt.QPlainTextEdit()
    self.logInfoDCL.setPlaceholderText("DeCAL log information")
    self.logInfoDCL.setReadOnly(True)
    DeCALWidgetLayout.addRow(self.logInfoDCL)

    #
    # Subsetting menu
    #
    self.subsetCollapsibleButton = ctk.ctkCollapsibleButton()
    self.subsetCollapsibleButton.text = "Subset output points"
    self.subsetCollapsibleButton.collapsed = True
    self.subsetCollapsibleButton.enabled = True
    DeCALWidgetLayout.addRow(self.subsetCollapsibleButton)
    DeCALSubsetLayout = qt.QFormLayout(self.subsetCollapsibleButton)

    #
    # Select landmark node
    #
    self.pointSelection = slicer.qMRMLNodeComboBox()
    self.pointSelection.nodeTypes = ( ("vtkMRMLMarkupsFiducialNode"), "" )
    self.pointSelection.setToolTip( "Atlas landmarks with subset points selected" )
    self.pointSelection.selectNodeUponCreation = False
    self.pointSelection.noneEnabled = True
    self.pointSelection.addEnabled = False
    self.pointSelection.removeEnabled = False
    self.pointSelection.showHidden = False
    self.pointSelection.setMRMLScene( slicer.mrmlScene )
    DeCALSubsetLayout.addRow("Atlas landmarks: ", self.pointSelection)

    #
    # Select DeCAL output directory
    #
    self.DCLLandmarkDirectory=ctk.ctkPathLineEdit()
    self.DCLLandmarkDirectory.filters = ctk.ctkPathLineEdit.Dirs
    self.DCLLandmarkDirectory.setToolTip( "Select directory for DeCAL sampled landmarks to subset" )
    DeCALSubsetLayout.addRow("DeCAL landmark directory: ", self.DCLLandmarkDirectory)

    #
    # Apply Subsetting Button
    #
    self.subsetApplyButton = qt.QPushButton("Run subsetting")
    self.subsetApplyButton.toolTip = "Generate a subset of corresponding landmarks"
    self.subsetApplyButton.enabled = False
    DeCALSubsetLayout.addRow(self.subsetApplyButton)

    # connections
    self.calculateAtlasOptionDCL.connect('toggled(bool)', self.onToggleAtlasDCL)
    self.loadAtlasOptionDCL.connect('toggled(bool)', self.onToggleAtlasDCL)
    self.DCLBaseModelSelector.connect('validInputChanged(bool)', self.onAtlasSelect)
    self.DCLBaseLMSelector.connect('validInputChanged(bool)', self.onAtlasSelect)
    self.meshDirectoryDCL.connect('validInputChanged(bool)', self.onAtlasSelect)
    self.landmarkDirectoryDCL.connect('validInputChanged(bool)', self.onAtlasSelect)
    self.DCLOutputDirectory.connect('validInputChanged(bool)', self.onAtlasSelect)
    self.getAtlasButton.connect('clicked(bool)', self.onGenerateAtlasButton)
    self.getPointNumberButton.connect('clicked(bool)', self.onGetPointNumberButton)
    self.DCLApplyButton.connect('clicked(bool)', self.onDCLApplyButton)
    self.subsetApplyButton.connect('clicked(bool)', self.onSubsetApplyButton)
    self.pointSelection.connect('currentNodeChanged(vtkMRMLNode*)', self.onPointSelectionSelect)
    self.DCLLandmarkDirectory.connect('validInputChanged(bool)', self.onDCLLandmarkDirectorySelect)

    ################################### Visualize Tab ###################################
    # Layout within the tab
    visualizeWidget=ctk.ctkCollapsibleButton()
    visualizeWidgetLayout = qt.QFormLayout(visualizeWidget)
    visualizeWidget.text = "Visualize the output feature heat maps"
    visualizeTabLayout.addRow(visualizeWidget)

    #
    # Select output model
    #
    self.meshSelect = slicer.qMRMLNodeComboBox()
    self.meshSelect.nodeTypes = ( ("vtkMRMLModelNode"), "" )
    self.meshSelect.setToolTip( "Select model node with result arrays" )
    self.meshSelect.selectNodeUponCreation = False
    self.meshSelect.noneEnabled = True
    self.meshSelect.addEnabled = False
    self.meshSelect.removeEnabled = False
    self.meshSelect.showHidden = False
    self.meshSelect.setMRMLScene( slicer.mrmlScene )
    visualizeWidgetLayout.addRow("Result Model: ", self.meshSelect)

    #
    # Select Subject ID
    #
    self.subjectIDBox=qt.QComboBox()
    self.subjectIDBox.enabled = False
    visualizeWidgetLayout.addRow("Subject ID: ", self.subjectIDBox)

  ################################### GUI SUpport Functions
  def setUpDeCADir(self, outDir, symmetryOption=False, errorDirectoryOption=False, DeCALOption=False, loadAtlasOption = False):
    dateTimeStamp = datetime.now().strftime('%Y_%m-%d_%H_%M_%S')
    outputFolderDC = os.path.join(outDir, dateTimeStamp)
    fileNameDictionary = {}
    try:
      os.makedirs(outputFolderDC)
      alignedLMFolderDC = os.path.join(outputFolderDC, "alignedLMs")
      os.makedirs(alignedLMFolderDC)
      alignedModelFolderDC = os.path.join(outputFolderDC, "alignedModels")
      os.makedirs(alignedModelFolderDC)
      # initialize the filename dictionary
      fileNameDictionary['output'] = str(outputFolderDC)
      fileNameDictionary['alignedLMs'] = str(alignedLMFolderDC)
      fileNameDictionary['alignedModels'] = str(alignedModelFolderDC)
      if not loadAtlasOption:
        tempLMFolderDC = os.path.join(outputFolderDC, "tempAlignedLMs")
        os.makedirs(tempLMFolderDC)
        tempModelFolderDC = os.path.join(outputFolderDC, "tempAlignedModels")
        os.makedirs(tempModelFolderDC)
        fileNameDictionary['tempAlignedLMs'] = str(tempLMFolderDC)
        fileNameDictionary['tempAlignedModels'] = str(tempModelFolderDC)
      if symmetryOption:
        mirrorLMFolderDC = os.path.join(outputFolderDC, "mirrorLMs")
        os.makedirs(mirrorLMFolderDC)
        mirrorModelFolderDC = os.path.join(outputFolderDC, "mirrorModels")
        os.makedirs(mirrorModelFolderDC)
        fileNameDictionary['mirrorLMs'] = str(mirrorLMFolderDC)
        fileNameDictionary['mirrorModels'] = str(mirrorModelFolderDC)
      if errorDirectoryOption:
        errorCheckingFolderDC = os.path.join(outputFolderDC, "errorChecking")
        os.makedirs(errorCheckingFolderDC)
        fileNameDictionary['error'] = str(errorCheckingFolderDC)
      if DeCALOption:
        DeCALOutputFolder = os.path.join(outputFolderDC, "DeCALOutput")
        os.makedirs(DeCALOutputFolder)
        fileNameDictionary['DeCALOutput'] = str(DeCALOutputFolder)
    except:
      logging.debug('Result directory failed: Could not create output folder')
    return fileNameDictionary

  def onToggleAnalysis(self):
    if self.analysisTypeSymmetry.checked == True:
      self.symmetryCollapsibleButton.collapsed = False
      self.symmetryCollapsibleButton.enabled = True
    else:
      self.symmetryCollapsibleButton.collapsed = True
      self.symmetryCollapsibleButton.enabled = False

  def onToggleAtlas(self):
    if self.calculateAtlasOptionDC.checked == True:
      self.atlasCollapsibleButton.collapsed = True
      self.atlasCollapsibleButton.enabled = False
    else:
      self.atlasCollapsibleButton.collapsed = False
      self.atlasCollapsibleButton.enabled = True
    atlasPathSelected = bool (self.DCLBaseModelSelector.currentPath and self.DCLBaseLMSelector.currentPath )
    inputPathsSelected = bool (self.meshDirectoryDCL.currentPath and self.landmarkDirectoryDCL.currentPath and self.DCLOutputDirectory.currentPath )
    self.getAtlasButton.enabled = bool( (atlasPathSelected or self.calculateAtlasOptionDC) and inputPathsSelected )

  def onToggleAtlasDCL(self):
    if self.calculateAtlasOptionDCL.checked == True:
      self.atlasCollapsibleButtonDCL.collapsed = True
      self.atlasCollapsibleButtonDCL.enabled = False
    else:
      self.atlasCollapsibleButtonDCL.collapsed = False
      self.atlasCollapsibleButtonDCL.enabled = True
    self.onAtlasSelect()

  def onSubjectIDSelect(self):
    try:
      subjectID = self.subjectIDBox.currentText
      self.resultNode.GetDisplayNode().SetActiveScalarName(subjectID)
      self.resultNode.GetDisplayNode().SetAndObserveColorNodeID('vtkMRMLColorTableNodeFilePlasma.txt')
      print(subjectID)
    except:
      print("Error: No array found")

  def onVisualizeMeshSelect(self):
    if bool(self.meshSelect.currentNode()):
      self.resultNode = self.meshSelect.currentNode()
      self.resultNode.GetDisplayNode().SetVisibility(True)
      self.resultNode.GetDisplayNode().SetScalarVisibility(True)
      resultData = self.resultNode.GetPolyData().GetPointData()
      self.subjectIDBox.enabled = True
      arrayNumber = resultData.GetNumberOfArrays()
      if arrayNumber > 0:
        for i in range(resultData.GetNumberOfArrays()):
          arrayName = resultData.GetArrayName(i)
          self.subjectIDBox.addItem(arrayName)
      else:
        self.subjectIDBox.clear()
        self.subjectIDBox.enabled = False

  def onSelect(self):
    self.applyButton.enabled = bool (self.outpuDirectory.currentPath )

  def onAtlasSelect(self):
    atlasPathSelected = bool (self.DCLBaseModelSelector.currentPath and self.DCLBaseLMSelector.currentPath )
    inputPathsSelected = bool (self.meshDirectoryDCL.currentPath and self.landmarkDirectoryDCL.currentPath and self.DCLOutputDirectory.currentPath )
    self.getAtlasButton.enabled = bool( (atlasPathSelected or self.calculateAtlasOptionDCL.checked) and inputPathsSelected )

  def onPointSelectionSelect(self):
    self.subsetApplyButton.enabled = bool(self.DCLLandmarkDirectory.currentPath and self.pointSelection.currentNode())

  def onDCLLandmarkDirectorySelect(self):
    self.subsetApplyButton.enabled = bool(self.DCLLandmarkDirectory.currentPath and self.pointSelection.currentNode())

  def onDCApplyButton(self):
    logic = DeCALogic()
    self.setUpDeCADir(self.outputDirectoryDC.currentPath, self.logInfoDC, self.removeScaleCheckBoxDC.checked, self.writeErrorCheckBox.checkedself.logInfoDC, )
    if self.analysisTypeShape.checked:
      logic.runDCAlign(self.DCBaseModelSelector.currentPath, self.DCBaseLMSelector.currentPath, self.DCMeshDirectory.currentPath,
      self.DCLandmarkDirectory.currentPath, self.outputDirectoryDC.currentPath, self.CPDCheckBox.checked, self.writeErrorCheckBox.checked)
    else:
      logic.runDCAlignSymmetric(self.DCBaseModelSelector.currentPath, self.DCBaseLMSelector.currentPath, self.DCMeshDirectory.currentPath,
      self.DCLandmarkDirectory.currentPath, self.mirrorMeshSelector.currentPath, self.mirrorLMSelector.currentPath, self.outputDirectoryDC.currentPath,
      self.CPDCheckBox.checked, self.writeErrorCheckBox.checked, self.WriteCorrPointsCheckBox.checked)

  def onDCSelect(self):
    if self.analysisTypeShape.checked == True:
      self.DCApplyButton.enabled = bool (self.DCMeshDirectory.currentPath and self.DCLandmarkDirectory.currentPath
      and self.outputDirectoryDC.currentPath and self.DCBaseModelSelector.currentPath and self.DCBaseLMSelector.currentPath)
    else:
      self.DCApplyButton.enabled = bool (self.DCMeshDirectory.currentPath and self.DCLandmarkDirectory.currentPath
      and self.outputDirectoryDC.currentPath and self.DCBaseModelSelector.currentPath and self.DCBaseLMSelector.currentPath
      and self.mirrorMeshSelector.currentPath and self.mirrorLMSelector.currentPath)

  def onDCLSelect(self):
    self.getPointNumberButton.enabled = bool (self.DCLMeshDirectory.currentPath and self.DCLLandmarkDirectory.currentPath
    and self.DCLOutputDirectory.currentPath and self.DCLBaseModelSelector.currentPath and self.DCLBaseLMSelector.currentPath)

  def onGenerateAtlasButton(self):
    logic = DeCALogic()
    #set up output directory
    self.folderNames = self.setUpDeCADir(self.DCLOutputDirectory.currentPath, False, False, True, self.loadAtlasOptionDCL.checked)
    if self.folderNames == {}:
      return
    self.folderNames['originalLMs'] = self.landmarkDirectoryDCL.currentPath
    self.folderNames['originalModels'] = self.meshDirectoryDCL.currentPath
    if self.loadAtlasOptionDCL.checked:
      try:
        atlasModelPath = self.DCLBaseModelSelector.currentPath
        self.atlasModel = slicer.util.loadModel(atlasModelPath)
      except:
        print("Can't load from: ", atlasModelPath)
      try:
        atlasLMPath = self.DCLBaseLMSelector.currentPath
        self.atlasLMs = slicer.util.loadMarkups(atlasLMPath)
      except:
        print("Can't load from: ", atlasLMPath)
    else:
      closestToMeanLandmarkPath = logic.getClosestToMeanPath(self.folderNames['originalLMs'])
      tempBaseLMs = slicer.util.loadMarkups(os.path.join(self.folderNames['originalLMs'],closestToMeanLandmarkPath))
      subjectID = Path(closestToMeanLandmarkPath)
      while subjectID.suffix in {'.fcsv', '.mrk', '.json'}:
        subjectID = subjectID.with_suffix('')
      self.logInfoDCL.appendPlainText(f"Closest sample to mean: {subjectID}")
      tempBaseModel = logic.getModelFileByID(self.folderNames['originalModels'], subjectID)
      self.logInfoDCL.appendPlainText(f"Rigid Alignment to: {subjectID}")
      removeScale = True
      logic.runAlign(tempBaseModel, tempBaseLMs, self.folderNames['originalModels'], self.folderNames['originalLMs'], self.folderNames['tempAlignedModels'], self.folderNames['tempAlignedLMs'], removeScale)
      self.logInfoDCL.appendPlainText(f"Generating the average template")
      self.atlasModel, self.atlasLMs = logic.runMean(self.folderNames['tempAlignedLMs'], self.folderNames['tempAlignedModels'])
      slicer.mrmlScene.RemoveNode(tempBaseModel)
      slicer.mrmlScene.RemoveNode(tempBaseLMs)
      shutil.rmtree(self.folderNames['tempAlignedModels'])
      shutil.rmtree(self.folderNames['tempAlignedLMs'])
    atlasModelPath = os.path.join(self.folderNames['output'], 'decaAtlasModel.ply')
    self.logInfoDCL.appendPlainText(f"Saving atlas model to {atlasModelPath}")
    slicer.util.saveNode(self.atlasModel, atlasModelPath)
    atlasLMPath = os.path.join(self.folderNames['output'], 'decaAtlasLM.mrk.json')
    self.logInfoDCL.appendPlainText(f"Saving atlas landmarks to {atlasLMPath}")
    slicer.util.saveNode(self.atlasLMs, atlasLMPath)
    self.getPointNumberButton.enabled = True

  def onGetPointNumberButton(self):
    logic = DeCALogic()
    subsampledTemplate, pointNumber = logic.runCheckPoints(self.atlasModel, self.spacingTolerance.value)
    self.logInfoDCL.appendPlainText(f'The subsampled template has a total of {pointNumber} points.')
    self.DCLApplyButton.enabled = True

  def onDCLApplyButton(self):
    logic = DeCALogic()
    # rigidly align to template
    self.logInfoDCL.appendPlainText(f"Rigid alignment to the atlas")
    removeScale = True
    logic.runAlign(self.atlasModel, self.atlasLMs, self.folderNames['originalModels'], self.folderNames['originalLMs'],
    self.folderNames['alignedModels'], self.folderNames['alignedLMs'], removeScale)
    # generate point correspondences
    self.logInfoDCL.appendPlainText(f"Calculating point correspondences")
    atlasDenseLandmarks = logic.runDeCAL(self.atlasModel, self.atlasLMs, self.folderNames['alignedModels'],
    self.folderNames['alignedLMs'], self.folderNames['DeCALOutput'], self.spacingTolerance.value)
    # setup for optional subsetting
    self.pointSelection.setCurrentNode(atlasDenseLandmarks)
    self.DCLLandmarkDirectory.setCurrentPath(self.folderNames['DeCALOutput'])

  def onSubsetApplyButton(self):
    logic = DeCALogic()
    topDir = os.path.dirname(self.DCLLandmarkDirectory.currentPath)
    lmDirectorySubset = os.path.join(topDir, "DeCALSubset")
    os.makedirs(lmDirectorySubset)
    atlasNode = self.pointSelection.currentNode()
    lmDirectorySubset = logic.runSubsetLandmarks(atlasNode, self.DCLLandmarkDirectory.currentPath, lmDirectorySubset)

  def onMirrorButton(self):
    logic = DeCALogic()
    axis = [-1,1,1] #set symmetry to x-axis
    logic.runMirroring(self.symMeshDirectory.currentPath, self.symLandmarkDirectory.currentPath, self.mirrorMeshDirectory.currentPath,
      self.mirrorLMDirectory.currentPath, axis, self.landmarkIndexText.text)

#
# DeCALogic
#

class DeCALogic(ScriptedLoadableModuleLogic):
  """This class should implement all the actual
    computation done by your module.  The interface
    should be such that other python code can import
    this class and make use of the functionality without
    requiring an instance of the Widget.
    Uses ScriptedLoadableModuleLogic base class, available at:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
    """
  def runSubsetLandmarks(self, baseNode, lmDirectory, lmDirectorySubset):
    deletionIndex = []
    for i in range(baseNode.GetNumberOfControlPoints()):
      if not baseNode.GetNthControlPointSelected(i):
        deletionIndex.append(i)
    for lmFileName in os.listdir(lmDirectory):
      if(not lmFileName.startswith(".")):
        currentLMNode = slicer.util.loadMarkups(os.path.join(lmDirectory, lmFileName))
        for index in reversed(deletionIndex):
          currentLMNode.RemoveNthControlPoint(index)
      slicer.util.saveNode(currentLMNode, os.path.join(lmDirectorySubset, lmFileName))
      slicer.mrmlScene.RemoveNode(currentLMNode)

  def runCheckPoints(self, atlasNode, spacingTolerance):
    spacingPercentage = spacingTolerance/100
    templateModel = self.downsampleModel(atlasNode, spacingPercentage)
    return templateModel, templateModel.GetNumberOfPoints()

  def runDeCAL(self, baseNode, baseLMPath, meshDirectory, landmarkDirectory, outputDirectory, spacingTolerance):
    spacingPercentage = spacingTolerance/100
    loadOption=False
    baseLandmarks=self.fiducialNodeToPolyData(baseLMPath, loadOption).GetPoints()
    modelExt=['ply','stl','vtp', 'vtk']
    self.modelNames, models = self.importMeshes(meshDirectory, modelExt)
    landmarkNames, landmarks = self.importLandmarks(landmarkDirectory)
    self.outputDirectory = outputDirectory
    denseCorrespondenceGroup = self.denseCorrespondenceBaseMesh(landmarks, models, baseNode.GetPolyData(), baseLandmarks)
    # get downsampled template with index array
    indexArrayName = "indexArray"
    self.addIndexArray(baseNode, indexArrayName)
    templateModel = self.downsampleModel(baseNode, spacingPercentage)
    templateIndex = templateModel.GetPointData().GetArray(indexArrayName)
    # saving point correspondences
    if(templateIndex):
      sampleNumber = denseCorrespondenceGroup.GetNumberOfBlocks()
      print("sample number:", sampleNumber)
      for i in range(sampleNumber):
        alignedMesh = denseCorrespondenceGroup.GetBlock(i)
        alignedPointNode= slicer.mrmlScene.AddNewNodeByClass('vtkMRMLMarkupsFiducialNode',"alignedPoints")
        for j in range(templateIndex.GetNumberOfValues()):
          baseIndex = templateIndex.GetValue(j)
          alignedPoint = alignedMesh.GetPoint(baseIndex)
          alignedPointNode.AddControlPoint(alignedPoint, str(j))
        outputLMPath = os.path.join(outputDirectory, self.modelNames[i]+".mrk.json")
        slicer.util.saveNode(alignedPointNode, outputLMPath)
        slicer.mrmlScene.RemoveNode(alignedPointNode)
      # save base node correspondences
      basePointNode= slicer.mrmlScene.AddNewNodeByClass('vtkMRMLMarkupsFiducialNode',"atlasLandmarks")
      for j in range(templateIndex.GetNumberOfValues()):
        baseIndex = templateIndex.GetValue(j)
        basePoint = baseNode.GetPolyData().GetPoint(baseIndex)
        basePointNode.AddControlPoint(basePoint, str(j))
      baseLMPath = os.path.join(outputDirectory, "atlas.mrk.json")
      slicer.util.saveNode(basePointNode, baseLMPath)
      #slicer.mrmlScene.RemoveNode(basePointNode)
      return basePointNode
    else:
      print("No index found")
      return None

  def downsampleModel(self, model, spacingPercentage):
    points=model.GetPolyData()
    cleanFilter=vtk.vtkCleanPolyData()
    cleanFilter.SetToleranceIsAbsolute(False)
    cleanFilter.SetTolerance(spacingPercentage)
    cleanFilter.SetInputData(points)
    cleanFilter.Update()
    return cleanFilter.GetOutput()

  def addIndexArray(self, mesh, arrayName):
    # Array of original index values
    indexArray = vtk.vtkIntArray()
    indexArray.SetNumberOfComponents(1)
    indexArray.SetName(arrayName)
    for i in range(mesh.GetPolyData().GetNumberOfPoints()):
      indexArray.InsertNextValue(i)
    mesh.GetPolyData().GetPointData().AddArray(indexArray)

  def computeNormals(self, inputModel):
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(inputModel.GetPolyData())
    normals.SetAutoOrientNormals(True)
    normals.Update()
    inputModel.SetAndObservePolyData(normals.GetOutput())

  def runMirroring(self, meshDirectory, lmDirectory, mirrorMeshDirectory, mirrorLMDirectory, mirrorAxis, mirrorIndexText, slmDirectory, outputSLMDirectory, mirrorSLMIndexText):
    mirrorMatrix = vtk.vtkMatrix4x4()
    mirrorMatrix.SetElement(0, 0, mirrorAxis[0])
    mirrorMatrix.SetElement(1, 1, mirrorAxis[1])
    mirrorMatrix.SetElement(2, 2, mirrorAxis[2])
    lmFileList = os.listdir(lmDirectory)
    point=[0,0,0]
    #get order of mirrored sets
    if len(mirrorIndexText) != 0:
      mirrorIndexList=mirrorIndexText.split(",")
      mirrorIndexList=[int(x) for x in mirrorIndexList]
      mirrorIndex=np.asarray(mirrorIndexList)
    else:
      print("Error: no landmark index for mirrored mesh")
    semilandmarkOption = bool(slmDirectory and outputSLMDirectory and (len(mirrorSLMIndexText) != 0 ))
    if semilandmarkOption:
      mirrorSLMIndexList=mirrorSLMIndexText.split(",")
      mirrorSLMIndexList=[int(x) for x in mirrorSLMIndexList]
      mirrorSLMIndex=np.asarray(mirrorSLMIndexList)
    for meshFileName in os.listdir(meshDirectory):
      if(not meshFileName.startswith(".")):
        meshFilePath = os.path.join(meshDirectory, meshFileName)
        currentMeshNode = slicer.util.loadModel(meshFilePath)
        subjectID = os.path.splitext(meshFileName)[0]
        currentLMNode = self.getLandmarkFileByID(lmDirectory, subjectID)
        if currentLMNode:
          lmFilePath = os.path.join(lmDirectory, subjectID)
          targetPoints = vtk.vtkPoints()
          for i in range(currentLMNode.GetNumberOfControlPoints()):
            point = currentLMNode.GetNthControlPointPosition(i)
            targetPoints.InsertNextPoint(point)
          mirrorTransform = vtk.vtkTransform()
          mirrorTransform.SetMatrix(mirrorMatrix)
          mirrorTransformNode=slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTransformNode","Mirror")
          mirrorTransformNode.SetAndObserveTransformToParent(mirrorTransform)
          # apply transform to the current surface mesh and landmarks
          currentMeshNode.SetAndObserveTransformNodeID(mirrorTransformNode.GetID())
          currentLMNode.SetAndObserveTransformNodeID(mirrorTransformNode.GetID())
          slicer.vtkSlicerTransformLogic().hardenTransform(currentMeshNode)
          slicer.vtkSlicerTransformLogic().hardenTransform(currentLMNode)
          # apply rigid transformation
          sourcePoints = vtk.vtkPoints()
          mirrorLMNode =slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode",subjectID)
          for i in range(currentLMNode.GetNumberOfControlPoints()):
            point = currentLMNode.GetNthControlPointPosition(mirrorIndex[i])
            mirrorLMNode.AddControlPoint(point, str(i))
            sourcePoints.InsertNextPoint(point)
          rigidTransform = vtk.vtkLandmarkTransform()
          rigidTransform.SetSourceLandmarks( sourcePoints )
          rigidTransform.SetTargetLandmarks( targetPoints )
          rigidTransform.SetModeToRigidBody()
          rigidTransformNode=slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTransformNode","Rigid")
          rigidTransformNode.SetAndObserveTransformToParent(rigidTransform)
          # compute normals
          self.computeNormals(currentMeshNode)
          currentMeshNode.SetAndObserveTransformNodeID(rigidTransformNode.GetID())
          mirrorLMNode.SetAndObserveTransformNodeID(rigidTransformNode.GetID())
          slicer.vtkSlicerTransformLogic().hardenTransform(currentMeshNode)
          slicer.vtkSlicerTransformLogic().hardenTransform(mirrorLMNode)
          # optional semi-landmark alignment
          if semilandmarkOption:
            currentSLMNode = self.getLandmarkFileByID(slmDirectory, subjectID)
            mirrorSLMNode =slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode",subjectID)
            for i in range(currentSLMNode.GetNumberOfControlPoints()):
              point = currentSLMNode.GetNthControlPointPosition(mirrorSLMIndex[i])
              mirrorSLMNode.AddControlPoint(point, str(i))
            if currentSLMNode :
              currentSLMNode.SetAndObserveTransformNodeID(mirrorTransformNode.GetID())
              slicer.vtkSlicerTransformLogic().hardenTransform(currentSLMNode)
              currentSLMNode.SetAndObserveTransformNodeID(rigidTransformNode.GetID())
              slicer.vtkSlicerTransformLogic().hardenTransform(currentSLMNode)
              outputSLMName = subjectID + '_mirror.mrk.json'
              outputSLMPath = os.path.join(outputSLMDirectory, outputSLMName)
              slicer.util.saveNode(mirrorSLMNode, outputSLMPath)
              slicer.mrmlScene.RemoveNode(currentSLMNode)
              slicer.mrmlScene.RemoveNode(mirrorSLMNode)
          # save output files
          outputMeshName = subjectID + '_mirror.ply'
          outputMeshPath = os.path.join(mirrorMeshDirectory, outputMeshName)
          slicer.util.saveNode(currentMeshNode, outputMeshPath)
          outputLMName = subjectID + '_mirror.mrk.json'
          outputLMPath = os.path.join(mirrorLMDirectory, outputLMName)
          slicer.util.saveNode(mirrorLMNode, outputLMPath)
          # clean up
          slicer.mrmlScene.RemoveNode(currentLMNode)
          slicer.mrmlScene.RemoveNode(currentMeshNode)
          slicer.mrmlScene.RemoveNode(mirrorTransformNode)
          slicer.mrmlScene.RemoveNode(rigidTransformNode)
          slicer.mrmlScene.RemoveNode(mirrorLMNode)

  def runDCAlign(self, baseMeshPath, baseLMPath, meshDirectory, landmarkDirectory, outputDirectory, optionCPD, optionErrorOutput):
    if optionErrorOutput:
      self.errorCheckPath = os.path.join(outputDirectory, "errorChecking")
      if not os.path.exists(self.errorCheckPath):
        os.mkdir(self.errorCheckPath)
    baseNode = slicer.util.loadModel(baseMeshPath)
    baseMesh = baseNode.GetPolyData()
    baseLandmarks=self.fiducialNodeToPolyData(baseLMPath).GetPoints()
    #base, modelExt = os.path.splitext(baseMeshPath)
    modelExt=['ply','stl','vtp']
    self.modelNames, models = self.importMeshes(meshDirectory, modelExt)
    landmarkNames,landmarks = self.importLandmarks(landmarkDirectory)
    self.outputDirectory = outputDirectory
    if not(optionCPD):
      denseCorrespondenceGroup = self.denseCorrespondenceBaseMesh(landmarks, models, baseMesh, baseLandmarks)
    else:
      denseCorrespondenceGroup = self.denseCorrespondenceCPD(landmarks, models, baseMesh, baseLandmarks)
    self.addMagnitudeFeature(denseCorrespondenceGroup, self.modelNames, baseMesh)
    # save results to output directory
    outputModelName = 'decaResultModel.vtp'
    outputModelPath = os.path.join(outputDirectory, outputModelName)
    slicer.util.saveNode(baseNode, outputModelPath)

  def runDCAlignSymmetric(self, baseMeshPath, baseLMPath, meshDir, landmarkDir, mirrorMeshDir, mirrorLandmarkDir, outputDir, optionCPD, optionErrorOutput, optionPointOutput):
    if optionErrorOutput:
      self.errorCheckPath = os.path.join(outputDir, "errorChecking")
      if not os.path.exists(self.errorCheckPath):
        os.mkdir(self.errorCheckPath)
    baseNode = slicer.util.loadModel(baseMeshPath)
    baseMesh = baseNode.GetPolyData()
    baseLandmarks=self.fiducialNodeToPolyData(baseLMPath).GetPoints()
    #base, modelExt = os.path.splitext(baseMeshPath)
    modelExt=['ply','stl','vtp']
    self.modelNames, models = self.importMeshes(meshDir, modelExt)
    landmarkNames, landmarks = self.importLandmarks(landmarkDir)
    modelMirrorNames, mirrorModels = self.importMeshes(mirrorMeshDir, modelExt)
    mirrorLandmarkNames, mirrorLandmarks = self.importLandmarks(mirrorLandmarkDir)
    self.outputDirectory = outputDir
    if not(optionCPD):
      denseCorrespondenceGroup = self.denseCorrespondenceBaseMesh(landmarks, models, baseMesh, baseLandmarks)
      denseCorrespondenceGroupMirror = self.denseCorrespondenceBaseMesh(mirrorLandmarks, mirrorModels, baseMesh, baseLandmarks)
    else:
      denseCorrespondenceGroup = self.denseCorrespondenceCPD(landmarks, models, baseMesh, baseLandmarks)
      denseCorrespondenceGroupMirror = self.denseCorrespondenceCPD(mirrorLandmarks, mirrorModels, baseMesh, baseLandmarks)
    self.addMagnitudeFeatureSymmetry(denseCorrespondenceGroup, denseCorrespondenceGroupMirror, self.modelNames, baseMesh)
    # save results to output directory
    outputModelName = 'decaSymmetryResultModel.vtp'
    outputModelPath = os.path.join(outputDir, outputModelName)
    slicer.util.saveNode(baseNode, outputModelPath)

  def runMean(self, landmarkDirectory, meshDirectory):
    modelExt=['ply','stl','vtp','vtk']
    self.modelNames, models = self.importMeshes(meshDirectory, modelExt)
    landmarkNames, landmarks = self.importLandmarks(landmarkDirectory)
    [denseCorrespondenceGroup, closestToMeanIndex] = self.denseCorrespondence(landmarks, models)
    print("Sample closest to mean: ", closestToMeanIndex)
    # compute mean model
    averagePolyData = self.computeAverageModelFromGroup(denseCorrespondenceGroup, closestToMeanIndex)
    averageModelNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLModelNode', 'meanTemplate')
    averageModelNode.CreateDefaultDisplayNodes()
    averageModelNode.SetAndObservePolyData(averagePolyData)
     # compute mean landmarks
    averageLandmarkNode = self.computeAverageLM(landmarks)
    averageLandmarkNode.GetDisplayNode().SetPointLabelsVisibility(False)
    return averageModelNode, averageLandmarkNode

  def getLandmarkFileByID(self, directory, subjectID):
    fileList = os.listdir(directory)
    for fileName in fileList:
      fileNameBase = Path(fileName)
      while fileNameBase.suffix in {'.fcsv', '.mrk', '.json'}:
        fileNameBase = fileNameBase.with_suffix('')
      if subjectID == str(fileNameBase):
        #print(fileName + " found matching: " + subjectID)
        # if file with this subject id exists, load into scene
        filePath = os.path.join(directory, fileName)
        currentNode = slicer.util.loadMarkups(filePath)
        return currentNode

  def getModelFileByID(self, directory, subjectID):
    fileList = os.listdir(directory)
    for fileName in fileList:
      fileNameBase = Path(fileName).stem
      if str(subjectID) == str(fileNameBase):
        filePath = os.path.join(directory, fileName)
        currentNode = slicer.util.loadModel(filePath)
        return currentNode

  def runAlign(self, baseMeshNode, baseLMNode, meshDirectory, lmDirectory, ouputMeshDirectory, outputLMDirectory, removeScaleOption, slmDirectory=False, outputSLMDirector=False):
    semilandmarkOption = bool(slmDirectory and outputSLMDirectory)
    targetPoints = vtk.vtkPoints()
    point=[0,0,0]
    # Set up base points for transform
    for i in range(baseLMNode.GetNumberOfControlPoints()):
      point = baseLMNode.GetNthControlPointPosition(i)
      targetPoints.InsertNextPoint(point)
    # Transform each subject to base
    for meshFileName in os.listdir(meshDirectory):
      if(not meshFileName.startswith(".")):
        lmFileList = os.listdir(lmDirectory)
        meshFilePath = os.path.join(meshDirectory, meshFileName)
        subjectID = os.path.splitext(meshFileName)[0]
        #print("working on ", subjectID)
        currentLMNode = self.getLandmarkFileByID(lmDirectory, subjectID)
        if currentLMNode :
          try:
            currentMeshNode = slicer.util.loadModel(meshFilePath)
          except:
            slicer.mrmlScene.RemoveNode(currentLMNode)
            continue
          # set up transform between base lms and current lms
          sourcePoints = vtk.vtkPoints()
          for i in range(currentLMNode.GetNumberOfControlPoints()):
            point = currentLMNode.GetNthControlPointPosition(i)
            sourcePoints.InsertNextPoint(point)
            transform = vtk.vtkLandmarkTransform()
            transform.SetSourceLandmarks(sourcePoints)
            transform.SetTargetLandmarks(targetPoints)
          if not removeScaleOption:
            transform.SetModeToRigidBody()
          else:
            transform.SetModeToSimilarity()
          transformNode=slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTransformNode","Alignment")
          transformNode.SetAndObserveTransformToParent(transform)
          # apply transform to the current surface mesh and landmarks
          currentMeshNode.SetAndObserveTransformNodeID(transformNode.GetID())
          currentLMNode.SetAndObserveTransformNodeID(transformNode.GetID())
          slicer.vtkSlicerTransformLogic().hardenTransform(currentMeshNode)
          slicer.vtkSlicerTransformLogic().hardenTransform(currentLMNode)
          # save output files
          outputMeshName = subjectID + '_align.ply'
          outputMeshPath = os.path.join(ouputMeshDirectory, outputMeshName)
          slicer.util.saveNode(currentMeshNode, outputMeshPath)
          outputLMName = subjectID + '_align.mrk.json'
          outputLMPath = os.path.join(outputLMDirectory, outputLMName)
          slicer.util.saveNode(currentLMNode, outputLMPath)
          # optional semi-landmark alignment
          if semilandmarkOption :
            currentSLMNode = self.getLandmarkFileByID(slmDirectory, subjectID)
            if currentSLMNode :
              currentSLMNode.SetAndObserveTransformNodeID(transformNode.GetID())
              slicer.vtkSlicerTransformLogic().hardenTransform(currentSLMNode)
              outputSLMName = subjectID + '_align.mrk.json'
              outputSLMPath = os.path.join(outputSLMDirectory, outputSLMName)
              slicer.util.saveNode(currentSLMNode, outputSLMPath)
              slicer.mrmlScene.RemoveNode(currentSLMNode)
          # clean up
          try:
            slicer.mrmlScene.RemoveNode(currentLMNode)
            slicer.mrmlScene.RemoveNode(currentMeshNode)
            slicer.mrmlScene.RemoveNode(transformNode)
            #slicer.mrmlScene.RemoveNode(baseMeshNode)
            #slicer.mrmlScene.RemoveNode(baseLMNode)
          except:
            print(f"could not find nodes to remove for {subjectID}")

  def distanceMatrix(self, a):
    """
    Computes the euclidean distance matrix for n points in a 3D space
    Returns a nXn matrix
     """
    id,jd=a.shape
    fnx = lambda q : q - np.reshape(q, (id, 1))
    dx=fnx(a[:,0])
    dy=fnx(a[:,1])
    dz=fnx(a[:,2])
    return (dx**2.0+dy**2.0+dz**2.0)**0.5

  def numpyToFiducialNode(self, numpyArray, nodeName):
    fiducialNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLMarkupsFiducialNode',nodeName)
    for index in range(len(numpyArray)):
      fiducialNode.AddControlPoint(numpyArray[index], str(index))
    return fiducialNode

  def computeAverageLM(self, fiducialGroup):
    sampleNumber = fiducialGroup.GetNumberOfBlocks()
    pointNumber = fiducialGroup.GetBlock(0).GetNumberOfPoints()
    groupArray_np = np.empty((pointNumber,3,sampleNumber))
    for i in range(sampleNumber):
      pointData = fiducialGroup.GetBlock(i).GetPoints().GetData()
      pointData_np = vtk_np.vtk_to_numpy(pointData)
      groupArray_np[:,:,i] = pointData_np
    #Calculate mean point positions of aligned group
    averagePoints_np = np.mean(groupArray_np, axis=2)
    averageLMNode = self.numpyToFiducialNode(averagePoints_np, "Atlas")
    return averageLMNode

  def fiducialNodeToPolyData(self, nodeLocation, loadOption=True):
    point = [0,0,0]
    polydataPoints = vtk.vtkPolyData()
    points = vtk.vtkPoints()
    if not loadOption:
      fiducialNode = nodeLocation
    else:
      [success,fiducialNode] = slicer.util.loadMarkupsFiducialList(nodeLocation)
      if not success:
        print("Could not load landmarks: ", nodeLocation)
        return
    for i in range(fiducialNode.GetNumberOfControlPoints()):
      point = fiducialNode.GetNthControlPointPosition(i)
      points.InsertNextPoint(point)
    polydataPoints.SetPoints(points)
    slicer.mrmlScene.RemoveNode(fiducialNode)
    return polydataPoints

  def importLandmarks(self, topDir):
    fiducialGroup = vtk.vtkMultiBlockDataGroupFilter()
    fileNameList = []
    for file in sorted(os.listdir(topDir)):
      if file.endswith(".fcsv") or file.endswith(".json"):
        fileNameList.append(file)
        inputFilePath = os.path.join(topDir, file)
        # may want to replace with vtk reader
        polydataPoints = self.fiducialNodeToPolyData(inputFilePath)
        fiducialGroup.AddInputData(polydataPoints)
    fiducialGroup.Update()
    return fileNameList, fiducialGroup.GetOutput()

  def importMeshes(self, topDir, extensions):
      modelGroup = vtk.vtkMultiBlockDataGroupFilter()
      fileNameList = []
      for file in sorted(os.listdir(topDir)):
        if file.endswith(tuple(extensions)):
          print("reading: ", file)
          base, ext = os.path.splitext(file)
          fileNameList.append(base)
          inputFilePath = os.path.join(topDir, file)
          # may want to replace with vtk reader
          modelNode = slicer.util.loadModel(inputFilePath)
          modelGroup.AddInputData(modelNode.GetPolyData())
          slicer.mrmlScene.RemoveNode(modelNode)
      modelGroup.Update()
      return fileNameList, modelGroup.GetOutput()

  def procrustesImposition(self, originalLandmarks, sizeOption):
    procrustesFilter = vtk.vtkProcrustesAlignmentFilter()
    if(sizeOption):
      procrustesFilter.GetLandmarkTransform().SetModeToRigidBody()

    procrustesFilter.SetInputData(originalLandmarks)
    procrustesFilter.Update()
    meanShape = procrustesFilter.GetMeanPoints()
    return [meanShape, procrustesFilter.GetOutput()]

  def getClosestToMeanIndex(self, meanShape, alignedPoints):
    import operator
    sampleNumber = alignedPoints.GetNumberOfBlocks()
    procrustesDistances = []
    for i in range(sampleNumber):
      alignedShape = alignedPoints.GetBlock(i)
      meanPoint = [0,0,0]
      alignedPoint = [0,0,0]
      distance = 0
      for j in range(meanShape.GetNumberOfPoints()):
        meanShape.GetPoint(j,meanPoint)
        alignedShape.GetPoint(j,alignedPoint)
        distance += np.sqrt(vtk.vtkMath.Distance2BetweenPoints(meanPoint,alignedPoint))
      procrustesDistances.append(distance)
    try:
      min_index, min_value = min(enumerate(procrustesDistances), key=operator.itemgetter(1))
      return min_index
    except:
      return 0

  def getClosestToMeanPath(self, landmarkDirectory):
    lmNames, landmarks = self.importLandmarks(landmarkDirectory)
    meanShape, alignedLandmarks = self.procrustesImposition(landmarks, False)
    closestToMeanIndex = self.getClosestToMeanIndex(meanShape, alignedLandmarks)
    return lmNames[closestToMeanIndex]

  def denseCorrespondence(self, originalLandmarks, originalMeshes, writeErrorOption=False):
    meanShape, alignedPoints = self.procrustesImposition(originalLandmarks, False)
    sampleNumber = alignedPoints.GetNumberOfBlocks()
    denseCorrespondenceGroup = vtk.vtkMultiBlockDataGroupFilter()
    # get base mesh as the closest to the mean shape
    baseIndex = self.getClosestToMeanIndex(meanShape, alignedPoints)
    baseMesh = originalMeshes.GetBlock(baseIndex)
    baseLandmarks = originalLandmarks.GetBlock(baseIndex).GetPoints()
    for i in range(sampleNumber):
      correspondingMesh = self.denseSurfaceCorrespondencePair(originalMeshes.GetBlock(i),
      originalLandmarks.GetBlock(i).GetPoints(), alignedPoints.GetBlock(i).GetPoints(),
      baseMesh, baseLandmarks, meanShape, i)
      denseCorrespondenceGroup.AddInputData(correspondingMesh)

    denseCorrespondenceGroup.Update()
    return denseCorrespondenceGroup.GetOutput(), baseIndex

  def denseCorrespondenceCPD(self, originalLandmarks, originalMeshes, baseMesh, baseLandmarks, writeErrorOption=False):
    meanShape, alignedPoints = self.procrustesImposition(originalLandmarks, False)
    sampleNumber = alignedPoints.GetNumberOfBlocks()
    denseCorrespondenceGroup = vtk.vtkMultiBlockDataGroupFilter()

    # assign parameters for CPD
    parameters = {
      "SpacingTolerance": .04,
      "CPDIterations": 100,
      "CPDTolerence": 0.001,
      "alpha": 2,
      "beta": 2,
     }

    for i in range(sampleNumber):
      correspondingPoints = self.runCPDRegistration(originalMeshes.GetBlock(i), baseMesh, parameters)
      # convert to vtkPoints
      correspondingMesh = self.convertPointsToVTK(correspondingPoints)
      correspondingMesh.SetPolys(baseMesh.GetPolys())
      # convert to polydata
      denseCorrespondenceGroup.AddInputData(correspondingMesh)
      # write ouput
      if writeErrorOption:
        plyWriterSubject = vtk.vtkPLYWriter()
        plyWriterSubject.SetFileName("/Users/sararolfe/Dropbox/SlicerWorkspace/SMwSML/Data/UBC/DECAOutCPD/" + str(i) + ".ply")
        plyWriterSubject.SetInputData(correspondingMesh)
        plyWriterSubject.Write()

        plyWriterBase = vtk.vtkPLYWriter()
        plyWriterBase.SetFileName("/Users/sararolfe/Dropbox/SlicerWorkspace/SMwSML/Data/UBC/DECAOutCPD/base.ply")
        plyWriterBase.SetInputData(baseMesh)
        plyWriterBase.Write()

    denseCorrespondenceGroup.Update()
    return denseCorrespondenceGroup.GetOutput()

  def runCPDRegistration(self,sourceData, targetData, parameters):
    from open3d import geometry
    from open3d import utility

    # Downsample meshes
    sourceFilter=vtk.vtkCleanPolyData()
    sourceFilter.SetTolerance(parameters["SpacingTolerance"])
    sourceFilter.SetInputData(sourceData)
    sourceFilter.Update()
    #sourceArray = sourceFilter.GetOutput().GetPoints().GetData()
    sourceArray = sourceData.GetPoints().GetData()

    targetFilter=vtk.vtkCleanPolyData()
    targetFilter.SetTolerance(parameters["SpacingTolerance"])
    targetFilter.SetInputData(targetData)
    targetFilter.Update()
    #targetArray = targetFilter.GetOutput().GetPoints().GetData()
    targetArray = targetData.GetPoints().GetData()

    # Convert to pointcloud for scaling
    sourceArray_np = vtk_np.vtk_to_numpy(sourceArray)
    targetArray_np = vtk_np.vtk_to_numpy(targetArray)
    sourceCloud = geometry.PointCloud()
    sourceCloud.points = utility.Vector3dVector(sourceArray_np)
    targetCloud = geometry.PointCloud()
    targetCloud.points = utility.Vector3dVector(targetArray_np)
    cloudSize = np.max(targetCloud.get_max_bound() - targetCloud.get_min_bound())
    targetCloud.scale(25 / cloudSize, center = False)
    sourceCloud.scale(25 / cloudSize, center = False)
    # Convert back to numpy for cpd
    sourceArray = np.asarray(sourceCloud.points,dtype=np.float32)
    targetArray = np.asarray(targetCloud.points,dtype=np.float32)
    registrationOutput = self.cpd_registration(targetArray, sourceArray, parameters["CPDIterations"], parameters["CPDTolerence"], parameters["alpha"], parameters["beta"])
    deformed_array, _ = registrationOutput.register()
    outputCloud = geometry.PointCloud()
    outputCloud.points = utility.Vector3dVector(deformed_array)
    outputCloud.scale(cloudSize/25, center = False)
    return np.asarray(outputCloud.points)

  def cpd_registration(self, targetArray, sourceArray, CPDIterations, CPDTolerence, alpha_parameter, beta_parameter):
    from pycpd import DeformableRegistration
    output = DeformableRegistration(**{'X': targetArray, 'Y': sourceArray,'max_iterations': CPDIterations, 'tolerance': CPDTolerence}, alpha = alpha_parameter, beta  = beta_parameter)
    return output

  def denseCorrespondenceBaseMesh(self, originalLandmarks, originalMeshes, baseMesh, baseLandmarks):
    meanShape, alignedPoints = self.procrustesImposition(originalLandmarks, False)
    sampleNumber = alignedPoints.GetNumberOfBlocks()
    print("procrustes aligned samples: ", sampleNumber)
    denseCorrespondenceGroup = vtk.vtkMultiBlockDataGroupFilter()
    for i in range(sampleNumber):
      correspondingMesh = self.denseSurfaceCorrespondencePair(originalMeshes.GetBlock(i),
      originalLandmarks.GetBlock(i).GetPoints(), alignedPoints.GetBlock(i).GetPoints(),
      baseMesh, baseLandmarks, meanShape, i)
      denseCorrespondenceGroup.AddInputData(correspondingMesh)

    denseCorrespondenceGroup.Update()
    return denseCorrespondenceGroup.GetOutput()

  def denseSurfaceCorrespondencePair(self, originalMesh, originalLandmarks, alignedLandmarks, baseMesh, baseLandmarks, meanShape, iteration):
    # TPS warp target and base mesh to meanshape
    meanTransform = vtk.vtkThinPlateSplineTransform()
    meanTransform.SetSourceLandmarks( originalLandmarks )
    meanTransform.SetTargetLandmarks( meanShape )
    meanTransform.SetBasisToR() # for 3D transform

    meanTransformFilter = vtk.vtkTransformPolyDataFilter()
    meanTransformFilter.SetInputData(originalMesh)
    meanTransformFilter.SetTransform(meanTransform)
    meanTransformFilter.Update()
    meanWarpedMesh = meanTransformFilter.GetOutput()

    meanTransformBase = vtk.vtkThinPlateSplineTransform()
    meanTransformBase.SetSourceLandmarks( baseLandmarks )
    meanTransformBase.SetTargetLandmarks( meanShape )
    meanTransformBase.SetBasisToR() # for 3D transform

    meanTransformBaseFilter = vtk.vtkTransformPolyDataFilter()
    meanTransformBaseFilter.SetInputData(baseMesh)
    meanTransformBaseFilter.SetTransform(meanTransformBase)
    meanTransformBaseFilter.Update()
    meanWarpedBase = meanTransformBaseFilter.GetOutput()

    # write ouput
    if hasattr(self,"errorCheckPath"):
      plyWriterSubject = vtk.vtkPLYWriter()
      print(self.modelNames)
      print(iteration)
      plyName = "subject_" + self.modelNames[iteration] + ".ply"
      plyPath = os.path.join(self.errorCheckPath, plyName)
      plyWriterSubject.SetFileName(plyPath)
      plyWriterSubject.SetInputData(meanWarpedMesh)
      plyWriterSubject.Write()

      plyWriterBase = vtk.vtkPLYWriter()
      plyName = "base.ply"
      plyPath = os.path.join(self.errorCheckPath, plyName)
      plyWriterBase.SetFileName(plyPath)
      plyWriterBase.SetInputData(meanWarpedBase)
      plyWriterBase.Write()

    # Dense correspondence
    cellLocator = vtk.vtkCellLocator()
    cellLocator.SetDataSet(meanWarpedMesh)
    cellLocator.BuildLocator()

    point = [0,0,0]
    correspondingPoint = [0,0,0]
    correspondingPoints = vtk.vtkPoints()
    cellId = vtk.reference(0)
    subId = vtk.reference(0)
    distance = vtk.reference(0.0)
    for i in range(meanWarpedBase.GetNumberOfPoints()):
      meanWarpedBase.GetPoint(i,point)
      cellLocator.FindClosestPoint(point,correspondingPoint,cellId, subId, distance)
      correspondingPoints.InsertPoint(i,correspondingPoint)

    #Copy points into mesh with base connectivity
    correspondingMesh = vtk.vtkPolyData()
    correspondingMesh.SetPoints(correspondingPoints)
    correspondingMesh.SetPolys(meanWarpedBase.GetPolys())

    # Apply inverse warping
    inverseTransform = vtk.vtkThinPlateSplineTransform()
    inverseTransform.SetSourceLandmarks( meanShape )
    inverseTransform.SetTargetLandmarks( originalLandmarks )
    inverseTransform.SetBasisToR() # for 3D transform

    inverseTransformFilter = vtk.vtkTransformPolyDataFilter()
    inverseTransformFilter.SetInputData(correspondingMesh)
    inverseTransformFilter.SetTransform(inverseTransform)
    inverseTransformFilter.Update()

    return inverseTransformFilter.GetOutput()

  def convertPointsToVTK(self, points):
    array_vtk = vtk_np.numpy_to_vtk(points, deep=True, array_type=vtk.VTK_FLOAT)
    points_vtk = vtk.vtkPoints()
    points_vtk.SetData(array_vtk)
    polydata_vtk = vtk.vtkPolyData()
    polydata_vtk.SetPoints(points_vtk)
    return polydata_vtk

  def computeAverageModelFromGroup(self, denseCorrespondenceGroup, baseIndex):
    sampleNumber = denseCorrespondenceGroup.GetNumberOfBlocks()
    pointNumber = denseCorrespondenceGroup.GetBlock(0).GetNumberOfPoints()
    groupArray_np = np.empty((pointNumber,3,sampleNumber))
    # get base mesh as closest to the meanshape
    baseMesh = denseCorrespondenceGroup.GetBlock(baseIndex)
     # get points as array
    for i in range(sampleNumber):
      alignedMesh = denseCorrespondenceGroup.GetBlock(i)
      alignedMesh_np = vtk_np.vtk_to_numpy(alignedMesh.GetPoints().GetData())
      groupArray_np[:,:,i] = alignedMesh_np
    #Calculate mean point positions of aligned group
    averagePoints_np = np.mean(groupArray_np, axis=2)
    averagePointsPolydata = self.convertPointsToVTK(averagePoints_np)
    #Copy points into mesh with base connectivity
    averageModel = vtk.vtkPolyData()
    averageModel.SetPoints(averagePointsPolydata.GetPoints())
    averageModel.SetPolys(baseMesh.GetPolys())
    return averageModel

  def addMagnitudeFeature(self, denseCorrespondenceGroup, modelNameArray, model):
    sampleNumber = denseCorrespondenceGroup.GetNumberOfBlocks()
    pointNumber = denseCorrespondenceGroup.GetBlock(0).GetNumberOfPoints()
    statsArray = np.zeros((pointNumber, sampleNumber))
    magnitudeMean = vtk.vtkDoubleArray()
    magnitudeMean.SetNumberOfComponents(1)
    magnitudeMean.SetName("Magnitude Mean")
    magnitudeSD = vtk.vtkDoubleArray()
    magnitudeSD.SetNumberOfComponents(1)
    magnitudeSD.SetName("Magnitude SD")

     # get distance arrays
    for i in range(sampleNumber):
      alignedMesh = denseCorrespondenceGroup.GetBlock(i)
      magnitudes = vtk.vtkDoubleArray()
      magnitudes.SetNumberOfComponents(1)
      magnitudes.SetName(modelNameArray[i])
      for j in range(pointNumber):
        modelPoint = model.GetPoint(j)
        targetPoint = alignedMesh.GetPoint(j)
        distance = np.sqrt(vtk.vtkMath.Distance2BetweenPoints(modelPoint,targetPoint))
        magnitudes.InsertNextValue(distance)
        statsArray[j,i]=distance

      model.GetPointData().AddArray(magnitudes)

    for i in range(pointNumber):
      pointMean = statsArray[i,:].mean()
      magnitudeMean.InsertNextValue(pointMean)
      pointSD = statsArray[i,:].std()
      magnitudeSD.InsertNextValue(pointSD)

    model.GetPointData().AddArray(magnitudeMean)
    model.GetPointData().AddArray(magnitudeSD)

  def addMagnitudeFeatureSymmetry(self, denseCorrespondenceGroup, denseCorrespondenceGroupMirror, modelNameArray, model):
    sampleNumber = denseCorrespondenceGroup.GetNumberOfBlocks()
    pointNumber = denseCorrespondenceGroup.GetBlock(0).GetNumberOfPoints()
    statsArray = np.zeros((pointNumber, sampleNumber))
    magnitudeMean = vtk.vtkDoubleArray()
    magnitudeMean.SetNumberOfComponents(1)
    magnitudeMean.SetName("Magnitude Mean")
    magnitudeSD = vtk.vtkDoubleArray()
    magnitudeSD.SetNumberOfComponents(1)
    magnitudeSD.SetName("Magnitude SD")

     # get distance arrays
    for i in range(sampleNumber):
      alignedMesh = denseCorrespondenceGroup.GetBlock(i)
      mirrorMesh = denseCorrespondenceGroupMirror.GetBlock(i)
      magnitudes = vtk.vtkDoubleArray()
      magnitudes.SetNumberOfComponents(1)
      magnitudes.SetName(modelNameArray[i])
      for j in range(pointNumber):
        modelPoint = model.GetPoint(j)
        targetPoint1 = alignedMesh.GetPoint(j)
        targetPoint2 = mirrorMesh.GetPoint(j)
        distance = np.sqrt(vtk.vtkMath.Distance2BetweenPoints(targetPoint1,targetPoint2))
        magnitudes.InsertNextValue(distance)
        statsArray[j,i]=distance

      model.GetPointData().AddArray(magnitudes)

    for i in range(pointNumber):
      pointMean = statsArray[i,:].mean()
      magnitudeMean.InsertNextValue(pointMean)
      pointSD = statsArray[i,:].std()
      magnitudeSD.InsertNextValue(pointSD)

    model.GetPointData().AddArray(magnitudeMean)
    model.GetPointData().AddArray(magnitudeSD)
