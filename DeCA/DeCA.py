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
    self.parent.categories = ["SlicerMorph.DeCA Toolbox"]
    self.parent.dependencies = []
    self.parent.contributors = ["Sara Rolfe (SCRI)"] # replace with "Firstname Lastname (Organization)"
    self.parent.helpText = """
      This module provides several flexible workflows for finding and analyzing dense correspondence points between models.
      """
    self.parent.helpText += self.getDefaultModuleDocumentationLink()
    self.parent.acknowledgementText = """This extension was developed by funding from National Institutes of Health (OD032627 and HD104435) to A. Murat Maga (SCRI)
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

    # Shared re-entrancy guard: the long-running handlers pump the Qt event loop
    # (via progressCallback) so their progress bars update, which also lets a click
    # on a different tab's trigger button be dispatched mid-run. All three run
    # handlers share self.folderNames / self.atlasModel state, so a re-entrant call
    # would corrupt an in-flight run. Each handler no-ops while this flag is set.
    self._busy = False

    # Set up tabs to split workflow
    tabsWidget = qt.QTabWidget()
    DeCATab = qt.QWidget()
    DeCATabLayout = qt.QFormLayout(DeCATab)
    DeCALTab = qt.QWidget()
    DeCALTabLayout = qt.QFormLayout(DeCALTab)
    visualizeTab = qt.QWidget()
    visualizeTabLayout = qt.QFormLayout(visualizeTab)

    tabsWidget.addTab(DeCATab, "DeCA")
    tabsWidget.addTab(DeCALTab, "DeCAL")
    tabsWidget.addTab(visualizeTab, "Visualize Results")

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
    self.atlasCollapsibleButtonDC = ctk.ctkCollapsibleButton()
    self.atlasCollapsibleButtonDC.text = "Atlas Options"
    self.atlasCollapsibleButtonDC.collapsed = True
    self.atlasCollapsibleButtonDC.enabled = False
    DeCAWidgetLayout.addRow(self.atlasCollapsibleButtonDC)
    atlasOptionLayout = qt.QFormLayout(self.atlasCollapsibleButtonDC)

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

    ##
    ## MODIFIED SECTION: Replaced single landmarkIndexText with three new fields
    ##
    self.midlineLandmarksText = qt.QLineEdit()
    self.midlineLandmarksText.setToolTip("Enter 1-based midline indices, separated by commas. Example: 1,2,3")
    symmetryOptionLayout.addRow("Midline landmarks:", self.midlineLandmarksText)

    self.leftLandmarksText = qt.QLineEdit()
    self.leftLandmarksText.setToolTip("Enter 1-based left-side indices, separated by commas. Example: 4,5,6")
    symmetryOptionLayout.addRow("Left landmarks:", self.leftLandmarksText)
    
    self.rightLandmarksText = qt.QLineEdit()
    self.rightLandmarksText.setToolTip("Enter 1-based right-side indices, in corresponding order to the left. Example: 7,8,9")
    symmetryOptionLayout.addRow("Right landmarks:", self.rightLandmarksText)
    ##
    ## END OF MODIFIED SECTION
    ##

    #
    # Select model directory
    #
    self.meshDirectoryDC=ctk.ctkPathLineEdit()
    self.meshDirectoryDC.filters = ctk.ctkPathLineEdit.Dirs
    self.meshDirectoryDC.setToolTip("Select directory containing models")
    DeCAWidgetLayout.addRow("Model directory: ", self.meshDirectoryDC)

    #
    # Select landmark directory
    #
    self.landmarkDirectoryDC=ctk.ctkPathLineEdit()
    self.landmarkDirectoryDC.filters = ctk.ctkPathLineEdit.Dirs
    self.landmarkDirectoryDC.setToolTip("Select directory containing landmarks")
    DeCAWidgetLayout.addRow("Landmark directory: ", self.landmarkDirectoryDC)

    #
    # Select DeCA output directory
    #
    self.outputDirectoryDC=ctk.ctkPathLineEdit()
    self.outputDirectoryDC.filters = ctk.ctkPathLineEdit.Dirs
    self.outputDirectoryDC.setToolTip("Select directory for DeCA output")
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
    self.applyButtonDC = qt.QPushButton("Run DeCA")
    self.applyButtonDC.toolTip = "Run non-rigid alignment"
    self.applyButtonDC.enabled = False
    DeCAWidgetLayout.addRow(self.applyButtonDC)

    #
    # Progress bar
    #
    self.progressBarDC = qt.QProgressBar()
    self.progressBarDC.minimum = 0
    self.progressBarDC.maximum = 1
    self.progressBarDC.value = 0
    self.progressBarDC.setFormat("Idle")
    DeCAWidgetLayout.addRow(self.progressBarDC)

    #
    # Log Information
    #
    self.logInfoDC = qt.QPlainTextEdit()
    self.logInfoDC.setPlaceholderText("DeCA log information")
    self.logInfoDC.setReadOnly(True)
    DeCAWidgetLayout.addRow(self.logInfoDC)

    # Connections
    self.analysisTypeShape.connect('toggled(bool)', self.onToggleAnalysis)
    self.analysisTypeSymmetry.connect('toggled(bool)', self.onToggleAnalysis)
    self.calculateAtlasOptionDC.connect('toggled(bool)', self.onToggleAtlasDC)
    self.loadAtlasOptionDC.connect('toggled(bool)', self.onToggleAtlasDC)
    self.DCBaseModelSelector.connect('validInputChanged(bool)', self.onParameterSelectDC)
    self.DCBaseLMSelector.connect('validInputChanged(bool)', self.onParameterSelectDC)
    self.meshDirectoryDC.connect('validInputChanged(bool)', self.onParameterSelectDC)
    self.landmarkDirectoryDC.connect('validInputChanged(bool)', self.onParameterSelectDC)
    self.outputDirectoryDC.connect('validInputChanged(bool)', self.onParameterSelectDC)
    self.applyButtonDC.connect('clicked(bool)', self.onDCApplyButton)

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
    self.meshDirectoryDCL.setToolTip("Select directory containing models")
    DeCALWidgetLayout.addRow("Model directory: ", self.meshDirectoryDCL)

    #
    # Select landmarks directory
    #
    self.landmarkDirectoryDCL=ctk.ctkPathLineEdit()
    self.landmarkDirectoryDCL.filters = ctk.ctkPathLineEdit.Dirs
    self.landmarkDirectoryDCL.setToolTip("Select directory containing landmarks")
    DeCALWidgetLayout.addRow("Landmark directory: ", self.landmarkDirectoryDCL)

    #
    # Select DeCA output directory
    #
    self.OutputDirectoryDCL=ctk.ctkPathLineEdit()
    self.OutputDirectoryDCL.filters = ctk.ctkPathLineEdit.Dirs
    self.OutputDirectoryDCL.setToolTip("Select directory for DeCAL output")
    DeCALWidgetLayout.addRow("DeCAL output directory: ", self.OutputDirectoryDCL)

    #
    # Generate Atlas Button
    #
    self.getAtlasButton = qt.QPushButton("Create\\Load atlas")
    self.getAtlasButton.toolTip = "Generate a new atlas model and landmark set from data"
    self.getAtlasButton.enabled = False
    DeCALWidgetLayout.addRow(self.getAtlasButton)

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
    # Get Subsample Rate Button
    #
    self.getPointNumberButton = qt.QPushButton("Run subsampling")
    self.getPointNumberButton.toolTip = "Get the number of output points that will be generated"
    self.getPointNumberButton.enabled = False
    DeCALWidgetLayout.addRow(self.getPointNumberButton)

    #
    # Merge generated semi-landmarks with fixed landmarks option
    #
    self.mergeLandmarksCheckBoxDCL = qt.QCheckBox()
    self.mergeLandmarksCheckBoxDCL.checked = True
    self.mergeLandmarksCheckBoxDCL.setToolTip("If checked, the generated semi-landmarks are merged with the fixed landmarks used to establish correspondence (using the SlicerMorph MergeMarkups module) and saved to a 'mergedLMs' folder.")
    DeCALWidgetLayout.addRow("Generate merged point lists: ", self.mergeLandmarksCheckBoxDCL)

    #
    # Also output landmarks in the original (un-aligned) model coordinate frame
    #
    self.originalFrameCheckBoxDCL = qt.QCheckBox()
    self.originalFrameCheckBoxDCL.checked = True
    self.originalFrameCheckBoxDCL.setToolTip("If checked, the generated semi-landmarks (and merged point lists) are also written in the original, un-aligned coordinate frame of each input model, by inverting the alignment that mapped the subject onto the atlas. Saved to 'DeCALOutput_originalFrame' (and 'mergedLMs_originalFrame').")
    DeCALWidgetLayout.addRow("Output landmarks in original frame: ", self.originalFrameCheckBoxDCL)

    #
    # Fast (approximate) correspondence option -- DeCAL only, off by default
    #
    self.fastCorrespondenceCheckBoxDCL = qt.QCheckBox()
    self.fastCorrespondenceCheckBoxDCL.checked = False
    self.fastCorrespondenceCheckBoxDCL.setToolTip("Off (default) uses the canonical exact closest-point-on-surface correspondence, matching the published DeCA method. If checked, DeCAL computes correspondences with a much faster approximate method that snaps each point to the nearest mesh vertex instead of the exact closest point on the surface; on dense meshes the difference is typically a few hundredths of a millimeter. The atlas/template is always built with the exact method, and this option does not affect the DeCA tab.")
    DeCALWidgetLayout.addRow("Compute fast correspondences: ", self.fastCorrespondenceCheckBoxDCL)

    #
    # Apply Button
    #
    self.DCLApplyButton = qt.QPushButton("Run DeCAL")
    self.DCLApplyButton.toolTip = "Generate a set of corresponding landmarks"
    self.DCLApplyButton.enabled = False
    DeCALWidgetLayout.addRow(self.DCLApplyButton)

    #
    # Progress bar
    #
    self.progressBarDCL = qt.QProgressBar()
    self.progressBarDCL.minimum = 0
    self.progressBarDCL.maximum = 1
    self.progressBarDCL.value = 0
    self.progressBarDCL.setFormat("Idle")
    DeCALWidgetLayout.addRow(self.progressBarDCL)

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
    self.pointSelection.nodeTypes = (("vtkMRMLMarkupsFiducialNode"), "")
    self.pointSelection.setToolTip("Atlas landmarks with subset points selected")
    self.pointSelection.selectNodeUponCreation = False
    self.pointSelection.noneEnabled = True
    self.pointSelection.addEnabled = False
    self.pointSelection.removeEnabled = False
    self.pointSelection.showHidden = False
    self.pointSelection.setMRMLScene(slicer.mrmlScene)
    DeCALSubsetLayout.addRow("Atlas landmarks: ", self.pointSelection)

    #
    # Select DeCAL output directory
    #
    self.DCLLandmarkDirectory=ctk.ctkPathLineEdit()
    self.DCLLandmarkDirectory.filters = ctk.ctkPathLineEdit.Dirs
    self.DCLLandmarkDirectory.setToolTip("Select directory for DeCAL sampled landmarks to subset")
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
    self.DCLBaseModelSelector.connect('validInputChanged(bool)', self.onParameterSelectDCL)
    self.DCLBaseLMSelector.connect('validInputChanged(bool)', self.onParameterSelectDCL)
    self.meshDirectoryDCL.connect('validInputChanged(bool)', self.onParameterSelectDCL)
    self.landmarkDirectoryDCL.connect('validInputChanged(bool)', self.onParameterSelectDCL)
    self.OutputDirectoryDCL.connect('validInputChanged(bool)', self.onParameterSelectDCL)
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
    self.meshSelect.nodeTypes = (("vtkMRMLModelNode"), "")
    self.meshSelect.setToolTip("Select model node with result arrays")
    self.meshSelect.selectNodeUponCreation = False
    self.meshSelect.noneEnabled = True
    self.meshSelect.addEnabled = False
    self.meshSelect.removeEnabled = False
    self.meshSelect.showHidden = False
    self.meshSelect.setMRMLScene(slicer.mrmlScene)
    visualizeWidgetLayout.addRow("Result Model: ", self.meshSelect)

    #
    # Select Subject ID
    #
    self.subjectIDBox=qt.QComboBox()
    self.subjectIDBox.enabled = False
    visualizeWidgetLayout.addRow("Subject ID: ", self.subjectIDBox)

    # Connections
    self.meshSelect.connect("currentNodeChanged(vtkMRMLNode*)", self.onVisualizeMeshSelect)
    self.subjectIDBox.connect("currentIndexChanged(int)", self.onSubjectIDSelect)

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
        # per-subject alignment transforms are saved here so DeCAL output can later
        # be mapped back into each subject's original (un-aligned) coordinate frame
        alignmentTransformFolder = os.path.join(outputFolderDC, "alignmentTransforms")
        os.makedirs(alignmentTransformFolder)
        fileNameDictionary['alignmentTransforms'] = str(alignmentTransformFolder)
    except:
      logging.debug('Result directory failed: Could not create output folder')
    return fileNameDictionary

  def makeProgressCallback(self, progressBar):
    # Returns a progressCallback(current, total, message) that updates the given
    # QProgressBar and pumps the Qt event loop so the UI stays responsive during
    # long single-threaded per-subject loops (atlas building, dense correspondence)
    # instead of appearing to stall.
    def progressCallback(current, total, message):
      total = max(total, 1)
      progressBar.minimum = 0
      progressBar.maximum = total
      progressBar.value = current
      progressBar.setFormat(f"{message}: {current}/{total}")
      slicer.app.processEvents()
    return progressCallback

  def resetProgressBar(self, progressBar, message="Idle"):
    progressBar.minimum = 0
    progressBar.maximum = 1
    progressBar.value = 0
    progressBar.setFormat(message)

  def onToggleAnalysis(self):
    if self.analysisTypeSymmetry.checked == True:
      self.symmetryCollapsibleButton.collapsed = False
      self.symmetryCollapsibleButton.enabled = True
    else:
      self.symmetryCollapsibleButton.collapsed = True
      self.symmetryCollapsibleButton.enabled = False

  def onToggleAtlasDC(self):
    if self.calculateAtlasOptionDC.checked == True:
      self.atlasCollapsibleButtonDC.collapsed = True
      self.atlasCollapsibleButtonDC.enabled = False
    else:
      self.atlasCollapsibleButtonDC.collapsed = False
      self.atlasCollapsibleButtonDC.enabled = True
    self.onParameterSelectDC()

  def onToggleAtlasDCL(self):
    if self.calculateAtlasOptionDCL.checked == True:
      self.atlasCollapsibleButtonDCL.collapsed = True
      self.atlasCollapsibleButtonDCL.enabled = False
    else:
      self.atlasCollapsibleButtonDCL.collapsed = False
      self.atlasCollapsibleButtonDCL.enabled = True
    self.onParameterSelectDCL()

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

  def onParameterSelectDC(self):
    atlasPathSelected = bool(self.DCBaseModelSelector.currentPath and self.DCBaseLMSelector.currentPath) or self.calculateAtlasOptionDC.checked
    inputPathsSelected = bool(self.meshDirectoryDC.currentPath and self.landmarkDirectoryDC.currentPath and self.outputDirectoryDC.currentPath)
    self.applyButtonDC.enabled = bool(atlasPathSelected and inputPathsSelected)

  def onParameterSelectDCL(self):
    atlasPathSelected = bool(self.DCLBaseModelSelector.currentPath and self.DCLBaseLMSelector.currentPath) or self.calculateAtlasOptionDCL.checked
    inputPathsSelected = bool(self.meshDirectoryDCL.currentPath and self.landmarkDirectoryDCL.currentPath and self.OutputDirectoryDCL.currentPath)
    self.getAtlasButton.enabled = bool(atlasPathSelected and inputPathsSelected)

  def onPointSelectionSelect(self):
    self.subsetApplyButton.enabled = bool(self.DCLLandmarkDirectory.currentPath and self.pointSelection.currentNode())

  def onDCLLandmarkDirectorySelect(self):
    self.subsetApplyButton.enabled = bool(self.DCLLandmarkDirectory.currentPath and self.pointSelection.currentNode())

  def onGenerateAtlasButton(self):
    if self._busy:
      return
    self._busy = True
    self.getAtlasButton.enabled = False
    succeeded = False
    try:
      logic = DeCALogic()
      progressCallback = self.makeProgressCallback(self.progressBarDCL)
      #set up output directory
      self.folderNames = self.setUpDeCADir(self.OutputDirectoryDCL.currentPath, False, False, True, self.loadAtlasOptionDCL.checked)
      if self.folderNames == {}:
        self.logInfoDCL.appendPlainText(f'Output folders could not be created in {self.OutputDirectoryDCL.currentPath}')
        return
      self.folderNames['originalLMs'] = self.landmarkDirectoryDCL.currentPath
      self.folderNames['originalModels'] = self.meshDirectoryDCL.currentPath
      if self.loadAtlasOptionDCL.checked:
        try:
          atlasModelPath = self.DCLBaseModelSelector.currentPath
          self.atlasModel = slicer.util.loadModel(atlasModelPath)
        except:
          self.logInfoDCL.appendPlainText(f"Can't load model from: {atlasModelPath}")
          return
        try:
          atlasLMPath = self.DCLBaseLMSelector.currentPath
          self.atlasLMs = slicer.util.loadMarkups(atlasLMPath)
        except:
          print("Can't load from: ", atlasLMPath)
          self.logInfoDCL.appendPlainText(f"Can't load landmarks from: {atlasLMPath}")
          return
      else:
        removeScale = True
        self.atlasModel, self.atlasLMs = self.generateNewAtlas(removeScale, self.logInfoDCL, progressCallback)
      atlasModelPath = os.path.join(self.folderNames['output'], 'decaAtlasModel.ply')
      self.logInfoDCL.appendPlainText(f"Saving atlas model to {atlasModelPath}")
      slicer.util.saveNode(self.atlasModel, atlasModelPath)
      atlasLMPath = os.path.join(self.folderNames['output'], 'decaAtlasLM.mrk.json')
      self.logInfoDCL.appendPlainText(f"Saving atlas landmarks to {atlasLMPath}")
      slicer.util.saveNode(self.atlasLMs, atlasLMPath)
      self.getPointNumberButton.enabled = True
      succeeded = True
    finally:
      self._busy = False
      self.getAtlasButton.enabled = True
      self.resetProgressBar(self.progressBarDCL, "Atlas ready" if succeeded else "Idle")

  def generateNewAtlas(self, removeScale, log, progressCallback=None):
    logic = DeCALogic()
    closestToMeanLandmarkPath = logic.getClosestToMeanPath(self.folderNames['originalLMs'])
    tempBaseLMs = slicer.util.loadMarkups(os.path.join(self.folderNames['originalLMs'],closestToMeanLandmarkPath))
    subjectID = Path(closestToMeanLandmarkPath)
    while subjectID.suffix in {'.fcsv', '.mrk', '.json'}:
      subjectID = subjectID.with_suffix('')
    log.appendPlainText(f"Sample selected for rigid alignment: {subjectID}")
    tempBaseModel = logic.getModelFileByID(self.folderNames['originalModels'], subjectID)
    try:
      logic.runAlign(tempBaseModel, tempBaseLMs, self.folderNames['originalModels'], self.folderNames['originalLMs'], self.folderNames['tempAlignedModels'], self.folderNames['tempAlignedLMs'], removeScale, progressCallback=progressCallback)
    except ValueError as errorText:
      log.appendPlainText(str(errorText))
      return
    log.appendPlainText(f"Generating the average template")
    atlasModel, atlasLMs = logic.runMean(self.folderNames['tempAlignedLMs'], self.folderNames['tempAlignedModels'], log, progressCallback)
    slicer.mrmlScene.RemoveNode(tempBaseModel)
    slicer.mrmlScene.RemoveNode(tempBaseLMs)
    shutil.rmtree(self.folderNames['tempAlignedModels'])
    shutil.rmtree(self.folderNames['tempAlignedLMs'])
    return atlasModel, atlasLMs

  def onGetPointNumberButton(self):
    logic = DeCALogic()
    subsampledTemplate, pointNumber = logic.runCheckPoints(self.atlasModel, self.spacingTolerance.value)
    self.logInfoDCL.appendPlainText(f'The subsampled template has a total of {pointNumber} points.')
    self.DCLApplyButton.enabled = True

  def onDCApplyButton(self):
    if self._busy:
      return
    self._busy = True
    self.applyButtonDC.enabled = False
    succeeded = False
    try:
      logic = DeCALogic()
      progressCallback = self.makeProgressCallback(self.progressBarDC)
      #set up output directory
      symmetryOption = self.analysisTypeSymmetry.checked
      writeErrorOption = self.writeErrorCheckBox.checked
      loadAtlasOption = self.loadAtlasOptionDC.checked
      removeScaleOption = self.removeScaleCheckBoxDC.checked

      # Validate symmetry inputs BEFORE running any pipeline steps
      if symmetryOption:
        # First, get the expected landmark count from the files
        try:
          expected_landmark_count = self.getActualLandmarkCount(self.landmarkDirectoryDC.currentPath)
        except Exception as e:
          self.logInfoDC.appendPlainText(f"Symmetry Error: {str(e)}")
          print(f"DeCA Symmetry Error: {str(e)}")
          return # Stop execution before any processing

        # Now validate and generate the mirror map with the expected count
        try:
          mirror_map_string = self.generateMirrorMapString(expected_landmark_count)
          self.logInfoDC.appendPlainText(f"Symmetry validation passed. All {expected_landmark_count} landmarks specified.")
        except Exception as e:
          self.logInfoDC.appendPlainText(f"Symmetry Error: {str(e)}")
          print(f"DeCA Symmetry Error: {str(e)}")
          return # Stop execution before any processing

      self.folderNames = self.setUpDeCADir(self.outputDirectoryDC.currentPath, symmetryOption, writeErrorOption, False, loadAtlasOption)
      if self.folderNames == {}:
        self.logInfoDC.appendPlainText(f'Output folders could not be created in {self.outputDirectoryDC.currentPath}')
        return
      self.folderNames['originalLMs'] = self.landmarkDirectoryDC.currentPath
      self.folderNames['originalModels'] = self.meshDirectoryDC.currentPath

      #generate or load atlas
      if loadAtlasOption:
        try:
          atlasModelPath = self.DCBaseModelSelector.currentPath
          self.atlasModel = slicer.util.loadModel(atlasModelPath)
        except:
          self.logInfoDC.appendPlainText(f"Can't load model from: {atlasModelPath}")
          return
        try:
          atlasLMPath = self.DCBaseLMSelector.currentPath
          self.atlasLMs = slicer.util.loadMarkups(atlasLMPath)
        except:
          print("Can't load from: ", atlasLMPath)
          self.logInfoDC.appendPlainText(f"Can't load landmarks from: {atlasLMPath}")
          return
      else:
        self.atlasModel, self.atlasLMs = self.generateNewAtlas(removeScaleOption, self.logInfoDC, progressCallback)
      # save atlas model and landmarks to output file
      atlasModelPath = os.path.join(self.folderNames['output'], 'decaAtlasModel.ply')
      self.logInfoDC.appendPlainText(f"Saving atlas model to {atlasModelPath}")
      slicer.util.saveNode(self.atlasModel, atlasModelPath)
      atlasLMPath = os.path.join(self.folderNames['output'], 'decaAtlasLM.mrk.json')
      self.logInfoDC.appendPlainText(f"Saving atlas landmarks to {atlasLMPath}")
      slicer.util.saveNode(self.atlasLMs, atlasLMPath)
      # rigid alignment to atlas
      try:
        logic.runAlign(self.atlasModel, self.atlasLMs, self.folderNames['originalModels'], self.folderNames['originalLMs'], self.folderNames['alignedModels'], self.folderNames['alignedLMs'], removeScaleOption, progressCallback=progressCallback)
      except ValueError as errorText:
        self.logInfoDC.appendPlainText(str(errorText))
        return
      # run DeCA shape analysis
      if self.analysisTypeShape.checked:
        self.logInfoDC.appendPlainText(f"Calculating point correspondences to atlas")
        logic.runDCAlign(atlasModelPath, atlasLMPath, self.folderNames['alignedModels'],
        self.folderNames['alignedLMs'], self.folderNames['output'], self.writeErrorCheckBox.checked, progressCallback)
      # run DeCA symmetry analysis
      else:
        ##
        ## MODIFIED SECTION: Use the mirror map string already validated and generated at start
        ##
        # generate mirrored landmarks and models
        axis = [-1,1,1] #set symmetry to x-axis

        # mirror_map_string was already generated and validated at the start of onDCApplyButton
        self.logInfoDC.appendPlainText(f"Generating mirrored models and landmarks")
        logic.runMirroring(self.folderNames['alignedModels'], self.folderNames['alignedLMs'], self.folderNames['mirrorModels'],
        self.folderNames['mirrorLMs'], axis, mirror_map_string) # Use the validated string
        self.logInfoDC.appendPlainText(f"Calculating point correspondences to atlas")
        logic.runDCAlignSymmetric(atlasModelPath, atlasLMPath, self.folderNames['alignedModels'],
        self.folderNames['alignedLMs'], self.folderNames['mirrorModels'], self.folderNames['mirrorLMs'], self.folderNames['output'],
        self.writeErrorCheckBox.checked, progressCallback)
        ##
        ## END OF MODIFIED SECTION
        ##
      slicer.mrmlScene.RemoveNode(self.atlasModel)
      slicer.mrmlScene.RemoveNode(self.atlasLMs)
      succeeded = True
    finally:
      self._busy = False
      self.applyButtonDC.enabled = True
      self.resetProgressBar(self.progressBarDC, "Done" if succeeded else "Idle")

  def onDCLApplyButton(self):
    if self._busy:
      return
    self._busy = True
    self.DCLApplyButton.enabled = False
    succeeded = False
    try:
      logic = DeCALogic()
      progressCallback = self.makeProgressCallback(self.progressBarDCL)
      # rigidly align to template
      self.logInfoDCL.appendPlainText(f"Rigid alignment to the atlas")
      removeScale = True
      # only persist per-subject alignment transforms when original-frame output is
      # requested, so an unchecked run does no extra transform I/O
      transformDirectory = self.folderNames['alignmentTransforms'] if self.originalFrameCheckBoxDCL.checked else None
      try:
        logic.runAlign(self.atlasModel, self.atlasLMs, self.folderNames['originalModels'], self.folderNames['originalLMs'],self.folderNames['alignedModels'], self.folderNames['alignedLMs'], removeScale, transformDirectory=transformDirectory, progressCallback=progressCallback)
      except ValueError as errorText:
        self.logInfoDCL.appendPlainText(str(errorText))
        return
      # generate point correspondences
      self.logInfoDCL.appendPlainText(f"Calculating point correspondences")
      atlasDenseLandmarks = logic.runDeCAL(self.atlasModel, self.atlasLMs, self.folderNames['alignedModels'],
      self.folderNames['alignedLMs'], self.folderNames['DeCALOutput'], self.spacingTolerance.value, progressCallback,
      useFastCorrespondence=self.fastCorrespondenceCheckBoxDCL.checked)
      # optionally merge the generated semi-landmarks with the fixed landmarks used
      # to establish correspondence (both are in the atlas-aligned coordinate frame)
      mergedCount = None
      mergedDirectory = os.path.join(self.folderNames['output'], "mergedLMs")
      if self.mergeLandmarksCheckBoxDCL.checked:
        try:
          os.makedirs(mergedDirectory, exist_ok=True)
        except OSError:
          self.logInfoDCL.appendPlainText(f"Could not create merged landmark folder: {mergedDirectory}")
        else:
          self.logInfoDCL.appendPlainText(f"Merging fixed and semi-landmarks into {mergedDirectory}")
          atlasFixedLMPath = os.path.join(self.folderNames['output'], 'decaAtlasLM.mrk.json')
          mergedCount = logic.runMergeLandmarks(self.folderNames['alignedLMs'], self.folderNames['DeCALOutput'], mergedDirectory, atlasFixedLMPath)
          if mergedCount is None:
            self.logInfoDCL.appendPlainText("Merge skipped: the SlicerMorph extension (MergeMarkups module) is required. Please install SlicerMorph.")
          else:
            self.logInfoDCL.appendPlainText(f"Saved {mergedCount} merged landmark files.")
      # optionally also express the output in each subject's original (un-aligned)
      # coordinate frame by inverting the saved per-subject alignment transform
      if self.originalFrameCheckBoxDCL.checked:
        originalSemiDirectory = os.path.join(self.folderNames['output'], "DeCALOutput_originalFrame")
        self.logInfoDCL.appendPlainText(f"Mapping semi-landmarks back to the original model coordinate frame")
        semiCount = logic.runBackTransformLandmarks(self.folderNames['DeCALOutput'], transformDirectory, originalSemiDirectory)
        self.logInfoDCL.appendPlainText(f"Saved {semiCount} original-frame semi-landmark files to {originalSemiDirectory}")
        # if merged files were produced, back-transform them too (descriptions preserved)
        if mergedCount:
          mergedOriginalDirectory = os.path.join(self.folderNames['output'], "mergedLMs_originalFrame")
          mergedOriginalCount = logic.runBackTransformLandmarks(mergedDirectory, transformDirectory, mergedOriginalDirectory, "_merged")
          self.logInfoDCL.appendPlainText(f"Saved {mergedOriginalCount} original-frame merged landmark files to {mergedOriginalDirectory}")
      # setup for optional subsetting
      self.pointSelection.setCurrentNode(atlasDenseLandmarks)
      self.DCLLandmarkDirectory.setCurrentPath(self.folderNames['DeCALOutput'])
      succeeded = True
    finally:
      self._busy = False
      self.DCLApplyButton.enabled = True
      self.resetProgressBar(self.progressBarDCL, "Done" if succeeded else "Idle")

  def onSubsetApplyButton(self):
    logic = DeCALogic()
    topDir = os.path.dirname(self.DCLLandmarkDirectory.currentPath)
    lmDirectorySubset = os.path.join(topDir, "DeCALSubset")
    os.makedirs(lmDirectorySubset)
    atlasNode = self.pointSelection.currentNode()
    lmDirectorySubset = logic.runSubsetLandmarks(atlasNode, self.DCLLandmarkDirectory.currentPath, lmDirectorySubset)

  ##
  ## NEW HELPER FUNCTION: getActualLandmarkCount
  ##
  def getActualLandmarkCount(self, landmarkDirectory):
    """
    Gets the actual number of landmarks from the first file in the directory.
    Raises ValueError if no files found or can't read them.
    """
    import os
    landmark_files = [f for f in os.listdir(landmarkDirectory) if f.endswith('.fcsv') or f.endswith('.json')]
    
    if not landmark_files:
      raise ValueError(f"No landmark files (.fcsv or .json) found in directory: {landmarkDirectory}")
    
    # Load the first landmark file to check the count
    sample_file = os.path.join(landmarkDirectory, landmark_files[0])
    try:
      sample_lm = slicer.util.loadMarkups(sample_file)
      landmark_count = sample_lm.GetNumberOfControlPoints()
      slicer.mrmlScene.RemoveNode(sample_lm)
      return landmark_count
    except Exception as e:
      raise ValueError(f"Error reading landmark file {sample_file}: {str(e)}")
  ##
  ## END OF NEW HELPER FUNCTION
  ##
  
  ##
  ## NEW HELPER FUNCTION: validateSymmetryLandmarksAgainstFiles (DEPRECATED - validation moved to generateMirrorMapString)
  ##
  def validateSymmetryLandmarksAgainstFiles(self, landmarkDirectory, mirror_map_string):
    """
    Validates that the generated mirror map matches the actual number of landmarks in the files.
    Raises ValueError if there's a mismatch.
    """
    # Get the expected number of landmarks from the mirror map
    expected_landmark_count = len(mirror_map_string.split(','))
    
    # Check a sample landmark file to get the actual landmark count
    import os
    landmark_files = [f for f in os.listdir(landmarkDirectory) if f.endswith('.fcsv') or f.endswith('.json')]
    
    if not landmark_files:
      raise ValueError(f"No landmark files (.fcsv or .json) found in directory: {landmarkDirectory}")
    
    # Load the first landmark file to check the count
    sample_file = os.path.join(landmarkDirectory, landmark_files[0])
    try:
      sample_lm = slicer.util.loadMarkups(sample_file)
      actual_landmark_count = sample_lm.GetNumberOfControlPoints()
      slicer.mrmlScene.RemoveNode(sample_lm)
      
      if expected_landmark_count != actual_landmark_count:
        raise ValueError(
          f"Landmark count mismatch: You specified {expected_landmark_count} landmarks for symmetry analysis, "
          f"but the landmark files contain {actual_landmark_count} landmarks. "
          f"Please ensure all landmarks (1-{actual_landmark_count}) are assigned to Midline, Left, or Right fields."
        )
      
      self.logInfoDC.appendPlainText(f"Validated: Symmetry map matches {actual_landmark_count} landmarks in files.")
      
    except Exception as e:
      if "Landmark count mismatch" in str(e):
        raise
      else:
        raise ValueError(f"Error validating landmark file {sample_file}: {str(e)}")
  ##
  ## END OF NEW HELPER FUNCTION
  ##
  
  ##
  ## NEW HELPER FUNCTION: _parse_indices
  ##
  def _parse_indices(self, text):
    """
    Parses a comma-separated string of 1-based indices into a list of 0-based ints.
    """
    if not text.strip():
      return []
    try:
      # Split by comma, strip whitespace, remove empty strings, convert to int, and make 0-based
      indices = [int(x.strip()) - 1 for x in text.split(',') if x.strip()]
      # Check for 0 or negative indices after 1-based conversion
      if any(i < 0 for i in indices):
        raise ValueError("Invalid index. Indices must be 1-based (e.g., '1, 2, 3'). Found '0' or a negative number.")
      return indices
    except (ValueError, TypeError) as e:
      raise ValueError(f"Error parsing indices: '{e}'. Please use comma-separated numbers (e.g., '1, 2, 3').")
  ##
  ## END OF NEW HELPER FUNCTION
  ##
  
  ##
  ## NEW HELPER FUNCTION: generateMirrorMapString
  ##
  def generateMirrorMapString(self, expected_landmark_count=None):
    """
    Generates the 0-based full mirror map string from the midline, left, and right UI fields.
    Prints the result to the console and returns the string.
    Raises a ValueError if validation fails.
    
    Args:
      expected_landmark_count: If provided, validates that all landmarks are specified
    """
    try:
      midline_indices = self._parse_indices(self.midlineLandmarksText.text)
      left_indices = self._parse_indices(self.leftLandmarksText.text)
      right_indices = self._parse_indices(self.rightLandmarksText.text)

      if len(left_indices) != len(right_indices):
        raise ValueError(f"Error: Left ({len(left_indices)}) and Right ({len(right_indices)}) landmark lists have different lengths.")
      
      num_specified_landmarks = len(midline_indices) + len(left_indices) + len(right_indices)
      if num_specified_landmarks == 0:
        raise ValueError("Error: No indices provided in Midline, Left, or Right fields.")
            
      # Check for duplicates
      all_indices_list = midline_indices + left_indices + right_indices
      all_indices_set = set(all_indices_list)
      if len(all_indices_list) != len(all_indices_set):
        raise ValueError("Error: Duplicate indices found. Each landmark must be in only one list (midline, left, or right).")

      # If expected count is provided, validate that ALL landmarks are specified
      if expected_landmark_count is not None:
        expected_indices = set(range(expected_landmark_count))
        if all_indices_set != expected_indices:
          missing_indices = expected_indices - all_indices_set
          extra_indices = all_indices_set - expected_indices
          
          error_msg = f"Error: Not all landmarks are specified. The landmark files contain {expected_landmark_count} landmarks."
          if missing_indices:
            missing_1based = sorted([x + 1 for x in missing_indices])
            error_msg += f"\nMissing landmarks (1-based): {missing_1based}"
          if extra_indices:
            extra_1based = sorted([x + 1 for x in extra_indices])
            error_msg += f"\nExtra landmarks not in files (1-based): {extra_1based}"
          error_msg += f"\nPlease ensure ALL landmarks (1-{expected_landmark_count}) are assigned to Midline, Left, or Right fields."
          raise ValueError(error_msg)

      # The mirror map size must accommodate the highest landmark index
      max_index = max(all_indices_list)
      total_landmarks = max_index + 1  # 0-based, so add 1
      
      # All checks passed, create the map
      # Initialize: each landmark maps to itself by default
      mirror_map = list(range(total_landmarks))
      
      for i in midline_indices:
        mirror_map[i] = i
      
      for l_idx, r_idx in zip(left_indices, right_indices):
        mirror_map[l_idx] = r_idx
        mirror_map[r_idx] = l_idx
      
      # Convert list of ints to comma-separated string
      map_string = ",".join(map(str, mirror_map))
      
      # Print to console as requested
      print("--- DeCA Symmetry Map Generated ---")
      print(map_string)
      print(f"Total Landmarks: {total_landmarks}")
      print("-------------------------------------")
      
      return map_string
      
    except Exception as e:
      # Re-raise the exception to be caught by onDCApplyButton
      raise e
  ##
  ## END OF NEW HELPER FUNCTION
  ##

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

  def _existingLandmarkFileIsComplete(self, markupsPath, expectedPointCount):
    # Resume must only skip a subject whose existing output is genuinely complete
    # for the CURRENT parameters. Load the file and require its control-point count
    # to equal expectedPointCount (derived from the current spacingTolerance / base
    # mesh). A file truncated by a crash mid-write, or left over from a run with a
    # different point density, fails this check and is recomputed rather than being
    # silently trusted. (Resume still assumes the same atlas; the normal workflow
    # writes each atlas run to its own timestamped output folder.)
    node = None
    try:
      node = slicer.util.loadMarkups(markupsPath)
      return node is not None and node.GetNumberOfControlPoints() == expectedPointCount
    except Exception:
      return False
    finally:
      if node is not None:
        self._removeNodeFully(node)

  def runDeCAL(self, baseNode, baseLMPath, meshDirectory, landmarkDirectory, outputDirectory, spacingTolerance, progressCallback=None, useFastCorrespondence=False):
    spacingPercentage = spacingTolerance/100
    loadOption=False
    baseLandmarks=self.fiducialNodeToPolyData(baseLMPath, loadOption).GetPoints()
    modelExt=['ply','stl','vtp', 'vtk']
    self.outputDirectory = outputDirectory
    # Landmarks are small, so load them all -- Procrustes needs the whole sample.
    landmarkNames, landmarks = self.importLandmarks(landmarkDirectory, progressCallback)
    # Mesh files in the same sorted order importMeshes/importLandmarks use, so the
    # i-th mesh matches the i-th landmark block. Meshes are loaded one at a time in
    # the loop below (not all up front) to keep memory flat on large datasets.
    meshFiles = sorted(f for f in os.listdir(meshDirectory) if f.endswith(tuple(modelExt)))
    self.modelNames = [os.path.splitext(f)[0] for f in meshFiles]
    # meanShape, meanWarpedBase and the subsampling index are all independent of the
    # per-subject correspondence, so compute them once up front. This lets each
    # subject be computed, written and discarded inside the loop instead of building
    # every corresponding mesh in memory and writing them all at the end.
    meanShape, alignedPoints = self.procrustesImposition(landmarks, False)
    meanWarpedBase = self._warpBaseMesh(baseNode.GetPolyData(), baseLandmarks, meanShape)
    indexArrayName = "indexArray"
    self.addIndexArray(baseNode, indexArrayName)
    templateModel = self.downsampleModel(baseNode, spacingPercentage)
    templateIndex = templateModel.GetPointData().GetArray(indexArrayName)
    if not templateIndex:
      print("No index found")
      return None
    sampleNumber = alignedPoints.GetNumberOfBlocks()
    pointCount = templateIndex.GetNumberOfValues()
    print("sample number:", sampleNumber)
    # The resume-by-file-existence below only checks point count, which is identical
    # for the exact and fast correspondence methods, so resuming into a folder that
    # already holds results from a different method (or point density) would silently
    # blend them. Record this run's method + point count once (in the parent run
    # folder, which no landmark loader reads) and refuse to reuse a folder whose
    # recorded settings differ, so mismatches fail loudly instead of mixing methods.
    # The normal workflow writes each run to its own timestamped folder, so this only
    # trips on deliberate folder reuse.
    import json
    runInfo = {"useFastCorrespondence": bool(useFastCorrespondence), "pointCount": int(pointCount)}
    runInfoPath = os.path.join(os.path.dirname(os.path.normpath(outputDirectory)), ".decal_run_info")
    if os.path.exists(runInfoPath):
      try:
        with open(runInfoPath) as runInfoFile:
          existingRunInfo = json.load(runInfoFile)
      except Exception:
        existingRunInfo = None
      if existingRunInfo != runInfo:
        raise ValueError(
          "The output folder already holds DeCAL results computed with different settings "
          "(%s vs requested %s). Use a fresh output folder so exact and fast correspondences "
          "are never mixed in one result." % (existingRunInfo, runInfo))
    else:
      try:
        with open(runInfoPath, "w") as runInfoFile:
          json.dump(runInfo, runInfoFile)
      except OSError:
        pass  # marker is best-effort; do not fail the run if it cannot be written
    # Write each subject's downsampled correspondence as soon as it is computed. A
    # crash then keeps every file already written, and re-running skips subjects
    # whose output already exists (resume). Batch the scene state and pause
    # rendering so the per-point AddControlPoint calls and the on-demand model
    # loads/removes do not fire per-item scene updates or renders.
    basePointNode = None
    slicer.app.pauseRender()
    slicer.mrmlScene.StartState(slicer.vtkMRMLScene.BatchProcessState)
    try:
      for i in range(sampleNumber):
        if progressCallback:
          progressCallback(i + 1, sampleNumber, "Computing dense correspondence")
        outputLMPath = os.path.join(outputDirectory, self.modelNames[i] + ".mrk.json")
        if os.path.exists(outputLMPath) and self._existingLandmarkFileIsComplete(outputLMPath, pointCount):
          print("Skipping " + self.modelNames[i] + ": complete output already present (resume)")
          continue
        subjectModelNode = slicer.util.loadModel(os.path.join(meshDirectory, meshFiles[i]))
        correspondingMesh = self.denseSurfaceCorrespondencePair(
          subjectModelNode.GetPolyData(), landmarks.GetBlock(i).GetPoints(),
          meanWarpedBase, meanShape, i, useFast=useFastCorrespondence)
        alignedPointNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLMarkupsFiducialNode', "alignedPoints")
        for j in range(pointCount):
          baseIndex = templateIndex.GetValue(j)
          alignedPointNode.AddControlPoint(correspondingMesh.GetPoint(baseIndex), str(j))
        slicer.util.saveNode(alignedPointNode, outputLMPath)
        self._removeNodeFully(alignedPointNode)
        self._removeNodeFully(subjectModelNode)
      # atlas (base) correspondence -- independent of the subjects
      basePointNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLMarkupsFiducialNode', "atlasLandmarks")
      for j in range(pointCount):
        baseIndex = templateIndex.GetValue(j)
        basePointNode.AddControlPoint(baseNode.GetPolyData().GetPoint(baseIndex), str(j))
      baseLMPath = os.path.join(outputDirectory, "atlas.mrk.json")
      slicer.util.saveNode(basePointNode, baseLMPath)
    finally:
      slicer.mrmlScene.EndState(slicer.vtkMRMLScene.BatchProcessState)
      slicer.app.resumeRender()
    return basePointNode

  def runMergeLandmarks(self, fixedLMDirectory, semiLMDirectory, outputDirectory, atlasFixedLMPath=None):
    # Merge each subject's fixed landmarks (used to establish correspondence) with
    # the DeCAL-generated semi-landmarks, and also merge the atlas itself (its fixed
    # landmarks at atlasFixedLMPath with the atlas dense points saved as
    # atlas.mrk.json). The per-subject fixed and semi files share a basename, so they
    # are matched by filename. Fixed points get the description "Fixed" and semi
    # points get "Semi". Returns the number of merged files written, or None if the
    # SlicerMorph MergeMarkups module is not available (extension not installed).
    try:
      import MergeMarkups
      mergeLogic = MergeMarkups.MergeMarkupsLogic()
    except (ImportError, AttributeError):
      return None

    def mergeAndSave(fixedPath, semiPath, outputName):
      fixedNode = semiNode = mergedNode = None
      try:
        fixedNode = slicer.util.loadMarkups(fixedPath)
        semiNode = slicer.util.loadMarkups(semiPath)
        # mergeLMNodes concatenates fixed then semi points and tags them with the
        # "Fixed"/"Semi" control point descriptions
        mergedNode = mergeLogic.mergeLMNodes(fixedNode, semiNode)
        mergedNode.SetName(outputName)
        slicer.util.saveNode(mergedNode, os.path.join(outputDirectory, outputName + ".mrk.json"))
      finally:
        # always remove any nodes we loaded/created, even if the merge failed
        for node in (fixedNode, semiNode, mergedNode):
          if node is not None:
            slicer.mrmlScene.RemoveNode(node)

    mergedCount = 0
    # mergeLMNodes inserts control points one at a time; while the node is in the
    # scene each insertion fires a scene/subject-hierarchy update, so a large point
    # set takes several seconds to merge. Batching the scene state and pausing
    # rendering around the whole loop suppresses those per-point updates (~150x
    # faster on dense DeCAL output). EndState/resumeRender must always run, hence
    # the try/finally.
    slicer.app.pauseRender()
    slicer.mrmlScene.StartState(slicer.vtkMRMLScene.BatchProcessState)
    try:
      for semiFileName in sorted(os.listdir(semiLMDirectory)):
        if semiFileName.startswith(".") or not semiFileName.endswith((".fcsv", ".json")):
          continue
        # the atlas point set (atlas.mrk.json) has no matching per-subject fixed
        # file here; it is merged separately below using atlasFixedLMPath
        fixedFilePath = os.path.join(fixedLMDirectory, semiFileName)
        if not os.path.exists(fixedFilePath):
          continue
        subjectID = Path(semiFileName)
        while subjectID.suffix in {'.fcsv', '.mrk', '.json'}:
          subjectID = subjectID.with_suffix('')
        # merging is an optional post-processing step; a failure on one subject
        # (e.g. a malformed file) must not abort the whole apply flow, so log and skip
        try:
          mergeAndSave(fixedFilePath, os.path.join(semiLMDirectory, semiFileName), str(subjectID) + "_merged")
          mergedCount += 1
        except Exception as e:
          logging.warning(f"DeCAL merge: skipping {semiFileName} ({e})")
      # merge the atlas: its fixed landmarks with its dense correspondence points
      atlasSemiPath = os.path.join(semiLMDirectory, "atlas.mrk.json")
      if atlasFixedLMPath and os.path.exists(atlasFixedLMPath) and os.path.exists(atlasSemiPath):
        try:
          mergeAndSave(atlasFixedLMPath, atlasSemiPath, "atlas_merged")
          mergedCount += 1
        except Exception as e:
          logging.warning(f"DeCAL merge: skipping atlas ({e})")
    finally:
      slicer.mrmlScene.EndState(slicer.vtkMRMLScene.BatchProcessState)
      slicer.app.resumeRender()
    return mergedCount

  def runBackTransformLandmarks(self, landmarkDirectory, transformDirectory, outputDirectory, transformSuffix=""):
    # Map each aligned-frame landmark file in landmarkDirectory back into the
    # original (pre-alignment) coordinate frame of its subject by inverting the
    # per-subject similarity/rigid transform saved by runAlign. The transform is
    # matched by the landmark file's basename (optionally with transformSuffix
    # removed, e.g. "_merged"). Files with no matching transform - notably the
    # atlas point set, which is already the reference frame - are skipped.
    # Returns the number of files written.
    if not os.path.isdir(landmarkDirectory):
      return 0
    os.makedirs(outputDirectory, exist_ok=True)
    writtenCount = 0
    # each file loads/saves markups and transform nodes; batching the scene state
    # and pausing rendering around the loop suppresses per-node scene updates (same
    # optimization as runMergeLandmarks). EndState/resumeRender must always run.
    slicer.app.pauseRender()
    slicer.mrmlScene.StartState(slicer.vtkMRMLScene.BatchProcessState)
    try:
      for lmFileName in sorted(os.listdir(landmarkDirectory)):
        if lmFileName.startswith(".") or not lmFileName.endswith((".fcsv", ".json")):
          continue
        base = Path(lmFileName)
        while base.suffix in {'.fcsv', '.mrk', '.json'}:
          base = base.with_suffix('')
        transformKey = str(base)
        if transformSuffix and transformKey.endswith(transformSuffix):
          transformKey = transformKey[:-len(transformSuffix)]
        transformPath = os.path.join(transformDirectory, transformKey + ".h5")
        if not os.path.exists(transformPath):
          # expected for the atlas point set (no per-subject alignment); log for others
          logging.info(f"DeCAL back-transform: no alignment transform for {lmFileName}, skipping")
          continue
        lmNode = xfNode = inverseNode = None
        try:
          lmNode = slicer.util.loadMarkups(os.path.join(landmarkDirectory, lmFileName))
          xfNode = slicer.util.loadTransform(transformPath)
          forwardMatrix = vtk.vtkMatrix4x4()
          if not xfNode.GetMatrixTransformToParent(forwardMatrix):
            raise ValueError("alignment transform is not linear")
          inverseMatrix = vtk.vtkMatrix4x4()
          vtk.vtkMatrix4x4.Invert(forwardMatrix, inverseMatrix)
          inverseNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLinearTransformNode", "DeCALInverseTransform")
          inverseNode.SetMatrixTransformToParent(inverseMatrix)
          lmNode.SetAndObserveTransformNodeID(inverseNode.GetID())
          slicer.vtkSlicerTransformLogic().hardenTransform(lmNode)
          slicer.util.saveNode(lmNode, os.path.join(outputDirectory, lmFileName))
          writtenCount += 1
        except Exception as e:
          logging.warning(f"DeCAL back-transform: skipping {lmFileName} ({e})")
        finally:
          for node in (lmNode, xfNode, inverseNode):
            if node is not None:
              slicer.mrmlScene.RemoveNode(node)
    finally:
      slicer.mrmlScene.EndState(slicer.vtkMRMLScene.BatchProcessState)
      slicer.app.resumeRender()
    return writtenCount

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

  def runMirroring(self, meshDirectory, lmDirectory, mirrorMeshDirectory, mirrorLMDirectory, mirrorAxis, mirrorIndexText, slmDirectory=None, outputSLMDirectory=None, mirrorSLMIndexText=None):
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
    semilandmarkOption = bool(slmDirectory and outputSLMDirectory and (len(mirrorSLMIndexText) != 0))
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
          ##
          ## MODIFICATION CHECK: This block needs to be robust to index errors
          ## The new validation in generateMirrorMapString should prevent out-of-bounds
          ## errors here, as long as all landmark files have the *same* number of points.
          ##
          try:
            for i in range(currentLMNode.GetNumberOfControlPoints()):
              mirror_idx = mirrorIndex[i]
              point = currentLMNode.GetNthControlPointPosition(mirror_idx)
              mirrorLMNode.AddControlPoint(point, str(i))
              sourcePoints.InsertNextPoint(point)
          except IndexError:
            # This would happen if the map string has a higher index than points available.
            # The validation should catch this, but good to be aware.
            slicer.mrmlScene.RemoveNode(currentLMNode)
            slicer.mrmlScene.RemoveNode(currentMeshNode)
            slicer.mrmlScene.RemoveNode(mirrorTransformNode)
            slicer.mrmlScene.RemoveNode(mirrorLMNode)
            raise ValueError(f"Symmetry map error for subject {subjectID}: The generated map has an index ({mirror_idx if 'mirror_idx' in locals() else 'unknown'}) that is out of bounds for the landmark file (total points: {currentLMNode.GetNumberOfControlPoints()}).")
          
          rigidTransform = vtk.vtkLandmarkTransform()
          rigidTransform.SetSourceLandmarks(sourcePoints)
          rigidTransform.SetTargetLandmarks(targetPoints)
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

  def runDCAlign(self, baseMeshPath, baseLMPath, meshDirectory, landmarkDirectory, outputDirectory, optionErrorOutput, progressCallback=None):
    if optionErrorOutput:
      self.errorCheckPath = os.path.join(outputDirectory, "errorChecking")
      if not os.path.exists(self.errorCheckPath):
        os.mkdir(self.errorCheckPath)
    baseNode = slicer.util.loadModel(baseMeshPath)
    baseMesh = baseNode.GetPolyData()
    baseLandmarks=self.fiducialNodeToPolyData(baseLMPath).GetPoints()
    modelExt=['ply','stl','vtp']
    self.modelNames, models = self.importMeshes(meshDirectory, modelExt, progressCallback)
    landmarkNames,landmarks = self.importLandmarks(landmarkDirectory, progressCallback)
    denseCorrespondenceGroup = self.denseCorrespondenceBaseMesh(landmarks, models, baseMesh, baseLandmarks, progressCallback)
    self.addMagnitudeFeature(denseCorrespondenceGroup, self.modelNames, baseMesh)
    # save results to output directory
    outputModelName = 'decaResultModel.vtp'
    outputModelPath = os.path.join(outputDirectory, outputModelName)
    slicer.util.saveNode(baseNode, outputModelPath)

  def runDCAlignSymmetric(self, baseMeshPath, baseLMPath, meshDir, landmarkDir, mirrorMeshDir, mirrorLandmarkDir, outputDir, optionErrorOutput, progressCallback=None):
    if optionErrorOutput:
      self.errorCheckPath = os.path.join(outputDir, "errorChecking")
      if not os.path.exists(self.errorCheckPath):
        os.mkdir(self.errorCheckPath)
    baseNode = slicer.util.loadModel(baseMeshPath)
    baseMesh = baseNode.GetPolyData()
    baseLandmarks=self.fiducialNodeToPolyData(baseLMPath).GetPoints()
    modelExt=['ply','stl','vtp']
    self.modelNames, models = self.importMeshes(meshDir, modelExt, progressCallback)
    landmarkNames, landmarks = self.importLandmarks(landmarkDir, progressCallback)
    modelMirrorNames, mirrorModels = self.importMeshes(mirrorMeshDir, modelExt, progressCallback)
    mirrorLandmarkNames, mirrorLandmarks = self.importLandmarks(mirrorLandmarkDir, progressCallback)
    denseCorrespondenceGroup = self.denseCorrespondenceBaseMesh(landmarks, models, baseMesh, baseLandmarks, progressCallback)
    denseCorrespondenceGroupMirror = self.denseCorrespondenceBaseMesh(mirrorLandmarks, mirrorModels, baseMesh, baseLandmarks, progressCallback)
    self.addMagnitudeFeatureSymmetry(denseCorrespondenceGroup, denseCorrespondenceGroupMirror, self.modelNames, baseMesh)
    # save results to output directory
    outputModelName = 'decaSymmetryResultModel.vtp'
    outputModelPath = os.path.join(outputDir, outputModelName)
    slicer.util.saveNode(baseNode, outputModelPath)

  def runMean(self, landmarkDirectory, meshDirectory, log=None, progressCallback=None):
    modelExt=['ply','stl','vtp','vtk']
    self.modelNames, models = self.importMeshes(meshDirectory, modelExt, progressCallback)
    landmarkNames, landmarks = self.importLandmarks(landmarkDirectory, progressCallback)
    [denseCorrespondenceGroup, closestToMeanIndex] = self.denseCorrespondence(landmarks, models, progressCallback=progressCallback)
    if log:
      log.appendPlainText(f"Sample selected for base model calculation: {self.modelNames[closestToMeanIndex]}")
    # compute mean model
    averagePolyData = self.computeAverageModelFromGroup(denseCorrespondenceGroup, closestToMeanIndex)
    averageModelNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLModelNode', 'Atlas Model')
    averageModelNode.CreateDefaultDisplayNodes()
    averageModelNode.SetAndObservePolyData(averagePolyData)
     # compute mean landmarks
    averageLandmarkNode = self.computeAverageLM(landmarks)
    averageLandmarkNode.GetDisplayNode().SetPointLabelsVisibility(False)
    return averageModelNode, averageLandmarkNode

  def buildLandmarkFileIndex(self, directory):
    # Map subjectID -> landmark filename by stripping landmark suffixes. Built
    # once by callers that would otherwise call getLandmarkFileByID in a loop,
    # which re-lists the whole directory on every lookup (O(N^2) for N subjects).
    fileIndex = {}
    for fileName in os.listdir(directory):
      fileNameBase = Path(fileName)
      while fileNameBase.suffix in {'.fcsv', '.mrk', '.json'}:
        fileNameBase = fileNameBase.with_suffix('')
      # first match wins, matching getLandmarkFileByID's os.listdir scan order
      fileIndex.setdefault(str(fileNameBase), fileName)
    return fileIndex

  def getLandmarkFileByID(self, directory, subjectID, fileIndex=None):
    if fileIndex is not None:
      # use a prebuilt {subjectID: fileName} index to avoid re-listing directory
      fileName = fileIndex.get(subjectID)
      if fileName is not None:
        return slicer.util.loadMarkups(os.path.join(directory, fileName))
      return None
    fileList = os.listdir(directory)
    for fileName in fileList:
      fileNameBase = Path(fileName)
      while fileNameBase.suffix in {'.fcsv', '.mrk', '.json'}:
        fileNameBase = fileNameBase.with_suffix('')
      if subjectID == str(fileNameBase):
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

  def _removeNodeFully(self, node):
    # Remove a node together with the display and storage nodes that
    # slicer.util.loadModel / loadMarkups create for it. A plain
    # RemoveNode(node) leaves those associated nodes orphaned in the scene, and
    # across a long per-subject loop they accumulate and progressively slow the
    # scene (and its rendering) down.
    if node is None:
      return
    associatedNodes = []
    if node.IsA("vtkMRMLDisplayableNode"):
      for i in range(node.GetNumberOfDisplayNodes()):
        associatedNodes.append(node.GetNthDisplayNode(i))
    if node.IsA("vtkMRMLStorableNode"):
      for i in range(node.GetNumberOfStorageNodes()):
        associatedNodes.append(node.GetNthStorageNode(i))
    slicer.mrmlScene.RemoveNode(node)
    for associatedNode in associatedNodes:
      if associatedNode is not None:
        slicer.mrmlScene.RemoveNode(associatedNode)

  def runAlign(self, baseMeshNode, baseLMNode, meshDirectory, lmDirectory, ouputMeshDirectory, outputLMDirectory, removeScaleOption, slmDirectory=False, outputSLMDirector=False, transformDirectory=None, progressCallback=None):
    semilandmarkOption = bool(slmDirectory and outputSLMDirectory)
    targetPoints = vtk.vtkPoints()
    point=[0,0,0]
    # Set up base points for transform
    for i in range(baseLMNode.GetNumberOfControlPoints()):
      point = baseLMNode.GetNthControlPointPosition(i)
      targetPoints.InsertNextPoint(point)
    # Transform each subject to base
    subjectFileNames = [f for f in os.listdir(meshDirectory) if not f.startswith(".")]
    subjectTotal = len(subjectFileNames)
    # Index the landmark directory once instead of re-listing it per subject
    # (avoids O(N^2) directory scans on large datasets / network storage).
    landmarkFileIndex = self.buildLandmarkFileIndex(lmDirectory)
    # Pause view rendering for the whole batch. Otherwise each per-subject
    # progressCallback -> slicer.app.processEvents() (and the model loader itself)
    # renders the freshly loaded mesh; that render is slow -- especially with
    # software OpenGL over a remote display -- and dominates alignment time. The
    # progress bar still updates because it is a Qt widget, not a rendered view.
    slicer.app.pauseRender()
    try:
      for subjectCount, meshFileName in enumerate(subjectFileNames, start=1):
        if progressCallback:
          progressCallback(subjectCount, subjectTotal, "Rigid alignment")
        if(not meshFileName.startswith(".")):
          meshFilePath = os.path.join(meshDirectory, meshFileName)
          subjectID = os.path.splitext(meshFileName)[0]
          currentLMNode = self.getLandmarkFileByID(lmDirectory, subjectID, landmarkFileIndex)
          if currentLMNode :
            try:
              currentMeshNode = slicer.util.loadModel(meshFilePath)
            except:
              self._removeNodeFully(currentLMNode)
              continue
            if currentLMNode.GetNumberOfControlPoints() != baseLMNode.GetNumberOfControlPoints():
              raise ValueError(f"Landmark points mismatch: subject has {currentLMNode.GetNumberOfControlPoints()} points, "
                f"atlas has {baseLMNode.GetNumberOfControlPoints()} points")
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
            # persist the (linear) alignment transform so downstream output can be
            # mapped back to this subject's original coordinate frame by inverting it.
            # The transform is keyed by the aligned output basename (subjectID + '_align')
            # so it matches the DeCAL output landmark filenames. See runBackTransformLandmarks.
            if transformDirectory:
              transform.Update()
              alignMatrix = vtk.vtkMatrix4x4()
              alignMatrix.DeepCopy(transform.GetMatrix())
              alignXfNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLinearTransformNode", subjectID + "_alignTransform")
              alignXfNode.SetMatrixTransformToParent(alignMatrix)
              transformPath = os.path.join(transformDirectory, subjectID + "_align.h5")
              slicer.util.saveNode(alignXfNode, transformPath)
              self._removeNodeFully(alignXfNode)
            # optional semi-landmark alignment
            if semilandmarkOption :
              currentSLMNode = self.getLandmarkFileByID(slmDirectory, subjectID)
              if currentSLMNode :
                currentSLMNode.SetAndObserveTransformNodeID(transformNode.GetID())
                slicer.vtkSlicerTransformLogic().hardenTransform(currentSLMNode)
                outputSLMName = subjectID + '_align.mrk.json'
                outputSLMPath = os.path.join(outputSLMDirectory, outputSLMName)
                slicer.util.saveNode(currentSLMNode, outputSLMPath)
                self._removeNodeFully(currentSLMNode)
            # clean up: remove the displayable nodes together with the display and
            # storage nodes that loadModel/loadMarkups create, which otherwise
            # orphan and accumulate across subjects.
            try:
              self._removeNodeFully(currentLMNode)
              self._removeNodeFully(currentMeshNode)
              self._removeNodeFully(transformNode)
            except:
              print(f"could not find nodes to remove for {subjectID}")
    finally:
      slicer.app.resumeRender()

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
    averageLMNode = self.numpyToFiducialNode(averagePoints_np, "Atlas Landmarks")
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

  def importLandmarks(self, topDir, progressCallback=None):
    fiducialGroup = vtk.vtkMultiBlockDataGroupFilter()
    fileNameList = []
    landmarkFiles = [f for f in sorted(os.listdir(topDir)) if f.endswith(".fcsv") or f.endswith(".json")]
    fileTotal = len(landmarkFiles)
    for fileCount, file in enumerate(landmarkFiles, start=1):
      if progressCallback:
        progressCallback(fileCount, fileTotal, "Loading landmarks")
      fileNameList.append(file)
      inputFilePath = os.path.join(topDir, file)
      # may want to replace with vtk reader
      polydataPoints = self.fiducialNodeToPolyData(inputFilePath)
      fiducialGroup.AddInputData(polydataPoints)
    fiducialGroup.Update()
    return fileNameList, fiducialGroup.GetOutput()

  def importMeshes(self, topDir, extensions, progressCallback=None):
      modelGroup = vtk.vtkMultiBlockDataGroupFilter()
      fileNameList = []
      meshFiles = [f for f in sorted(os.listdir(topDir)) if f.endswith(tuple(extensions))]
      fileTotal = len(meshFiles)
      for fileCount, file in enumerate(meshFiles, start=1):
        if progressCallback:
          progressCallback(fileCount, fileTotal, "Loading meshes")
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
    if sampleNumber == 0:
      raise ValueError("No landmark data available for Procrustes analysis")
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
    min_index, min_value = min(enumerate(procrustesDistances), key=operator.itemgetter(1))
    return min_index

  def getClosestToMeanPath(self, landmarkDirectory):
    lmNames, landmarks = self.importLandmarks(landmarkDirectory)
    if not lmNames:
      raise ValueError(f"No landmark files found in directory: {landmarkDirectory}")
    meanShape, alignedLandmarks = self.procrustesImposition(landmarks, False)
    closestToMeanIndex = self.getClosestToMeanIndex(meanShape, alignedLandmarks)
    if closestToMeanIndex >= len(lmNames):
      raise ValueError(f"Index mismatch: computed index {closestToMeanIndex} but only {len(lmNames)} landmarks available")
    return lmNames[closestToMeanIndex]

  def denseCorrespondence(self, originalLandmarks, originalMeshes, writeErrorOption=False, progressCallback=None):
    meanShape, alignedPoints = self.procrustesImposition(originalLandmarks, False)
    sampleNumber = alignedPoints.GetNumberOfBlocks()
    denseCorrespondenceGroup = vtk.vtkMultiBlockDataGroupFilter()
    # get base mesh as the closest to the mean shape
    baseIndex = self.getClosestToMeanIndex(meanShape, alignedPoints)
    baseMesh = originalMeshes.GetBlock(baseIndex)
    baseLandmarks = originalLandmarks.GetBlock(baseIndex).GetPoints()
    # The base mesh warped onto the mean shape is identical for every sample, so
    # compute it once here instead of re-running the TPS solve + warp per sample.
    meanWarpedBase = self._warpBaseMesh(baseMesh, baseLandmarks, meanShape)
    for i in range(sampleNumber):
      if progressCallback:
        progressCallback(i + 1, sampleNumber, "Computing dense correspondence")
      correspondingMesh = self.denseSurfaceCorrespondencePair(originalMeshes.GetBlock(i),
      originalLandmarks.GetBlock(i).GetPoints(), meanWarpedBase, meanShape, i)
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

  def denseCorrespondenceBaseMesh(self, originalLandmarks, originalMeshes, baseMesh, baseLandmarks, progressCallback=None):
    meanShape, alignedPoints = self.procrustesImposition(originalLandmarks, False)
    sampleNumber = alignedPoints.GetNumberOfBlocks()
    print("procrustes aligned samples: ", sampleNumber)
    denseCorrespondenceGroup = vtk.vtkMultiBlockDataGroupFilter()
    # The base mesh warped onto the mean shape is identical for every sample, so
    # compute it once here instead of re-running the TPS solve + warp per sample.
    meanWarpedBase = self._warpBaseMesh(baseMesh, baseLandmarks, meanShape)
    for i in range(sampleNumber):
      if progressCallback:
        progressCallback(i + 1, sampleNumber, "Computing dense correspondence")
      correspondingMesh = self.denseSurfaceCorrespondencePair(originalMeshes.GetBlock(i),
      originalLandmarks.GetBlock(i).GetPoints(), meanWarpedBase, meanShape, i)
      denseCorrespondenceGroup.AddInputData(correspondingMesh)
    denseCorrespondenceGroup.Update()
    return denseCorrespondenceGroup.GetOutput()

  def _warpBaseMesh(self, baseMesh, baseLandmarks, meanShape):
    # TPS-warp the base mesh onto the mean shape. This depends only on the base
    # mesh / base landmarks / mean shape, which are all fixed across the per-sample
    # correspondence loop, so callers compute it once and pass the result into
    # denseSurfaceCorrespondencePair rather than recomputing it for every sample.
    meanTransformBase = vtk.vtkThinPlateSplineTransform()
    meanTransformBase.SetSourceLandmarks(baseLandmarks)
    meanTransformBase.SetTargetLandmarks(meanShape)
    meanTransformBase.SetBasisToR() # for 3D transform

    meanTransformBaseFilter = vtk.vtkTransformPolyDataFilter()
    meanTransformBaseFilter.SetInputData(baseMesh)
    meanTransformBaseFilter.SetTransform(meanTransformBase)
    meanTransformBaseFilter.Update()
    return meanTransformBaseFilter.GetOutput()

  def _closestPointsToMesh(self, queryPoints, targetMesh, useFast=False):
    # For each point in queryPoints (vtkPoints), return the corresponding point
    # relative to targetMesh (vtkPolyData) as a new vtkPoints, index-aligned.
    #
    # Default (exact): closest point on the target *surface* via vtkCellLocator,
    # looping per point in Python.
    # Fast (useFast=True): nearest target *vertex* via a single vectorized scipy
    # cKDTree query. Much faster but approximate; see issue #15. If scipy cannot be
    # imported or installed, this logs a warning and falls back to the exact path.
    if useFast:
      cKDTree = None
      try:
        from scipy.spatial import cKDTree
      except ImportError:
        try:
          slicer.util.pip_install('scipy')
          from scipy.spatial import cKDTree
        except Exception:
          logging.warning("Fast correspondence requires scipy, which could not be "
                          "imported or installed; falling back to the exact method.")
      if cKDTree is not None:
        targetXYZ = vtk_np.vtk_to_numpy(targetMesh.GetPoints().GetData())
        queryXYZ = vtk_np.vtk_to_numpy(queryPoints.GetData())
        tree = cKDTree(targetXYZ)
        try:
          _, matchedIndices = tree.query(queryXYZ, k=1, workers=-1)
        except TypeError:  # older scipy without the workers kwarg
          _, matchedIndices = tree.query(queryXYZ, k=1)
        matchedXYZ = np.ascontiguousarray(targetXYZ[matchedIndices])
        correspondingPoints = vtk.vtkPoints()
        correspondingPoints.SetData(vtk_np.numpy_to_vtk(matchedXYZ, deep=True))
        return correspondingPoints

    cellLocator = vtk.vtkCellLocator()
    cellLocator.SetDataSet(targetMesh)
    cellLocator.BuildLocator()
    point = [0,0,0]
    correspondingPoint = [0,0,0]
    correspondingPoints = vtk.vtkPoints()
    cellId = vtk.reference(0)
    subId = vtk.reference(0)
    distance = vtk.reference(0.0)
    for i in range(queryPoints.GetNumberOfPoints()):
      queryPoints.GetPoint(i, point)
      cellLocator.FindClosestPoint(point, correspondingPoint, cellId, subId, distance)
      correspondingPoints.InsertPoint(i, correspondingPoint)
    return correspondingPoints

  def denseSurfaceCorrespondencePair(self, originalMesh, originalLandmarks, meanWarpedBase, meanShape, iteration, useFast=False):
    # TPS warp target mesh to meanshape. meanWarpedBase (the base mesh already
    # warped onto the mean shape) is supplied by the caller, computed once via
    # _warpBaseMesh since it is identical for every sample.
    meanTransform = vtk.vtkThinPlateSplineTransform()
    meanTransform.SetSourceLandmarks(originalLandmarks)
    meanTransform.SetTargetLandmarks(meanShape)
    meanTransform.SetBasisToR() # for 3D transform

    meanTransformFilter = vtk.vtkTransformPolyDataFilter()
    meanTransformFilter.SetInputData(originalMesh)
    meanTransformFilter.SetTransform(meanTransform)
    meanTransformFilter.Update()
    meanWarpedMesh = meanTransformFilter.GetOutput()

    # write ouput
    if hasattr(self,"errorCheckPath"):
      plyWriterSubject = vtk.vtkPLYWriter()
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
    correspondingPoints = self._closestPointsToMesh(meanWarpedBase.GetPoints(), meanWarpedMesh, useFast=useFast)

    #Copy points into mesh with base connectivity
    correspondingMesh = vtk.vtkPolyData()
    correspondingMesh.SetPoints(correspondingPoints)
    correspondingMesh.SetPolys(meanWarpedBase.GetPolys())

    # Apply inverse warping
    inverseTransform = vtk.vtkThinPlateSplineTransform()
    inverseTransform.SetSourceLandmarks(meanShape)
    inverseTransform.SetTargetLandmarks(originalLandmarks)
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