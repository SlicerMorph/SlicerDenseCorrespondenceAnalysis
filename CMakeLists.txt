cmake_minimum_required(VERSION 3.5)

project(DenseCorrespondenceAnalysis)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://slicermorph.github.io/")
set(EXTENSION_CATEGORY "DeCA")
set(EXTENSION_CONTRIBUTORS "Sara Rolfe (SCRI), Murat Maga (UW, SCRI)")
set(EXTENSION_DESCRIPTION "Dense Correspondence Analysis for Surfaces")
set(EXTENSION_ICONURL "https://github.com/SlicerMorph/SlicerDenseCorrespondenceAnalysis/raw/main/DeCA.png")
set(EXTENSION_SCREENSHOTURLS "https://na-mic.github.io/ProjectWeek/PW30_2019_GranCanaria/Projects/SlicerMorphGeometricMorphometricToolset/SM_screen.png https://na-mic.github.io/ProjectWeek/PW30_2019_GranCanaria/Projects/SlicerMorphGeometricMorphometricToolset/SM2.png")
set(EXTENSION_DEPENDS "NA") # Specified as a space separated string, a list or 'NA' if any

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(DeCA)

## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
