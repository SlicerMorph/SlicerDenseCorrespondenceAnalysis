cmake_minimum_required(VERSION 3.5)

project(DeCA)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://slicermorph.github.io/")
set(EXTENSION_CATEGORY "DeCA Toolbox")
set(EXTENSION_CONTRIBUTORS "Sara Rolfe (SCRI), Murat Maga (UW, SCRI)")
set(EXTENSION_DESCRIPTION "This extension provides several flexible workflows and utilities to support finding and analyzing dense correspondence points between models.")
set(EXTENSION_ICONURL "https://github.com/SlicerMorph/SlicerDeCA/raw/main/DeCA.png")
set(EXTENSION_SCREENSHOTURLS "https://na-mic.github.io/ProjectWeek/PW30_2019_GranCanaria/Projects/SlicerMorphGeometricMorphometricToolset/SM_screen.png https://na-mic.github.io/ProjectWeek/PW30_2019_GranCanaria/Projects/SlicerMorphGeometricMorphometricToolset/SM2.png")
set(EXTENSION_DEPENDS "SlicerMorph") # Specified as a space separated string, a list or 'NA' if any

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
