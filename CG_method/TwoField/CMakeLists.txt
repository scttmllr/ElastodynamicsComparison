##
#  CMake script for TwoPhaseDG:
##

CMAKE_MINIMUM_REQUIRED(VERSION 2.8.8)

    #
    # Find and import the deal.II project configuration:
    #
FIND_PACKAGE(deal.II 8.0 REQUIRED HINTS ${DEAL_II_DIR} $ENV{DEAL_II_DIR} )

DEAL_II_INITIALIZE_CACHED_VARIABLES()

PROJECT(CG_two_field)

################################ User Variables ################################                              

#sets the dimension for the case
add_definitions( -D deal_II_dimension=2)

# set debug or release
set(CMAKE_BUILD_TYPE "release")

#############################################################################

#INCLUDE_DIRECTORIES(include/base
#					  include/fluid)

SET(CMAKE_CXX_FLAGS ${DEAL_II_CXX_FLAGS})
SET(CMAKE_CXX_FLAGS_RELEASE ${DEAL_II_CXX_FLAGS_RELEASE})
SET(CMAKE_CXX_FLAGS_DEBUG ${DEAL_II_CXX_FLAGS_DEBUG})

FILE(GLOB CG_two_field.cc)

#ADD_LIBRARY(TwoPhaseDGSources SHARED ${MyCSources})

#DEAL_II_SETUP_TARGET(TwoPhaseDGSources) 

ADD_EXECUTABLE(CG_two_field CG_two_field.cc)

DEAL_II_SETUP_TARGET(CG_two_field) 

include_directories ($BOOST_ROOT)
link_directories($BOOST_LIB)

#TARGET_LINK_LIBRARIES(CG_two_field /InstalledPackages/boost_1_52_0/stage/lib/libboost_chrono.dylib)
##TARGET_LINK_LIBRARIES(CG_two_field boost_chrono)


SET_TARGET_PROPERTIES(CG_two_field PROPERTIES
      LINK_FLAGS
      "${DEAL_II_LINKER_FLAGS}"
      LINK_FLAGS_DEBUG
      "${DEAL_II_LINKER_FLAGS_DEBUG}"
      LINK_FLAGS_RELEASE
      "${DEAL_II_LINKER_FLAGS_RELEASE}"
      )
