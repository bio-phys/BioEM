#
#  locate the FFTW library
#

if(NOT DEFINED FFTW_HOME)
   set(FFTW_HOME "$ENV{FFTW_HOME}")
endif()

#The shared object does not exist on hydra
#set(CMAKE_FIND_LIBRARY_SUFFIXES ".so")

find_path(FFTW_INCLUDE_DIR "fftw3.h" HINTS ${FFTW_HOME}/include)
find_library(FFTW_LIBRARY  "fftw3" HINTS ${FFTW_HOME}/lib )
#find_library(FFTW_LIBRARY_MT  "fftw3_omp" HINTS ${FFTW_HOME}/lib )
find_library(FFTW_LIBRARY_F  "fftw3f" HINTS ${FFTW_HOME}/lib )

if(NOT FFTW_LIBRARY OR NOT FFTW_INCLUDE_DIR)
   if(NOT FFTW_HOME)
      message(WARNING "FFTW_HOME undefined - try 'module load fftw' ")
   endif()
endif()

# set default find_package outcome variables
set(FFTWF_LIBRARIES ${FFTW_LIBRARY_F})
set(FFTW_INCLUDE_DIRS ${FFTW_INCLUDE_DIR})

