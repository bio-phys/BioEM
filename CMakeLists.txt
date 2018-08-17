cmake_minimum_required(VERSION 2.6)

project(BioEM)

###use g++ and gcc instead of c++ and cc

##this works but would change the default cmake toolchain precedence c++,g++,...
#if (NOT (CMAKE_CXX_COMPILER OR $ENV{CXX}))
#    set (CMAKE_CXX_COMPILER g++)
#endif()

##this works but would change the default cmake toolchain precedence cc,gcc,...
#if (NOT (CMAKE_C_COMPILER OR $ENV{CC}))
#    set (CMAKE_C_COMPILER gcc)
#endif()

###Set up options
option (USE_CUDA "Build BioEM with CUDA support" OFF)
option (USE_OPENMP "Build BioEM with OpenMP support" ON)
option (USE_MPI "Build BioEM with MPI support" ON)
option (PRINT_CMAKE_VARIABLES "List all CMAKE Variables" OFF)
option (CUDA_FORCE_GCC "Force GCC as host compiler for CUDA part (If standard host compiler is incompatible with CUDA)" ON)
option (USE_NVTX "Build BioEM with additional NVTX information" OFF)


###Set up general variables
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}")

include_directories(include)

set (BIOEM_ICC_FLAGS "-O3 -fno-alias -fno-fnalias -unroll -g0 -ip")
set (BIOEM_GCC_FLAGS "-O3 -march=native -fweb -mfpmath=sse -frename-registers -minline-all-stringops -ftracer -funroll-loops -fpeel-loops -fprefetch-loop-arrays -ffast-math -ggdb")

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${BIOEM_ICC_FLAGS}")
else()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${BIOEM_GCC_FLAGS}")
endif()

set (BIOEM_SOURCE_FILES "bioem.cpp" "main.cpp" "map.cpp" "model.cpp" "param.cpp" "timer.cpp" "autotuner.cpp")

###Find Required Packages
find_package(PkgConfig)
if (PKG_CONFIG_FOUND)
    pkg_check_modules(FFTW fftw3)
endif()
if (NOT FFTW_FOUND)
    find_package(FFTW 3 REQUIRED)
endif()
include_directories(${FFTW_INCLUDE_DIRS})

###Find Optional Packages

###Find CUDA
set (BIOEM_CUDA_STATUS "Disabled")
if (USE_CUDA)
        set (BIOEM_CUDA_STATUS "Not Found")
        find_package(CUDA)
endif()

if (CUDA_FOUND)
        if (CUDA_FORCE_GCC)
                cmake_minimum_required(VERSION 2.8.10.1)
                #Use GCC as host compiler for CUDA even though host compiler for other files is not GCC
                #set (CUDA_HOST_COMPILER /mpcdf/soft/SLES122/common/gcc/5.4.0/bin/gcc)
                set (CUDA_HOST_COMPILER gcc)
        endif()
        set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};--use_fast_math;-ftz=true;-O4;-Xptxas -O4")

        if (${CUDA_VERSION} LESS 7.0) 
                list(APPEND CUDA_NVCC_FLAGS "-gencode=arch=compute_13,code=sm_13")
        endif()
        if (${CUDA_VERSION} LESS 8.5) 
                list(APPEND CUDA_NVCC_FLAGS "-gencode=arch=compute_20,code=[sm_20,sm_21]")
	endif()	
        list(APPEND CUDA_NVCC_FLAGS "-gencode=arch=compute_30,code=sm_30")
        list(APPEND CUDA_NVCC_FLAGS "-gencode=arch=compute_35,code=sm_35")
        if (NOT (${CUDA_VERSION} LESS 7.5)) 
                list(APPEND CUDA_NVCC_FLAGS "-gencode=arch=compute_52,code=sm_52")
	endif()
        if (NOT (${CUDA_VERSION} LESS 8.0)) 
		list(APPEND CUDA_NVCC_FLAGS "-gencode=arch=compute_60,code=[sm_60,sm_61]")
	endif()
        if (NOT (${CUDA_VERSION} LESS 9.0))
		list(APPEND CUDA_NVCC_FLAGS "-gencode=arch=compute_70,code=sm_70")
	endif()
        add_definitions(-DWITH_CUDA)
        set (BIOEM_CUDA_STATUS "Found")
endif()

###Find OpenMP
set (BIOEM_OPENMP_STATUS "Disabled")
if (USE_OPENMP)
        set (BIOEM_OPENMP_STATUS "Not Found")
        find_package(OpenMP)
endif()

if(OPENMP_FOUND)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        add_definitions(-DWITH_OPENMP)
        set (BIOEM_OPENMP_STATUS "Found")
endif()

###Find MPI
set (BIOEM_MPI_STATUS "Disabled")
if (USE_MPI)
        set (BIOEM_MPI_STATUS "Not Found")
        find_package(MPI)
endif()

if (MPI_FOUND)
        include_directories(${MPI_INCLUDE_PATH})
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${MPI_COMPILE_FLAGS} ${MPI_LINK_FLAGS}")
        add_definitions(-DWITH_MPI)
        set (BIOEM_MPI_STATUS "Found")
endif()

###Build Executable
if (CUDA_FOUND)
        set(BIOEM_TMP_CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
        if (CUDA_FORCE_GCC)
                #Hack to use GCC flags for GCC host compiler during NVCC compilation, although host compiler is in fact not GCC for other files
                set(CMAKE_CXX_FLAGS ${BIOEM_GCC_FLAGS})
        endif()
        cuda_add_executable(bioEM ${BIOEM_SOURCE_FILES} bioem_cuda.cu)
        set(CMAKE_CXX_FLAGS ${BIOEM_TMP_CMAKE_CXX_FLAGS})
else()
        add_executable(bioEM ${BIOEM_SOURCE_FILES})
endif()

#Additional CXX Flags not used by CUDA compiler
#Additional CXX Flags not used by CUDA compiler
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -pedantic")
else()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-vla -Wno-long-long  -Wall -pedantic")
endif()

###Enable CUDA debugging with NVTX
if (USE_NVTX)
        if (CUDA_FOUND)
                set(CUDA_CUDA_LIBRARY ${CUDA_CUDA_LIBRARY} "${CUDA_TOOLKIT_ROOT_DIR}/lib64/libnvToolsExt.so")
                add_definitions(-DBIOEM_USE_NVTX)
        else()
                message(FATAL_ERROR "Cannot use NVTX if CUDA is not found")
        endif()
endif()

###Add Libraries
if (CUDA_FOUND)
        cuda_add_cufft_to_target(bioEM)
#set(CUDA_LIBRARIES "/afs/ipp/.cs/cuda/6.5-gtx9/amd64_sles11/lib64/stubs/libcuda.so")
#cuda_add_library() 
        target_link_libraries(bioEM ${CUDA_CUDA_LIBRARY})
#	  target_link_libraries(bioEM ${CUDA_LIBRARIES})
endif()

if (FFTWF_LIBRARIES)
        target_link_libraries(bioEM ${FFTWF_LIBRARIES})
else()
        target_link_libraries(bioEM -L${FFTW_LIBDIR} -lfftw3 -lfftw3f)
endif()

if (MPI_FOUND)
        target_link_libraries(bioEM ${MPI_LIBRARIES})
endif()

###Show Status
message(STATUS "Build Status")
message(STATUS "FFTW library: ${FFTW_LIBDIR}")
message(STATUS "FFTW includedir: ${FFTW_INCLUDEDIR}")
message(STATUS "CUDA libraries:  ${CUDA_CUDA_LIBRARY}")
message(STATUS "CUDART libraries:  ${CUDA_LIBRARIES}")
message(STATUS "CUDA: ${BIOEM_CUDA_STATUS}")
message(STATUS "OpenMP: ${BIOEM_OPENMP_STATUS}")
message(STATUS "MPI: ${BIOEM_MPI_STATUS}")


if (PRINT_CMAKE_VARIABLES)
        get_cmake_property(_variableNames VARIABLES)
        foreach (_variableName ${_variableNames})
                message(STATUS "${_variableName}=${${_variableName}}")
        endforeach()
endif()
