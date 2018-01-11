/* ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
   < BioEM software for Bayesian inference of Electron Microscopy images>
   Copyright (C) 2017 Pilar Cossio, David Rohr, Fabio Baruffa, Markus Rampp,
        Luka Stanisic, Volker Lindenstruth and Gerhard Hummer.
   Max Planck Institute of Biophysics, Frankfurt, Germany.
   Frankfurt Institute for Advanced Studies, Goethe University Frankfurt,
   Germany.
   Max Planck Computing and Data Facility, Garching, Germany.

   Released under the GNU Public License, v3.
   See license statement for terms of distribution.

   ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/

#ifdef WITH_MPI
#include <mpi.h>

#define MPI_CHK(expr)                                                          \
  if (expr != MPI_SUCCESS)                                                     \
  {                                                                            \
    fprintf(stderr, "Error in MPI function %s: %d\n", __FILE__, __LINE__);     \
  }
#endif

#include "MersenneTwister.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <getopt.h>
#include <iostream>
#include <iterator>
#include <queue>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <vector>

#ifdef WITH_OPENMP
#include <omp.h>
#endif

#include "autotuner.h"
#include "timer.h"
#include <fftw3.h>
#include <math.h>

#include "bioem.h"
#include "map.h"
#include "model.h"
#include "param.h"

#ifdef BIOEM_USE_NVTX
#include "nvToolsExt.h"

const uint32_t colors[] = {0x0000ff00, 0x000000ff, 0x00ffff00, 0x00ff00ff,
                           0x0000ffff, 0x00ff0000, 0x00ffffff};
const int num_colors = sizeof(colors) / sizeof(colors[0]);
enum myColor
{
  COLOR_PROJECTION,
  COLOR_CONVOLUTION,
  COLOR_COMPARISON,
  COLOR_WORKLOAD,
  COLOR_INIT
};

// Projection number is stored in category attribute
// Convolution number is stored in payload attribute

#define cuda_custom_timeslot(name, iMap, iConv, cid)                           \
  {                                                                            \
    int color_id = cid;                                                        \
    color_id = color_id % num_colors;                                          \
    nvtxEventAttributes_t eventAttrib = {0};                                   \
    eventAttrib.version = NVTX_VERSION;                                        \
    eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;                          \
    eventAttrib.colorType = NVTX_COLOR_ARGB;                                   \
    eventAttrib.color = colors[color_id];                                      \
    eventAttrib.category = iMap;                                               \
    eventAttrib.payloadType = NVTX_PAYLOAD_TYPE_UNSIGNED_INT64;                \
    eventAttrib.payload.llValue = iConv;                                       \
    eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;                         \
    eventAttrib.message.ascii = name;                                          \
    nvtxRangePushEx(&eventAttrib);                                             \
  }
#define cuda_custom_timeslot_end nvtxRangePop();
#else
#define cuda_custom_timeslot(name, iMap, iConv, cid)
#define cuda_custom_timeslot_end
#endif

#include "bioem_algorithm.h"

using namespace std;

bioem::bioem()
{
  BioEMAlgo = getenv("BIOEM_ALGO") == NULL ? 1 : atoi(getenv("BIOEM_ALGO"));

  DebugOutput = getenv("BIOEM_DEBUG_OUTPUT") == NULL ?
                    0 :
                    atoi(getenv("BIOEM_DEBUG_OUTPUT"));

  if (getenv("BIOEM_PROJ_CONV_AT_ONCE") != NULL)
  {
    nProjectionsAtOnce = atoi(getenv("BIOEM_PROJ_CONV_AT_ONCE"));
    if (BioEMAlgo == 1 && getenv("GPU") && atoi(getenv("GPU")) &&
        nProjectionsAtOnce > 1)
    {
      printf("Warning: using parallel convolutions with GPUs can create race "
             "condition and lead to inaccurate results. "
             "BIOEM_PROJ_CONV_AT_ONCE is going to be set 1.\n");
      nProjectionsAtOnce = 1;
    }
  }
  else if (BioEMAlgo == 1)
    nProjectionsAtOnce = 1;
  else
    nProjectionsAtOnce =
        getenv("OMP_NUM_THREADS") == NULL ? 1 : atoi(getenv("OMP_NUM_THREADS"));

  if (getenv("BIOEM_CUDA_THREAD_COUNT") != NULL)
    CudaThreadCount = atoi(getenv("BIOEM_CUDA_THREAD_COUNT"));
  else if (BioEMAlgo == 1)
    CudaThreadCount = CUDA_THREAD_COUNT_ALGO1;
  else
    CudaThreadCount = CUDA_THREAD_COUNT_ALGO2;

  Autotuning = false;
}

bioem::~bioem() {}

void bioem::printOptions(myoption_t *myoptions, int myoptions_length)
{
  printf("\nCommand line inputs:\n");

  // Find longest column width
  int maxlen = 0;
  for (int i = 0; i < myoptions_length; i++)
  {
    if (myoptions[i].hidden)
      continue;
    if (maxlen < strlen(myoptions[i].name))
      maxlen = strlen(myoptions[i].name);
  }

  for (int i = 0; i < myoptions_length; i++)
  {
    if (myoptions[i].hidden)
      continue;
    printf("  --%-*s", maxlen, myoptions[i].name);
    if (myoptions[i].arg == required_argument)
      printf(" arg");
    else
      printf("    ");
    printf(" %s\n", myoptions[i].desc);
  }
  printf("\n");
}

int bioem::readOptions(int ac, char *av[])
{
  HighResTimer timer;

  // *** Inizialzing default variables ***
  std::string infile, modelfile, mapfile, Inputanglefile, Inputbestmap;
  Model.readPDB = false;
  param.param_device.writeAngles = 0;
  param.dumpMap = false;
  param.loadMap = false;
  param.printModel = false;
  RefMap.readMRC = false;
  RefMap.readMultMRC = false;
  param.notuniformangles = false;
  OutfileName = "Output_Probabilities";

  cout << " ++++++++++++ FROM COMMAND LINE +++++++++++\n\n";

  // Write your options here
  myoption_t myoptions[] = {
      {"Modelfile", required_argument, "(Mandatory) Name of model file", false},
      {"Particlesfile", required_argument,
       "(Mandatory) Name of particle-image file", false},
      {"Inputfile", required_argument,
       "(Mandatory) Name of input parameter file", false},
      {"PrintBestCalMap", required_argument,
       "(Optional) Only print best calculated map. NO BioEM!", true},
      {"ReadOrientation", required_argument,
       "(Optional) Read file name containing orientations", false},
      {"ReadPDB", no_argument, "(Optional) If reading model file in PDB format",
       false},
      {"ReadMRC", no_argument,
       "(Optional) If reading particle file in MRC format", false},
      {"ReadMultipleMRC", no_argument, "(Optional) If reading Multiple MRCs",
       false},
      {"DumpMaps", no_argument,
       "(Optional) Dump maps after they were read from particle-image file",
       false},
      {"LoadMapDump", no_argument, "(Optional) Read Maps from dump option",
       false},
      {"OutputFile", required_argument,
       "(Optional) For changing the outputfile name", false},
      {"help", no_argument, "(Optional) Produce help message", false}};
  int myoptions_length = sizeof(myoptions) / sizeof(myoption_t);

  // If not all Mandatory parameters are defined
  if ((ac < 2))
  {
    printf("Error: Need to specify all mandatory options\n");
    printOptions(myoptions, myoptions_length);
    return 1;
  }

  // Creating options structure for getopt_long()
  struct option *long_options =
      (option *) calloc((myoptions_length + 1), sizeof(option));
  for (int i = 0; i < myoptions_length; i++)
  {
    long_options[i].name = myoptions[i].name;
    long_options[i].has_arg = myoptions[i].arg;
  }

  int myopt;
  while (1)
  {
    /* getopt_long stores the option index here. */
    int option_index = 0;
    myopt = getopt_long(ac, av, "", long_options, &option_index);

    /* Detect the end of the options. */
    if (myopt == -1)
      break;

    switch (myopt)
    {
      case 0:
#ifdef DEBUG
        printf("option %s", long_options[option_index].name);
        if (optarg)
          printf(" with arg %s", optarg);
        printf("\n");
#endif
        // Here write actions for each option
        if (!strcmp(long_options[option_index].name, "help"))
        {
          cout << "Usage: options_description [options]\n";
          printOptions(myoptions, myoptions_length);
          return 1;
        }
        if (!strcmp(long_options[option_index].name, "Inputfile"))
        {
          cout << "Input file is: " << optarg << "\n";
          infile = optarg;
        }
        if (!strcmp(long_options[option_index].name, "Modelfile"))
        {
          cout << "Model file is: " << optarg << "\n";
          modelfile = optarg;
        }
        if (!strcmp(long_options[option_index].name, "ReadPDB"))
        {
          cout << "Reading model file in PDB format.\n";
          Model.readPDB = true;
        }
        if (!strcmp(long_options[option_index].name, "ReadOrientation"))
        {
          cout << "Reading Orientation from file: " << optarg << "\n";
          cout << "Important! if using Quaternions, include \n";
          cout << "QUATERNIONS keyword in INPUT PARAMETER FILE\n";
          cout << "First row in file should be the total number of "
                  "orientations "
                  "(int)\n";
          cout << "Euler angle format should be alpha (12.6f) beta (12.6f) "
                  "gamma (12.6f)\n";
          cout << "Quaternion format q1 (12.6f) q2 (12.6f) q3 (12.6f) q4 "
                  "(12.6f)\n";
          Inputanglefile = optarg;
          param.notuniformangles = true;
        }
        if (!strcmp(long_options[option_index].name, "OutputFile"))
        {
          cout << "Writing OUTPUT to: " << optarg << "\n";
          OutfileName = optarg;
        }
        if (!strcmp(long_options[option_index].name, "PrintBestCalMap"))
        {
          cout << "Reading Best Parameters from file: " << optarg << "\n";
          Inputbestmap = optarg;
          param.printModel = true;
        }
        if (!strcmp(long_options[option_index].name, "ReadMRC"))
        {
          cout << "Reading particle file in MRC format.\n";
          RefMap.readMRC = true;
        }
        if (!strcmp(long_options[option_index].name, "ReadMultipleMRC"))
        {
          cout << "Reading Multiple MRCs.\n";
          RefMap.readMultMRC = true;
        }
        if (!strcmp(long_options[option_index].name, "DumpMaps"))
        {
          cout << "Dumping Maps after reading from file.\n";
          param.dumpMap = true;
        }
        if (!strcmp(long_options[option_index].name, "LoadMapDump"))
        {
          cout << "Loading Map dump.\n";
          param.loadMap = true;
        }
        if (!strcmp(long_options[option_index].name, "Particlesfile"))
        {
          cout << "Particle file is: " << optarg << "\n";
          mapfile = optarg;
        }
        break;
      case '?':
        /* getopt_long already printed an error message. */
        printOptions(myoptions, myoptions_length);
        return 1;
      default:
        abort();
    }
  }
  /* Print any remaining command line arguments (not options) and exit */
  if (optind < ac)
  {
    printf("Error: non-option ARGV-elements: ");
    while (optind < ac)
      printf("%s ", av[optind++]);
    putchar('\n');
    printOptions(myoptions, myoptions_length);
    return 1;
  }

  // check for consitency in multiple MRCs
  if (RefMap.readMultMRC && not(RefMap.readMRC))
  {
    cout << "For Multiple MRCs command --ReadMRC is necesary too";
    exit(1);
  }

  if (!Model.readPDB)
  {
    cout << "Note: Reading model in simple text format (not PDB)\n";
    cout << "----  x   y   z  radius  density ------- \n";
  }

  if (DebugOutput >= 2 && mpi_rank == 0)
    timer.ResetStart();

  // *** Reading Parameter Input ***
  if (!param.printModel)
  {
    // Standard definition for BioEM
    param.readParameters(infile.c_str());
    // *** Reading Particle Maps Input ***
    RefMap.readRefMaps(param, mapfile.c_str());
  }
  else
  {
    // Reading parameters for only writting down Best projection
    param.forprintBest(Inputbestmap.c_str());
  }

  // *** Reading Model Input ***
  Model.readModel(param, modelfile.c_str());

  cout << "**NOTE:: look at file COORDREAD to confirm that the Model "
          "coordinates are correct\n";

  if (DebugOutput >= 2 && mpi_rank == 0)
    printf("Reading Input Data Time: %f\n", timer.GetCurrentElapsedTime());

  // Generating Grids of orientations
  if (!param.printModel)
    param.CalculateGridsParam(Inputanglefile.c_str());

  return (0);
}

int bioem::configure(int ac, char *av[])
{
  // **************************************************************************************
  // **** Configuration Routine using getopts for extracting parameters, models
  // and maps ****
  // **************************************************************************************
  // ****** And Precalculating necessary grids, map crosscorrelations and
  // kernels  ********
  // *************************************************************************************

  HighResTimer timer;

  if (mpi_rank == 0 && readOptions(ac, av))
    return 1;

#ifdef WITH_MPI

  // ********************* MPI inizialization/ Transfer of
  // parameters******************
  if (mpi_size > 1)
  {
    if (DebugOutput >= 2 && mpi_rank == 0)
      timer.ResetStart();
    MPI_Bcast(&param, sizeof(param), MPI_BYTE, 0, MPI_COMM_WORLD);
    // We have to reinitialize all pointers !!!!!!!!!!!!
    if (mpi_rank != 0)
      param.angprior = NULL;

    if (mpi_rank != 0)
      param.angles =
          (myfloat3_t *) mallocchk(param.nTotGridAngles * sizeof(myfloat3_t));
    MPI_Bcast(param.angles, param.nTotGridAngles * sizeof(myfloat3_t), MPI_BYTE,
              0, MPI_COMM_WORLD);

#ifdef DEBUG
    for (int n = 0; n < param.nTotGridAngles; n++)
    {
      cout << "CHECK: Angle orient " << mpi_rank << " " << n << " "
           << param.angles[n].pos[0] << " " << param.angles[n].pos[1] << " "
           << param.angles[n].pos[2] << " " << param.angles[n].quat4 << " "
           << "\n";
    }

#endif
    //****refCtf, CtfParam, angles automatically filled by precalculate function
    // bellow

    MPI_Bcast(&Model, sizeof(Model), MPI_BYTE, 0, MPI_COMM_WORLD);
    if (mpi_rank != 0)
      Model.points = (bioem_model::bioem_model_point *) mallocchk(
          sizeof(bioem_model::bioem_model_point) * Model.nPointsModel);
    MPI_Bcast(Model.points,
              sizeof(bioem_model::bioem_model_point) * Model.nPointsModel,
              MPI_BYTE, 0, MPI_COMM_WORLD);

    MPI_Bcast(&RefMap, sizeof(RefMap), MPI_BYTE, 0, MPI_COMM_WORLD);
    if (mpi_rank != 0)
      RefMap.maps = (myfloat_t *) mallocchk(
          RefMap.refMapSize * sizeof(myfloat_t) * RefMap.ntotRefMap);
    MPI_Bcast(RefMap.maps,
              RefMap.refMapSize * sizeof(myfloat_t) * RefMap.ntotRefMap,
              MPI_BYTE, 0, MPI_COMM_WORLD);
    if (DebugOutput >= 2 && mpi_rank == 0)
      printf("MPI Broadcast of Input Data %f\n", timer.GetCurrentElapsedTime());
  }
#endif

  // ****************** Precalculating Necessary Stuff *********************
  if (DebugOutput >= 2 && mpi_rank == 0)
    timer.ResetStart();
  param.PrepareFFTs();

  if (DebugOutput >= 2 && mpi_rank == 0)
  {
    printf("Time Prepare FFTs %f\n", timer.GetCurrentElapsedTime());
    timer.ResetStart();
  }
  precalculate();

  // ****************** For debugging *********************
  if (getenv("BIOEM_DEBUG_BREAK"))
  {
    const int cut = atoi(getenv("BIOEM_DEBUG_BREAK"));
    if (param.nTotGridAngles > cut)
      param.nTotGridAngles = cut;
    if (param.nTotCTFs > cut)
      param.nTotCTFs = cut;
  }

  if (DebugOutput >= 2 && mpi_rank == 0)
  {
    printf("Time Precalculate %f\n", timer.GetCurrentElapsedTime());
    timer.ResetStart();
  }

  // Number of parallel Convolutions and Comparisons
  param.nTotParallelConv = min(param.nTotCTFs, nProjectionsAtOnce);

  // ****************** For autotuning **********************
  if ((getenv("GPU") && atoi(getenv("GPU"))) && (BioEMAlgo == 1) &&
      ((!getenv("GPUWORKLOAD") || (atoi(getenv("GPUWORKLOAD")) == -1))) &&
      (!getenv("BIOEM_DEBUG_BREAK") ||
       (atoi(getenv("BIOEM_DEBUG_BREAK")) > FIRST_STABLE)))
  {
    Autotuning = true;
    if (mpi_rank == 0)
      printf("Autotuning of GPUWorkload enabled:\n\tAlgorithm "
             "%d\n\tRecalibration at every %d projections\n\tComparisons are "
             "considered stable after first %d comparisons\n",
             AUTOTUNING_ALGORITHM, RECALIB_FACTOR, FIRST_STABLE);
  }
  else
  {
    Autotuning = false;
    if (mpi_rank == 0)
    {
      printf("Autotuning of GPUWorkload disabled");
      if (getenv("GPU") && atoi(getenv("GPU")))
        printf(", using GPUWorkload: %d%%\n",
               (getenv("GPUWORKLOAD") && (atoi(getenv("GPUWORKLOAD")) != -1)) ?
                   atoi(getenv("GPUWORKLOAD")) :
                   100);
      else
        printf(", please enable GPUs\n");
    }
  }

  // ****************** Initializing pointers *********************

  deviceInit();

  if (DebugOutput >= 2 && mpi_rank == 0)
  {
    printf("Time Device Init %f\n", timer.GetCurrentElapsedTime());
    timer.ResetStart();
  }

  if (!param.printModel)
    pProb.init(RefMap.ntotRefMap, param.nTotGridAngles, *this);

  if (DebugOutput >= 2 && mpi_rank == 0)
  {
    printf("Time Init Probabilities %f\n", timer.GetCurrentElapsedTime());
    timer.ResetStart();
  }

  return (0);
}

void bioem::cleanup()
{
  // Deleting allocated pointers
  free_device_host(pProb.ptr);
  RefMap.freePointers();
}

int bioem::precalculate()
{
  // **************************************************************************************
  // **Precalculating Routine of Orientation grids, Map crosscorrelations and
  // CTF Kernels**
  // **************************************************************************************
  HighResTimer timer;
  if (DebugOutput >= 2)
  {
    printf("\tTime Precalculate Grids Param: %f\n",
           timer.GetCurrentElapsedTime());
    timer.ResetStart();
  }
  // Precalculating CTF Kernels stored in class Param
  param.CalculateRefCTF();

  if (DebugOutput >= 2)
  {
    printf("\tTime Precalculate CTFs: %f\n", timer.GetCurrentElapsedTime());
    timer.ResetStart();
  }
  // Precalculate Maps
  if (!param.printModel)
    RefMap.precalculate(param, *this);
  if (DebugOutput >= 2)
    printf("\tTime Precalculate Maps: %f\n", timer.GetCurrentElapsedTime());

  return (0);
}

int bioem::printModel()
{
  // **************************************************************************************
  // ********** Secondary routine for printing out the only best projection
  // ***************
  // **************************************************************************************

  cout << "\nAnalysis for printing best projection::: \n \n";
  mycomplex_t *proj_mapsFFT;
  myfloat_t *conv_map = NULL;
  mycomplex_t *conv_mapFFT;
  myfloat_t sumCONV, sumsquareCONV;

  proj_mapsFFT = (mycomplex_t *) myfftw_malloc(
      sizeof(mycomplex_t) * param.param_device.NumberPixels *
      param.param_device.NumberFFTPixels1D);
  conv_mapFFT = (mycomplex_t *) myfftw_malloc(
      sizeof(mycomplex_t) * param.param_device.NumberPixels *
      param.param_device.NumberFFTPixels1D);
  conv_map = (myfloat_t *) myfftw_malloc(sizeof(myfloat_t) *
                                         param.param_device.NumberPixels *
                                         param.param_device.NumberPixels);

  cout << "...... Calculating Projection .......................\n ";

  createProjection(0, proj_mapsFFT);

  cout << "...... Calculating Convolution .......................\n ";

  createConvolutedProjectionMap_noFFT(proj_mapsFFT, conv_map, conv_mapFFT,
                                      sumCONV, sumsquareCONV);

  return (0);
}

int bioem::run()
{
  // **************************************************************************************
  // **** Main BioEM routine, projects, convolutes and compares with Map using
  // OpenMP ****
  // **************************************************************************************

  // **** If we want to control the number of threads ->
  // omp_set_num_threads(XX); ******
  // ****************** Declarying class of Probability Pointer
  // *************************
  cuda_custom_timeslot("Initialization", -1, -1, COLOR_INIT);
  if (mpi_rank == 0)
    printf("\tInitializing Probabilities\n");

  // Contros for MPI
  if (mpi_size > param.nTotGridAngles)
  {
    cout << "EXIT: Wrong MPI setup More MPI processes than orientations\n";
    exit(1);
  }

  // Inizialzing Probabilites to zero and constant to -Infinity
  for (int iRefMap = 0; iRefMap < RefMap.ntotRefMap; iRefMap++)
  {
    bioem_Probability_map &pProbMap = pProb.getProbMap(iRefMap);

    pProbMap.Total = 0.0;
    pProbMap.Constoadd = MIN_PROB;

    if (param.param_device.writeAngles)
    {
      for (int iOrient = 0; iOrient < param.nTotGridAngles; iOrient++)
      {
        bioem_Probability_angle &pProbAngle =
            pProb.getProbAngle(iRefMap, iOrient);

        pProbAngle.forAngles = 0.0;
        pProbAngle.ConstAngle = MIN_PROB;
      }
    }
  }

  // **************************************************************************************

  deviceStartRun();

  // ******************************** MAIN CYCLE
  // ******************************************

  mycomplex_t *proj_mapsFFT;
  mycomplex_t *conv_mapsFFT;
  myparam5_t *comp_params =
      new myparam5_t[param.nTotParallelConv * PIPELINE_LVL];
  int iPipeline = 0;

  // allocating fftw_complex vector
  const int ProjMapSize =
      (param.FFTMapSize + 64) & ~63; // Make sure this is properly aligned for
  // fftw..., Actually this should be ensureb by
  // using FFTMapSize, but it is not due to a bug
  // in CUFFT which cannot handle padding properly
  //******** Allocating Vectors *************
  proj_mapsFFT = (mycomplex_t *) myfftw_malloc(
      sizeof(mycomplex_t) * ProjMapSize * nProjectionsAtOnce);
  conv_mapsFFT =
      (mycomplex_t *) myfftw_malloc(sizeof(mycomplex_t) * param.FFTMapSize *
                                    param.nTotParallelConv * PIPELINE_LVL);

  cuda_custom_timeslot_end; // Ending initialization

  HighResTimer timer, timer2;

  /* Autotuning */
  Autotuner aut;
  if (Autotuning)
  {
    aut.Initialize(AUTOTUNING_ALGORITHM, FIRST_STABLE);
    rebalanceWrapper(aut.Workload());
  }

  if (DebugOutput >= 1 && mpi_rank == 0)
    printf("\tMain Loop GridAngles %d, CTFs %d, RefMaps %d, Shifts (%d/%d)², "
           "Pixels %d², OMP Threads %d, MPI Ranks %d\n",
           param.nTotGridAngles, param.nTotCTFs, RefMap.ntotRefMap,
           2 * param.param_device.maxDisplaceCenter +
               param.param_device.GridSpaceCenter,
           param.param_device.GridSpaceCenter, param.param_device.NumberPixels,
           omp_get_max_threads(), mpi_size);

  const int iOrientStart =
      (int) ((long long int) mpi_rank * param.nTotGridAngles / mpi_size);
  int iOrientEnd =
      (int) ((long long int) (mpi_rank + 1) * param.nTotGridAngles / mpi_size);
  if (iOrientEnd > param.nTotGridAngles)
    iOrientEnd = param.nTotGridAngles;

  /* Vectors for computing statistic on different parts of the code */
  TimeStat ts((iOrientEnd - iOrientStart), param.nTotCTFs);
  if (DebugOutput >= 1)
    ts.InitTimeStat(4);

  // **************************Loop Over
  // orientations***************************************

  for (int iOrientAtOnce = iOrientStart; iOrientAtOnce < iOrientEnd;
       iOrientAtOnce += nProjectionsAtOnce)
  {
    // ***************************************************************************************
    // ***** Creating Projection for given orientation and transforming to
    // Fourier space *****
    if (DebugOutput >= 1)
    {
      timer2.ResetStart();
      timer.ResetStart();
    }
    int iOrientEndAtOnce =
        std::min(iOrientEnd, iOrientAtOnce + nProjectionsAtOnce);

// **************************Parallel orientations for projections at
// once***************

#pragma omp parallel for
    for (int iOrient = iOrientAtOnce; iOrient < iOrientEndAtOnce; iOrient++)
    {
      createProjection(iOrient,
                       &proj_mapsFFT[(iOrient - iOrientAtOnce) * ProjMapSize]);
    }
    if (DebugOutput >= 1)
    {
      ts.time = timer.GetCurrentElapsedTime();
      ts.Add(TS_PROJECTION);
      if (DebugOutput >= 2)
        printf("\tTime Projection %d-%d: %f (rank %d)\n", iOrientAtOnce,
               iOrientEndAtOnce - 1, ts.time, mpi_rank);
    }
    /* Recalibrate if needed */
    if (Autotuning && ((iOrientAtOnce - iOrientStart) % RECALIB_FACTOR == 0) &&
        ((iOrientEnd - iOrientAtOnce) > RECALIB_FACTOR) &&
        (iOrientAtOnce != iOrientStart))
    {
      aut.Reset();
      rebalanceWrapper(aut.Workload());
    }

    for (int iOrient = iOrientAtOnce; iOrient < iOrientEndAtOnce; iOrient++)
    {
      mycomplex_t *proj_mapFFT =
          &proj_mapsFFT[(iOrient - iOrientAtOnce) * ProjMapSize];

      // ***************************************************************************************
      // ***** **** Internal Loop over PSF/CTF convolutions **** *****
      for (int iConvAtOnce = 0; iConvAtOnce < param.nTotCTFs;
           iConvAtOnce += param.nTotParallelConv)
      {
        if (DebugOutput >= 1)
          timer.ResetStart();
        int iConvEndAtOnce =
            std::min(param.nTotCTFs, iConvAtOnce + param.nTotParallelConv);
        // Total number of convolutions that can be treated in this iteration in
        // parallel
        int maxParallelConv = iConvEndAtOnce - iConvAtOnce;
#pragma omp parallel for
        for (int iConv = iConvAtOnce; iConv < iConvEndAtOnce; iConv++)
        {
          // *** Calculating convolutions of projection map and
          // crosscorrelations ***
          int i =
              (iPipeline & 1) * param.nTotParallelConv + (iConv - iConvAtOnce);
          mycomplex_t *localmultFFT = &conv_mapsFFT[i * param.FFTMapSize];

          createConvolutedProjectionMap(iOrient, iConv, proj_mapFFT,
                                        localmultFFT, comp_params[i].sumC,
                                        comp_params[i].sumsquareC);

          comp_params[i].amp = param.CtfParam[iConv].pos[0];
          comp_params[i].pha = param.CtfParam[iConv].pos[1];
          comp_params[i].env = param.CtfParam[iConv].pos[2];
        }
        if (DebugOutput >= 1)
        {
          ts.time = timer.GetCurrentElapsedTime();
          ts.Add(TS_CONVOLUTION);
          if (DebugOutput >= 2)
            printf("\t\tTime Convolution %d %d-%d: %f (rank %d)\n", iOrient,
                   iConvAtOnce, iConvEndAtOnce - 1, ts.time, mpi_rank);
        }

        // ******************Internal loop over Reference images CUDA or
        // OpenMP******************
        // *** Comparing each calculated convoluted map with all experimental
        // maps ***
        ts.time = 0.;
        if ((DebugOutput >= 1) || (Autotuning && aut.Needed(iConvAtOnce)))
          timer.ResetStart();
        compareRefMaps(iPipeline++, iOrient, iConvAtOnce, maxParallelConv,
                       conv_mapsFFT, comp_params);
        if (DebugOutput >= 1)
        {
          ts.time = timer.GetCurrentElapsedTime();
          ts.Add(TS_COMPARISON);
        }
        if (DebugOutput >= 2)
        {
          if (Autotuning)
            printf("\t\tTime Comparison %d %d-%d: %f sec with GPU workload "
                   "%d%% (rank %d)\n",
                   iOrient, iConvAtOnce, iConvEndAtOnce - 1, ts.time,
                   aut.Workload(), mpi_rank);
          else
            printf("\t\tTime Comparison %d %d-%d: %f sec (rank %d)\n", iOrient,
                   iConvAtOnce, iConvEndAtOnce - 1, ts.time, mpi_rank);
        }
        if (Autotuning && aut.Needed(iConvAtOnce))
        {
          if (ts.time == 0.)
            ts.time = timer.GetCurrentElapsedTime();
          aut.Tune(ts.time);
          if (aut.Finished() && DebugOutput >= 1)
            printf("\tOptimal GPU workload %d%% (rank %d)\n", aut.Workload(),
                   mpi_rank);
          rebalanceWrapper(aut.Workload());
        }
      }
      if (DebugOutput >= 1)
      {
        ts.time = timer2.GetCurrentElapsedTime();
        ts.Add(TS_TPROJECTION);
        printf("\tTotal time for projection %d: %f (rank %d)\n", iOrient,
               ts.time, mpi_rank);
        timer2.ResetStart();
      }
    }
  }
  /* Statistical summary on different parts of the code */
  if (DebugOutput >= 1)
  {
    ts.PrintTimeStat(mpi_rank);
    ts.EmptyTimeStat();
  }

  // deallocating fftw_complex vector
  myfftw_free(proj_mapsFFT);
  myfftw_free(conv_mapsFFT);

  deviceFinishRun();

// *******************************************************************************
// ************* Collecing all the probabilities from MPI replicas
// ***************

#ifdef WITH_MPI
  if (mpi_size > 1)
  {
    if (DebugOutput >= 1 && mpi_rank == 0)
      timer.ResetStart();
    // Reduce Constant and summarize probabilities
    {
      myprob_t *tmp1 = new myprob_t[RefMap.ntotRefMap];
      myprob_t *tmp2 = new myprob_t[RefMap.ntotRefMap];
      myprob_t *tmp3 = new myprob_t[RefMap.ntotRefMap];
      for (int i = 0; i < RefMap.ntotRefMap; i++)
      {
        tmp1[i] = pProb.getProbMap(i).Constoadd;
      }
      MPI_Allreduce(tmp1, tmp2, RefMap.ntotRefMap, MY_MPI_FLOAT, MPI_MAX,
                    MPI_COMM_WORLD);

      for (int i = 0; i < RefMap.ntotRefMap; i++)
      {
        bioem_Probability_map &pProbMap = pProb.getProbMap(i);
#ifdef DEBUG
        cout << "Reduction " << mpi_rank << " Map " << i << " Prob "
             << pProbMap.Total << " Const " << pProbMap.Constoadd << "\n";
#endif
        tmp1[i] = pProbMap.Total * exp(pProbMap.Constoadd - tmp2[i]);
      }
      MPI_Reduce(tmp1, tmp3, RefMap.ntotRefMap, MY_MPI_FLOAT, MPI_SUM, 0,
                 MPI_COMM_WORLD);

      // Find MaxProb
      MPI_Status mpistatus;
      {
        int *tmpi1 = new int[RefMap.ntotRefMap];
        int *tmpi2 = new int[RefMap.ntotRefMap];
        for (int i = 0; i < RefMap.ntotRefMap; i++)
        {
          bioem_Probability_map &pProbMap = pProb.getProbMap(i);
          tmpi1[i] = tmp2[i] <= pProbMap.Constoadd ? mpi_rank : -1;
          // temporary array that has the mpirank for the highest pProb.constant
        }
        MPI_Allreduce(tmpi1, tmpi2, RefMap.ntotRefMap, MPI_INT, MPI_MAX,
                      MPI_COMM_WORLD);
        for (int i = 0; i < RefMap.ntotRefMap; i++)
        {
          if (tmpi2[i] == -1)
          {
            if (mpi_rank == 0)
              printf("Error: Could not find highest probability\n");
          }
          else if (tmpi2[i] !=
                   0) // Skip if rank 0 already has highest probability
          {
            if (mpi_rank == 0)
            {
              MPI_Recv(&pProb.getProbMap(i).max,
                       sizeof(pProb.getProbMap(i).max), MPI_BYTE, tmpi2[i], i,
                       MPI_COMM_WORLD, &mpistatus);
            }
            else if (mpi_rank == tmpi2[i])
            {
              MPI_Send(&pProb.getProbMap(i).max,
                       sizeof(pProb.getProbMap(i).max), MPI_BYTE, 0, i,
                       MPI_COMM_WORLD);
            }
          }
        }
        delete[] tmpi1;
        delete[] tmpi2;
      }

      if (mpi_rank == 0)
      {
        for (int i = 0; i < RefMap.ntotRefMap; i++)
        {
          bioem_Probability_map &pProbMap = pProb.getProbMap(i);
          pProbMap.Total = tmp3[i];
          pProbMap.Constoadd = tmp2[i];
        }
      }

      delete[] tmp1;
      delete[] tmp2;
      delete[] tmp3;
      if (DebugOutput >= 1 && mpi_rank == 0 && mpi_size > 1)
        printf("Time MPI Reduction: %f\n", timer.GetCurrentElapsedTime());
    }

    // Angle Reduction and Probability summation for individual angles
    if (param.param_device.writeAngles)
    {
      const int count = RefMap.ntotRefMap * param.nTotGridAngles;
      myprob_t *tmp1 = new myprob_t[count];
      myprob_t *tmp2 = new myprob_t[count];
      myprob_t *tmp3 = new myprob_t[count];
      for (int i = 0; i < RefMap.ntotRefMap; i++)
      {
        for (int j = 0; j < param.nTotGridAngles; j++)
        {
          //	      tmp1[i] = pProb.getProbMap(i).Constoadd;
          //	      bioem_Probability_angle& pProbAngle =
          // pProb.getProbAngle(i, j);
          tmp1[i * param.nTotGridAngles + j] =
              pProb.getProbAngle(i, j).ConstAngle;
        }
      }

      MPI_Allreduce(tmp1, tmp2, count, MY_MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);
      for (int i = 0; i < RefMap.ntotRefMap; i++)
      {
        for (int j = 0; j < param.nTotGridAngles; j++)
        {
          bioem_Probability_angle &pProbAngle = pProb.getProbAngle(i, j);
          tmp1[i * param.nTotGridAngles + j] =
              pProbAngle.forAngles *
              exp(pProbAngle.ConstAngle - tmp2[i * param.nTotGridAngles + j]);
        }
      }
      MPI_Reduce(tmp1, tmp3, count, MY_MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
      if (mpi_rank == 0)
      {
        for (int i = 0; i < RefMap.ntotRefMap; i++)
        {
          for (int j = 0; j < param.nTotGridAngles; j++)
          {
            bioem_Probability_angle &pProbAngle = pProb.getProbAngle(i, j);
            pProbAngle.forAngles = tmp3[i * param.nTotGridAngles + j];
            pProbAngle.ConstAngle = tmp2[i * param.nTotGridAngles + j];
          }
        }
      }
      delete[] tmp1;
      delete[] tmp2;
      delete[] tmp3;
    }
  }
#endif

  // ************* Writing Out Probabilities ***************
  if (mpi_rank == 0)
  {

    // Output for Angle Probability File
    ofstream angProbfile;
    angProbfile.precision(OUTPUT_PRECISION);
    angProbfile.setf(ios::fixed);
    if (param.param_device.writeAngles)
    {
      angProbfile.open("ANG_PROB");
      angProbfile << "************************* HEADER:: NOTATION "
                     "*******************************************\n";
      if (!param.doquater)
      {
        angProbfile << " RefMap:  MapNumber ; alpha[rad] - beta[rad] - "
                       "gamma[rad] - logP - cal log Probability + Constant: "
                       "Numerical Const.+ log (volume) + prior ang\n";
      }
      else
      {
        angProbfile << " RefMap:  MapNumber ; q1 - q2 -q3 - logP- cal log "
                       "Probability + Constant: Numerical Const. + log "
                       "(volume) + prior ang\n";
      };
      angProbfile << "************************* HEADER:: NOTATION "
                     "*******************************************\n";
      //          angProbfile <<"Model Used: " << modelfile.c_str() << "\n";
      //          angProbfile <<"Input Used: " << infile.c_str() << "\n";
    }

    // Output for Standard Probability
    ofstream outputProbFile;
    outputProbFile.precision(OUTPUT_PRECISION);
    outputProbFile.setf(ios::fixed);
    outputProbFile.open(OutfileName.c_str());
    outputProbFile << "************************* HEADER:: NOTATION "
                      "*******************************************\n";
    outputProbFile << "Notation= RefMap:  MapNumber ; LogProb natural "
                      "logarithm of posterior Probability ; Constant: "
                      "Numerical Const. for adding Probabilities \n";
    if (!param.doquater)
    {
      if (param.usepsf)
      {
        outputProbFile << "Notation= RefMap:  MapNumber ; Maximizing Param: "
                          "MaxLogProb - alpha[rad] - beta[rad] - gamma[rad] - "
                          "PSF amp - PSF phase - PSF envelope - center x - "
                          "center y - normalization - offsett \n";
      }
      else
      {
        outputProbFile << "Notation= RefMap:  MapNumber ; Maximizing Param: "
                          "MaxLogProb - alpha[rad] - beta[rad] - gamma[rad] - "
                          "CTF amp - CTF defocus - CTF B-Env - center x - "
                          "center y - normalization - offsett \n";
      }
    }
    else
    {
      if (param.usepsf)
      {
        //     if( localcc[rx * param.param_device.NumberPixels + ry] <
        outputProbFile << "Notation= RefMap:  MapNumber ; Maximizing Param: "
                          "MaxLogProb - q1 - q2 - q3 - q4 -PSF amp - PSF phase "
                          "- PSF envelope - center x - center y - "
                          "normalization - offsett \n";
      }
      else
      {
        outputProbFile << "Notation= RefMap:  MapNumber ; Maximizing Param: "
                          "MaxLogProb - q1 - q2 - q3 - q4 - CTF amp - CTF "
                          "defocus - CTF B-Env - center x - center y - "
                          "normalization - offsett \n";
      }
    }
    if (param.writeCTF)
      outputProbFile << " RefMap:  MapNumber ; CTFMaxParm: defocus - b-Env (B "
                        "ref. Penzeck 2010)\n";
    if (param.yespriorAngles)
      outputProbFile << "**** Remark: Using Prior Proability in Angles ****\n";
    outputProbFile << "************************* HEADER:: NOTATION "
                      "*******************************************\n\n";

    // Loop over reference maps
    // ************* Over all maps ***************

    for (int iRefMap = 0; iRefMap < RefMap.ntotRefMap; iRefMap++)
    {
      // **** Total Probability ***
      bioem_Probability_map &pProbMap = pProb.getProbMap(iRefMap);

      // Controll for Value of Total Probability
      // cout << pProbMap.Total << " " <<  pProbMap.Constoadd << " " << FLT_MAX
      // <<" " << log(FLT_MAX) << "\n";
      if (pProbMap.Total > 1.e-38)
      {

        outputProbFile << "RefMap: " << iRefMap << " LogProb:  "
                       << log(pProbMap.Total) + pProbMap.Constoadd +
                              0.5 * log(M_PI) +
                              (1 - param.param_device.Ntotpi * 0.5) *
                                  (log(2 * M_PI) + 1) +
                              log(param.param_device.volu)
                       << " Constant: " << pProbMap.Constoadd << "\n";
        outputProbFile << "RefMap: " << iRefMap << " Maximizing Param: ";
        // *** Param that maximize probability****
        outputProbFile << (log(pProbMap.Total) + pProbMap.Constoadd +
                           0.5 * log(M_PI) +
                           (1 - param.param_device.Ntotpi * 0.5) *
                               (log(2 * M_PI) + 1) +
                           log(param.param_device.volu))
                       << " ";
      }
      else
      {
        outputProbFile
            << "Warining! with Map " << iRefMap
            << "Numerical Integrated Probability without constant = 0.0;\n";
        outputProbFile << "Warining RefMap: " << iRefMap
                       << "Check that constant is finite: "
                       << pProbMap.Constoadd << "\n";
        outputProbFile << "Warining RefMap: i) check model, ii) check refmap , "
                          "iii) check GPU on/off command inconsitency\n";
        //	    outputProbFile << "Warning! " << iRefMap << " LogProb:  "
        //<< pProbMap.Constoadd + 0.5 * log(M_PI) + (1 -
        // param.param_device.Ntotpi * 0.5)*(log(2 * M_PI) + 1) +
        // log(param.param_device.volu) << " Constant: " << pProbMap.Constoadd
        //<< "\n";
      }
      //	    outputProbFile << "RefMap: " << iRefMap << " Maximizing
      // Param: ";

      // *** Param that maximize probability****
      //	    outputProbFile << (pProbMap.Constoadd + 0.5 * log(M_PI) + (1
      //- param.param_device.Ntotpi * 0.5) * (log(2 * M_PI) + 1) +
      // log(param.param_device.volu)) << " ";

      outputProbFile << param.angles[pProbMap.max.max_prob_orient].pos[0]
                     << " [] ";
      outputProbFile << param.angles[pProbMap.max.max_prob_orient].pos[1]
                     << " [] ";
      outputProbFile << param.angles[pProbMap.max.max_prob_orient].pos[2]
                     << " [] ";
      if (param.doquater)
        outputProbFile << param.angles[pProbMap.max.max_prob_orient].quat4
                       << " [] ";
      outputProbFile << param.CtfParam[pProbMap.max.max_prob_conv].pos[0]
                     << " [] ";
      if (!param.usepsf)
      {
        outputProbFile << param.CtfParam[pProbMap.max.max_prob_conv].pos[1] /
                              2.f / M_PI / param.elecwavel * 0.0001
                       << " [micro-m] ";
      }
      else
      {
        outputProbFile << param.CtfParam[pProbMap.max.max_prob_conv].pos[1]
                       << " [1/A²] ";
      }
      if (!param.usepsf)
      {
        outputProbFile << param.CtfParam[pProbMap.max.max_prob_conv].pos[2]
                       << " [A²] ";
      }
      else
      {
        outputProbFile << param.CtfParam[pProbMap.max.max_prob_conv].pos[2]
                       << " [1/A²] ";
      }
      outputProbFile << pProbMap.max.max_prob_cent_x << " [pix] ";
      outputProbFile << pProbMap.max.max_prob_cent_y << " [pix] ";
      outputProbFile << pProbMap.max.max_prob_norm << " [] ";
      outputProbFile << pProbMap.max.max_prob_mu << " [] ";
      outputProbFile << "\n";

      // Writing out CTF parameters if requiered
      if (param.writeCTF && param.usepsf)
      {

        myfloat_t denomi;
        denomi = param.CtfParam[pProbMap.max.max_prob_conv].pos[1] *
                     param.CtfParam[pProbMap.max.max_prob_conv].pos[1] +
                 param.CtfParam[pProbMap.max.max_prob_conv].pos[2] *
                     param.CtfParam[pProbMap.max.max_prob_conv].pos[2];
        outputProbFile << "RefMap: " << iRefMap << " CTFMaxParam: ";
        outputProbFile
            << 2 * M_PI * param.CtfParam[pProbMap.max.max_prob_conv].pos[1] /
                   denomi / param.elecwavel * 0.0001
            << " [micro-m] ";
        outputProbFile
            << 4 * M_PI * M_PI *
                   param.CtfParam[pProbMap.max.max_prob_conv].pos[2] / denomi
            << " [A²] \n";
      }

      //*************** Writing Individual Angle probabilities
      if (param.param_device.writeAngles)
      {
        // Finding the best param.param_device.writeAngles probabilities
        // This implementation is clean, but not the most optimal one
        // and it supposes param.param_device.writeAngles <<
        // param.nTotGridAngles
        unsigned K =
            param.param_device.writeAngles; // number of best probabilities
                                            // clang-format off
        std::priority_queue<std::pair<double, int>,
                            std::vector<std::pair<double, int> >,
                            std::greater<std::pair<double, int> > >
            q;
        // clang-format on
        for (int iOrient = 0; iOrient < param.nTotGridAngles; iOrient++)
        {
          bioem_Probability_angle &pProbAngle =
              pProb.getProbAngle(iRefMap, iOrient);

          myprob_t logp =
              log(pProbAngle.forAngles) + pProbAngle.ConstAngle +
              0.5 * log(M_PI) +
              (1 - param.param_device.Ntotpi * 0.5) * (log(2 * M_PI) + 1) +
              log(param.param_device.volu);

          if (q.size() < K)
            q.push(std::pair<double, int>(logp, iOrient));
          else if (q.top().first < logp)
          {
            q.pop();
            q.push(std::pair<double, int>(logp, iOrient));
          }
        }
        K = q.size();
        int *rev_iOrient = (int *) malloc(K * sizeof(int));
        myprob_t *rev_logp = (myprob_t *) malloc(K * sizeof(myprob_t));
        for (int i = K - 1; i >= 0; i--)
        {
          rev_iOrient[i] = q.top().second;
          rev_logp[i] = q.top().first;
          q.pop();
        }
        for (unsigned i = 0; i < K; i++)
        {
          int iOrient = rev_iOrient[i];
          bioem_Probability_angle &pProbAngle =
              pProb.getProbAngle(iRefMap, iOrient);
          myprob_t logp = rev_logp[i];

          if (!param.doquater)
          {
            // For Euler Angles
            if (param.yespriorAngles)
            {
              logp += param.angprior[iOrient];
              angProbfile << " " << iRefMap << " "
                          << param.angles[iOrient].pos[0] << " "
                          << param.angles[iOrient].pos[1] << " "
                          << param.angles[iOrient].pos[2] << " " << logp
                          << " Separated: " << log(pProbAngle.forAngles) << " "
                          << pProbAngle.ConstAngle << " "
                          << 0.5 * log(M_PI) +
                                 (1 - param.param_device.Ntotpi * 0.5) *
                                     (log(2 * M_PI) + 1) +
                                 log(param.param_device.volu)
                          << " " << param.angprior[iOrient] << "\n";
            }
            else
            {
              angProbfile << " " << iRefMap << " "
                          << param.angles[iOrient].pos[0] << " "
                          << param.angles[iOrient].pos[1] << " "
                          << param.angles[iOrient].pos[2] << " " << logp
                          << " Separated: " << log(pProbAngle.forAngles) << " "
                          << pProbAngle.ConstAngle << " "
                          << 0.5 * log(M_PI) +
                                 (1 - param.param_device.Ntotpi * 0.5) *
                                     (log(2 * M_PI) + 1) +
                                 log(param.param_device.volu)
                          << "\n";
            }
          }
          else
          {
            // Samething but for Quaternions
            if (param.yespriorAngles)
            {
              logp += param.angprior[iOrient];
              angProbfile << " " << iRefMap << " "
                          << param.angles[iOrient].pos[0] << " "
                          << param.angles[iOrient].pos[1] << " "
                          << param.angles[iOrient].pos[2] << " "
                          << param.angles[iOrient].quat4 << " " << logp
                          << " Separated: " << log(pProbAngle.forAngles) << " "
                          << pProbAngle.ConstAngle << " "
                          << 0.5 * log(M_PI) +
                                 (1 - param.param_device.Ntotpi * 0.5) *
                                     (log(2 * M_PI) + 1) +
                                 log(param.param_device.volu)
                          << " " << param.angprior[iOrient] << "\n";
            }
            else
            {
              angProbfile << " " << iRefMap << " "
                          << param.angles[iOrient].pos[0] << " "
                          << param.angles[iOrient].pos[1] << " "
                          << param.angles[iOrient].pos[2] << " "
                          << param.angles[iOrient].quat4 << " " << logp
                          << " Separated: " << log(pProbAngle.forAngles) << " "
                          << pProbAngle.ConstAngle << " "
                          << 0.5 * log(M_PI) +
                                 (1 - param.param_device.Ntotpi * 0.5) *
                                     (log(2 * M_PI) + 1) +
                                 log(param.param_device.volu)
                          << "\n";
            }
          }
        }
        free(rev_iOrient);
        free(rev_logp);
      }
    }

    if (param.param_device.writeAngles)
    {
      angProbfile.close();
    }

    outputProbFile.close();
  }

  return (0);
}

int bioem::compareRefMaps(int iPipeline, int iOrient, int iConvStart,
                          int maxParallelConv, mycomplex_t *localmultFFT,
                          myparam5_t *comp_params, const int startMap)
{
  //***************************************************************************************
  //***** BioEM routine for comparing reference maps to convoluted maps *****
  //***************************************************************************************
  cuda_custom_timeslot("Comparison", iOrient, iConvStart, COLOR_COMPARISON);

  int k = (iPipeline & 1) * param.nTotParallelConv;

  if (BioEMAlgo == 1)
  {
#pragma omp parallel for schedule(dynamic, 1)
    for (int iRefMap = startMap; iRefMap < RefMap.ntotRefMap; iRefMap++)
    {
      const int num = omp_get_thread_num();
      for (int iConv = 0; iConv < maxParallelConv; iConv++)
      {
        calculateCCFFT(iRefMap, &localmultFFT[(k + iConv) * param.FFTMapSize],
                       param.fft_scratch_complex[num],
                       param.fft_scratch_real[num]);
        doRefMapFFT(
            iRefMap, iOrient, iConvStart + iConv, comp_params[k + iConv].amp,
            comp_params[k + iConv].pha, comp_params[k + iConv].env,
            comp_params[k + iConv].sumC, comp_params[k + iConv].sumsquareC,
            param.fft_scratch_real[num], pProb, param.param_device, RefMap);
      }
    }
  }
  else
  {
    myblockCPU_t *comp_blocks = new myblockCPU_t[maxParallelConv];
    for (int iRefMap = startMap; iRefMap < RefMap.ntotRefMap; iRefMap++)
    {
#pragma omp parallel for schedule(dynamic, 1)
      for (int iConv = 0; iConv < maxParallelConv; iConv++)
      {
        const int num = omp_get_thread_num();
        calculateCCFFT(iRefMap, &localmultFFT[(k + iConv) * param.FFTMapSize],
                       param.fft_scratch_complex[num],
                       param.fft_scratch_real[num]);
        doRefMap_CPU_Parallel(iRefMap, iOrient, iConv,
                              param.fft_scratch_real[num], &comp_params[k],
                              comp_blocks);
      }
      doRefMap_CPU_Reduce(iRefMap, iOrient, iConvStart, maxParallelConv,
                          &comp_params[k], comp_blocks);
    }
    delete[] comp_blocks;
  }

  cuda_custom_timeslot_end;
  return (0);
}

inline void bioem::calculateCCFFT(int iRefMap, mycomplex_t *localConvFFT,
                                  mycomplex_t *localCCT, myfloat_t *lCC)
{
  //***************************************************************************************
  //***** Calculating cross correlation with FFT algorithm *****

  for (int i = 0; i < param.param_device.NumberPixels; i++)
  {
    for (int j = 0; j < param.param_device.NumberPixels; j++)
      lCC[i * param.param_device.NumberPixels + j] = 0.f;
  }

  const mycomplex_t *RefMapFFT = &RefMap.RefMapsFFT[iRefMap * param.FFTMapSize];
  for (int i = 0; i < param.param_device.NumberPixels *
                          param.param_device.NumberFFTPixels1D;
       i++)
  {
    localCCT[i][0] = localConvFFT[i][0] * RefMapFFT[i][0] +
                     localConvFFT[i][1] * RefMapFFT[i][1];
    localCCT[i][1] = localConvFFT[i][1] * RefMapFFT[i][0] -
                     localConvFFT[i][0] * RefMapFFT[i][1];
  }

  myfftw_execute_dft_c2r(param.fft_plan_c2r_backward, localCCT, lCC);
}

inline void bioem::doRefMap_CPU_Parallel(int iRefMap, int iOrient, int iConv,
                                         myfloat_t *lCC,
                                         myparam5_t *comp_params,
                                         myblockCPU_t *comp_block)
{
  //***************************************************************************************
  //***** Computation of log probabilities, done in parallel by OMP

  int myGlobalId = iConv * param.param_device.NtotDisp;
  myfloat_t bestLogpro = MIN_PROB;
  int dispC =
      param.param_device.NumberPixels - param.param_device.maxDisplaceCenter;
  int cent_x, cent_y, address, bestId = 0;
  myfloat_t value, bestValue = 0.;
  myprob_t logpro = 0., sumExp = 0.;

  for (int myX = 0; myX < param.param_device.NxDisp; myX++)
  {
    for (int myY = 0; myY < param.param_device.NxDisp; myY++, myGlobalId++)
    {
      cent_x = (myX * param.param_device.GridSpaceCenter + dispC) %
               param.param_device.NumberPixels;
      cent_y = (myY * param.param_device.GridSpaceCenter + dispC) %
               param.param_device.NumberPixels;
      address = cent_x * param.param_device.NumberPixels + cent_y;
      value = (myfloat_t) lCC[address] /
              (myfloat_t)(param.param_device.NumberPixels *
                          param.param_device.NumberPixels);

      logpro = calc_logpro(
          param.param_device, comp_params[iConv].amp, comp_params[iConv].pha,
          comp_params[iConv].env, comp_params[iConv].sumC,
          comp_params[iConv].sumsquareC, value, RefMap.sum_RefMap[iRefMap],
          RefMap.sumsquare_RefMap[iRefMap]);
#ifdef DEBUG_PROB
      printf("\t\t\tProb: iRefMap %d, iOrient %d, iConv %d, "
             "cent_x %d, cent_y %d, address %d, value %f, logpro %f\n",
             iRefMap, iOrient, iConv, cent_x, cent_y, address, value, logpro);
#endif
      if (bestLogpro < logpro)
      {
        sumExp *= exp(-logpro + bestLogpro);
        bestLogpro = logpro;
        bestId = myGlobalId;
        bestValue = value;
      }
      sumExp += exp(logpro - bestLogpro);
    }
  }

  comp_block[iConv].logpro = bestLogpro;
  comp_block[iConv].sumExp = sumExp;
  comp_block[iConv].id = bestId;
  comp_block[iConv].value = bestValue;
}

inline void bioem::doRefMap_CPU_Reduce(int iRefMap, int iOrient, int iConvStart,
                                       int maxParallelConv,
                                       myparam5_t *comp_params,
                                       myblockCPU_t *comp_block)
{
  //***************************************************************************************
  //***** Reduction of previously compututed log probabilities

  bioem_Probability_map &pProbMap = pProb.getProbMap(iRefMap);

  for (int i = 0; i < maxParallelConv; i++)
  {
    if (pProbMap.Constoadd < comp_block[i].logpro)
    {
      pProbMap.Total *= exp(-comp_block[i].logpro + pProbMap.Constoadd);
      pProbMap.Constoadd = comp_block[i].logpro;

      // ********** Getting parameters that maximize the probability ***********
      int myGlobalId = comp_block[i].id;
      int myConv = myGlobalId / param.param_device.NtotDisp;
      myGlobalId -= myConv * param.param_device.NtotDisp;
      int myX = myGlobalId / param.param_device.NxDisp;
      myGlobalId -= myX * param.param_device.NxDisp;
      int myY = myGlobalId;

      int dispC = param.param_device.NumberPixels -
                  param.param_device.maxDisplaceCenter;
      myfloat_t value = comp_block[i].value;

      pProbMap.max.max_prob_cent_x =
          -((myX * param.param_device.GridSpaceCenter + dispC) -
            param.param_device.NumberPixels);
      pProbMap.max.max_prob_cent_y =
          -((myY * param.param_device.GridSpaceCenter + dispC) -
            param.param_device.NumberPixels);
      pProbMap.max.max_prob_orient = iOrient;
      pProbMap.max.max_prob_conv = iConvStart + myConv;

      pProbMap.max.max_prob_norm =
          -(-comp_params[myConv].sumC * RefMap.sum_RefMap[iRefMap] +
            param.param_device.Ntotpi * value) /
          (comp_params[myConv].sumC * comp_params[myConv].sumC -
           comp_params[myConv].sumsquareC * param.param_device.Ntotpi);
      pProbMap.max.max_prob_mu =
          -(-comp_params[myConv].sumC * value +
            comp_params[myConv].sumsquareC * RefMap.sum_RefMap[iRefMap]) /
          (comp_params[myConv].sumC * comp_params[myConv].sumC -
           comp_params[myConv].sumsquareC * param.param_device.Ntotpi);

#ifdef DEBUG_PROB
      printf("\tProbabilities change: iRefMap %d, iOrient %d, iConv %d, "
             "Total %f, Const %f, bestlogpro %f, sumExp %f, bestId %d\n",
             iRefMap, iOrient, iConvStart + myConv, pProbMap.Total,
             pProbMap.Constoadd, comp_block[i].logpro, comp_block[i].sumExp,
             comp_block[i].id);
      printf("\tParameters: myConv %d, myX %d, myY %d, cent_x -, cent_y -, "
             "probX %d, probY %d\n",
             myConv, myX, myY, pProbMap.max.max_prob_cent_x,
             pProbMap.max.max_prob_cent_y);
#endif
    }
    pProbMap.Total +=
        comp_block[i].sumExp * exp(comp_block[i].logpro - pProbMap.Constoadd);
#ifdef DEBUG_PROB
    printf("\t\tProbabilities after Reduce: iRefMap %d, iOrient %d, iConv "
           "%d, Total %f, Const %f, bestlogpro %f, sumExp %f, bestId %d\n",
           iRefMap, iOrient, iConvStart, pProbMap.Total, pProbMap.Constoadd,
           comp_block[i].logpro, comp_block[i].sumExp, comp_block[i].id);
#endif

    // Code for writing angles, not used by default
    if (param.param_device.writeAngles)
    {
      bioem_Probability_angle &pProbAngle =
          pProb.getProbAngle(iRefMap, iOrient);
      if (pProbAngle.ConstAngle < comp_block[i].logpro)
      {
        pProbAngle.forAngles *=
            exp(-comp_block[i].logpro + pProbAngle.ConstAngle);
        pProbAngle.ConstAngle = comp_block[i].logpro;
      }
      pProbAngle.forAngles += comp_block[i].sumExp *
                              exp(comp_block[i].logpro - pProbAngle.ConstAngle);
    }
  }
}

int bioem::createProjection(int iMap, mycomplex_t *mapFFT)
{
  // **************************************************************************************
  // ****  BioEM Create Projection routine in Euler angles / Quaternions
  // ******************
  // ********************* and turns projection into Fourier space
  // ************************
  // **************************************************************************************

  cuda_custom_timeslot("Projection", iMap, 0, COLOR_PROJECTION);

  myfloat3_t RotatedPointsModel[Model.nPointsModel];
  myfloat_t rotmat[3][3];
  myfloat_t alpha, gam, beta;
  myfloat_t *localproj;

  localproj = param.fft_scratch_real[omp_get_thread_num()];
  memset(localproj, 0, param.param_device.NumberPixels *
                           param.param_device.NumberPixels *
                           sizeof(*localproj));

  //*************** Rotating the model ****************************
  //*************** Quaternions ****************************
  if (param.doquater)
  {

    myfloat_t quater[4];
    // quaternion
    quater[0] = param.angles[iMap].pos[0];
    quater[1] = param.angles[iMap].pos[1];
    quater[2] = param.angles[iMap].pos[2];
    quater[3] = param.angles[iMap].quat4;

    // Rotation Matrix for Quaterions (wikipeda)
    rotmat[0][0] = 1 - 2 * quater[1] * quater[1] - 2 * quater[2] * quater[2];
    rotmat[1][0] = 2 * (quater[0] * quater[1] - quater[2] * quater[3]);
    rotmat[2][0] = 2 * (quater[0] * quater[2] + quater[1] * quater[3]);
    rotmat[0][1] = 2 * (quater[0] * quater[1] + quater[2] * quater[3]);
    rotmat[1][1] = 1 - 2 * quater[0] * quater[0] - 2 * quater[2] * quater[2];
    rotmat[2][1] = 2 * (quater[1] * quater[2] - quater[0] * quater[3]);
    rotmat[0][2] = 2 * (quater[0] * quater[2] - quater[1] * quater[3]);
    rotmat[1][2] = 2 * (quater[1] * quater[2] + quater[0] * quater[3]);
    rotmat[2][2] = 1 - 2 * quater[0] * quater[0] - 2 * quater[1] * quater[1];
  }
  else
  {

    //*************** Euler Angles****************************
    // Doing Euler angles instead of Quaternions
    alpha = param.angles[iMap].pos[0];
    beta = param.angles[iMap].pos[1];
    gam = param.angles[iMap].pos[2];

//*** To see how things are going:
#ifdef DEBUG
    cout << "Id " << omp_get_thread_num() << " Angs: " << alpha << " " << beta
         << " " << gam << "\n";
#endif
    // ********** Creat Rotation with pre-defiend grid of orientations**********
    // Same notation as in Goldstein and Mathematica
    rotmat[0][0] = cos(gam) * cos(alpha) - cos(beta) * sin(alpha) * sin(gam);
    rotmat[0][1] = cos(gam) * sin(alpha) + cos(beta) * cos(alpha) * sin(gam);
    rotmat[0][2] = sin(gam) * sin(beta);
    rotmat[1][0] = -sin(gam) * cos(alpha) - cos(beta) * sin(alpha) * cos(gam);
    rotmat[1][1] = -sin(gam) * sin(alpha) + cos(beta) * cos(alpha) * cos(gam);
    rotmat[1][2] = cos(gam) * sin(beta);
    rotmat[2][0] = sin(beta) * sin(alpha);
    rotmat[2][1] = -sin(beta) * cos(alpha);
    rotmat[2][2] = cos(beta);
  }

  // The rotation matrix is calculated either for the quaternions or for the
  // euler angles
  for (int n = 0; n < Model.nPointsModel; n++)
  {
    RotatedPointsModel[n].pos[0] = 0.0;
    RotatedPointsModel[n].pos[1] = 0.0;
    RotatedPointsModel[n].pos[2] = 0.0;
  }
  for (int n = 0; n < Model.nPointsModel; n++)
  {
    for (int k = 0; k < 3; k++)
    {
      for (int j = 0; j < 3; j++)
      {
        RotatedPointsModel[n].pos[k] +=
            rotmat[k][j] * Model.points[n].point.pos[j];
      }
    }
  }

  if (param.printrotmod)
  {
    for (int n = 0; n < Model.nPointsModel; n++)
      cout << "ROTATED " << iMap << " " << n << " "
           << RotatedPointsModel[n].pos[0] << " "
           << RotatedPointsModel[n].pos[1] << " "
           << RotatedPointsModel[n].pos[2] << "\n";
  }
  int i, j;

  //*************** Creating projection ****************************
  //********** Projection with radius ***************
  int irad;
  myfloat_t dist, rad2;

  myfloat_t tempden = 0.0;

  for (int n = 0; n < Model.nPointsModel; n++)
  {
    if (Model.points[n].radius <= param.pixelSize)
    {
      //   cout << "Radius less than Pixel size: use keyword NO_PROJECT_RADIUS
      //   in inputfile\n";
      i = floor(RotatedPointsModel[n].pos[0] / param.pixelSize +
                (myfloat_t) param.param_device.NumberPixels / 2.0f + 0.5f);
      j = floor(RotatedPointsModel[n].pos[1] / param.pixelSize +
                (myfloat_t) param.param_device.NumberPixels / 2.0f + 0.5f);

      if (i < 0 || j < 0 || i >= param.param_device.NumberPixels ||
          j >= param.param_device.NumberPixels)
      {
        if (DebugOutput >= 0)
          cout << "WARNING:::: Model Point out of Projection map: " << i << ", "
               << j << "\n";
        //              continue;
        if (not param.ignorepointsout)
          exit(1);
      }

      localproj[i * param.param_device.NumberPixels + j] +=
          Model.points[n].density;
      tempden += Model.points[n].density;

      // exit(1);
    }
    else
    {

      // Getting Centers of Sphere
      i = floor(RotatedPointsModel[n].pos[0] / param.pixelSize +
                (myfloat_t) param.param_device.NumberPixels / 2.0f + 0.5f) -
          param.shiftX;
      j = floor(RotatedPointsModel[n].pos[1] / param.pixelSize +
                (myfloat_t) param.param_device.NumberPixels / 2.0f + 0.5f) -
          param.shiftY;
      // Getting the radius
      irad = int(Model.points[n].radius / param.pixelSize) + 1;
      rad2 = Model.points[n].radius * Model.points[n].radius;

      if (i < irad || j < irad || i >= param.param_device.NumberPixels-irad ||
          j >= param.param_device.NumberPixels-irad )
      {
 
        if (DebugOutput >= 0)
          cout << "WARNING::: Model Point out of Projection map: " << i << ", "
               << j << "\n";
        cout << "Model point " << n << "Rotation: " << iMap << " "
             << RotatedPointsModel[n].pos[0] << " "
             << RotatedPointsModel[n].pos[1] << " "
             << RotatedPointsModel[n].pos[2] << "\n";
        cout << "Original coor " << n << " " << Model.points[n].point.pos[0]
             << " " << Model.points[n].point.pos[1] << " "
             << Model.points[n].point.pos[2] << "\n";
        cout << "WARNING: Angle orient " << n << " "
             << param.angles[iMap].pos[0] << " " << param.angles[iMap].pos[1]
             << " " << param.angles[iMap].pos[2] << " out " << i << " " << j
             << "\n";
        cout << "WARNING: MPI rank " << mpi_rank << "\n";
        //              continue;
        if (not param.ignorepointsout)
          exit(1);
      }else{

      // Projecting over the radius
      for (int ii = i - irad; ii < i + irad + 1; ii++)
      {
        for (int jj = j - irad; jj < j + irad + 1; jj++)
        {
          dist = ((myfloat_t)(ii - i) * (ii - i) + (jj - j) * (jj - j)) *
                 param.pixelSize * param.pixelSize; // at pixel center
          if (dist < rad2)
          {
            localproj[ii * param.param_device.NumberPixels + jj] +=
                param.pixelSize * param.pixelSize * 2 * sqrt(rad2 - dist) *
                Model.points[n].density * 3 /
                (4 * M_PI * Model.points[n].radius * rad2);
            tempden += param.pixelSize * param.pixelSize * 2 *
                       sqrt(rad2 - dist) * Model.points[n].density * 3 /
                       (4 * M_PI * Model.points[n].radius * rad2);
          }
         }
        }
      }
    }
  }

  // To avoid numerical mismatch in projection errors we normalize by the
  // initial density

  myfloat_t ratioDen;

  ratioDen = Model.NormDen / tempden;

  for (int i = 0; i < param.param_device.NumberPixels; i++)
  {
    for (int j = 0; j < param.param_device.NumberPixels; j++)
    {
      localproj[i * param.param_device.NumberPixels + j] *= ratioDen;
    }
  }

// **** Output Just to check****
#ifdef DEBUG
  //  if(iMap == 0)
  {
    ofstream myexamplemap;
    ofstream myexampleRot;
    myexamplemap.open("MAP_i10");
    myexampleRot.open("Rot_i10");
    myexamplemap << "ANGLES " << alpha << " " << beta << " " << gam << "\n";
    for (int k = 0; k < param.param_device.NumberPixels; k++)
    {
      for (int j = 0; j < param.param_device.NumberPixels; j++)
        myexamplemap << "\nMAP " << k << " " << j << " "
                     << localproj[k * param.param_device.NumberPixels + j];
    }
    myexamplemap << " \n";
    for (int n = 0; n < Model.nPointsModel; n++)
      myexampleRot << "\nCOOR " << RotatedPointsModel[n].pos[0] << " "
                   << RotatedPointsModel[n].pos[1] << " "
                   << RotatedPointsModel[n].pos[2];
    myexamplemap.close();
    myexampleRot.close();
  }
#endif

  // ***** Converting projection to Fourier Space for Convolution later with
  // kernel****
  // ********** Omp Critical is necessary with FFTW*******

  myfftw_execute_dft_r2c(param.fft_plan_r2c_forward, localproj, mapFFT);

  cuda_custom_timeslot_end;

  return (0);
}

int bioem::createConvolutedProjectionMap(int iMap, int iConv,
                                         mycomplex_t *lproj,
                                         mycomplex_t *localmultFFT,
                                         myfloat_t &sumC, myfloat_t &sumsquareC)
{
  // **************************************************************************************
  // ****  BioEM Create Convoluted Projection Map routine, multiplies in Fourier
  // **********
  // **************** calculated Projection with convoluted precalculated
  // Kernel***********
  // *************** and Backtransforming it to real Space
  // ********************************
  // **************************************************************************************

  cuda_custom_timeslot("Convolution", iMap, iConv, COLOR_CONVOLUTION);

  // **** Multiplying FFTmap of model with corresponding kernel *******

  const mycomplex_t *refCTF = &param.refCTF[iConv * param.FFTMapSize];

  for (int i = 0; i < param.param_device.NumberPixels *
                          param.param_device.NumberFFTPixels1D;
       i++)
  {
    localmultFFT[i][0] =
        (lproj[i][0] * refCTF[i][0] + lproj[i][1] * refCTF[i][1]);
    localmultFFT[i][1] =
        (lproj[i][1] * refCTF[i][0] - lproj[i][0] * refCTF[i][1]);
  }

  // *** Calculating Cross-correlations of cal-convoluted map with its self
  // ***** (for BioEM formula)
  sumC = localmultFFT[0][0];

  //**** Calculating the second norm and storing it (for BioEM formula)
  sumsquareC = 0;

  //*** With FFT algorithm
  int jloopend = param.param_device.NumberFFTPixels1D;
  if ((param.param_device.NumberPixels & 1) == 0)
    jloopend--;
  for (int i = 0; i < param.param_device.NumberPixels; i++)
  {
    for (int j = 1; j < jloopend; j++)
    {
      int k = i * param.param_device.NumberFFTPixels1D + j;
      sumsquareC += (localmultFFT[k][0] * localmultFFT[k][0] +
                     localmultFFT[k][1] * localmultFFT[k][1]) *
                    2;
    }
    int k = i * param.param_device.NumberFFTPixels1D;
    sumsquareC += localmultFFT[k][0] * localmultFFT[k][0] +
                  localmultFFT[k][1] * localmultFFT[k][1];
    if ((param.param_device.NumberPixels & 1) == 0)
    {
      k += param.param_device.NumberFFTPixels1D - 1;
      sumsquareC += localmultFFT[k][0] * localmultFFT[k][0] +
                    localmultFFT[k][1] * localmultFFT[k][1];
    }
  }

  myfloat_t norm2 = (myfloat_t)(param.param_device.NumberPixels *
                                param.param_device.NumberPixels);
  sumsquareC = sumsquareC / norm2;

  cuda_custom_timeslot_end;

  return (0);
}

int bioem::createConvolutedProjectionMap_noFFT(mycomplex_t *lproj,
                                               myfloat_t *Mapconv,
                                               mycomplex_t *localmultFFT,
                                               myfloat_t &sumC,
                                               myfloat_t &sumsquareC)
{
  // **************************************************************************************
  // ****  BioEM Create Convoluted Projection Map routine, multiplies in Fourier
  // **********
  // **************** calculated Projection with convoluted precalculated
  // Kernel***********
  // *************** and Backtransforming it to real Space
  // ********************************
  // **************************************************************************************
  // *************** This routine is only for printing Model
  // ******************************
  // **************************************************************************************

  mycomplex_t *tmp = param.fft_scratch_complex[omp_get_thread_num()];

  // **** Multiplying FFTmap of model with corresponding kernel *******
  const mycomplex_t *refCTF = param.refCTF;

  for (int i = 0; i < param.param_device.NumberPixels *
                          param.param_device.NumberFFTPixels1D;
       i++)
  {
    localmultFFT[i][0] =
        (lproj[i][0] * refCTF[i][0] + lproj[i][1] * refCTF[i][1]);
    localmultFFT[i][1] =
        (lproj[i][1] * refCTF[i][0] - lproj[i][0] * refCTF[i][1]);
  }

  // *** Calculating Cross-correlations of cal-convoluted map with its self
  // ***** (for BioEM formula)
  sumC = localmultFFT[0][0];

  //**** Calculating the second norm and storing it (for BioEM formula)
  sumsquareC = 0;

  //***** Slow No FFT ***
  //**** Backtransforming the convoluted map it into real space
  // FFTW_C2R will destroy the input array, so we have to work on a copy here
  memcpy(tmp, localmultFFT, sizeof(mycomplex_t) *
                                param.param_device.NumberPixels *
                                param.param_device.NumberFFTPixels1D);

  // **** Bringing convoluted Map to real Space ****
  myfftw_execute_dft_c2r(param.fft_plan_c2r_backward, tmp, Mapconv);

  for (int i = 0;
       i < param.param_device.NumberPixels * param.param_device.NumberPixels;
       i++)
  {
    sumsquareC += Mapconv[i] * Mapconv[i];
    //	  cout << "CONV " << i << " " << Mapconv[i] << "\n";
  }

  myfloat_t norm2 = (myfloat_t)(param.param_device.NumberPixels *
                                param.param_device.NumberPixels);
  myfloat_t norm4 = norm2 * norm2;
  sumsquareC = sumsquareC / norm4;

  // **************************************************************************************
  // *********** Routine for printing out the best projetion
  // ******************************
  // **************************************************************************************

  // Calling random number routine from MersenneTwister.h
  MTRand mtr;

  // Generating random seed so the maps do not have correlated Noise
  mtr.seed(static_cast<unsigned int>(std::time(0)));

  memcpy(tmp, localmultFFT, sizeof(mycomplex_t) *
                                param.param_device.NumberPixels *
                                param.param_device.NumberFFTPixels1D);

  // **** Bringing convoluted Map to real Space ****
  myfftw_execute_dft_c2r(param.fft_plan_c2r_backward, tmp, Mapconv);

  // Calculating the cross-correlation to the ref maps
  // PILAR WORK RefMap.maps
  if (param.BestmapCalcCC)
  {
    myfloat_t ccbm = 0.;
    int kk, jj;

    for (int k = 0; k < param.param_device.NumberPixels; k++)
    {
      for (int j = 0; j < param.param_device.NumberPixels; j++)
      {
        // Missing periodicity and centers;
        kk = k;
        jj = j;
        if (k - param.ddx < 0)
          kk = param.param_device.NumberPixels - (k - param.ddx);
        if (j - param.ddy < 0)
          jj = param.param_device.NumberPixels - (j - param.ddy);
        if (k - param.ddx >= param.param_device.NumberPixels)
          kk = k - param.ddx - param.param_device.NumberPixels;
        if (j - param.ddy >= param.param_device.NumberPixels)
          jj = j - param.ddy - param.param_device.NumberPixels;

        ccbm += (Mapconv[kk * param.param_device.NumberPixels + jj] / norm2 *
                     param.bestnorm -
                 RefMap.maps[k * param.param_device.NumberPixels + j]) *
                (Mapconv[kk * param.param_device.NumberPixels + jj] / norm2 *
                     param.bestnorm -
                 RefMap.maps[k * param.param_device.NumberPixels + j]);
      }
    }
    cout << "CROSS CORELATION " << ccbm << "\n";
  }
  else
  {
    ofstream myexamplemap;
    myexamplemap.open("BESTMAP");
    for (int k = 0; k < param.param_device.NumberPixels; k++)
    {
      for (int j = 0; j < param.param_device.NumberPixels; j++)
      {
        if (!param.withnoise)
        {
          myexamplemap << "\nMAP " << k + param.ddx << " " << j + param.ddy
                       << " "
                       << Mapconv[k * param.param_device.NumberPixels + j] /
                                  norm2 * param.bestnorm +
                              param.bestoff;
          if (k + param.ddx < param.param_device.NumberPixels &&
              j + param.ddy < param.param_device.NumberPixels)
          {
            myexamplemap
                << "\nMAPddx " << k << " " << j << " "
                << Mapconv[(k - param.ddx) * param.param_device.NumberPixels +
                           j - param.ddy] /
                           norm2 * param.bestnorm +
                       param.bestoff;
          }
        }
        else
        {
          myexamplemap << "\nMAP " << k + param.ddx << " " << j + param.ddy
                       << " "
                       << Mapconv[k * param.param_device.NumberPixels + j] /
                                  norm2 * param.bestnorm +
                              param.bestoff +
                              mtr.randNorm(0.0,
                                           param.stnoise); //\\+ distn(gen);
          //		cout << distn(gen) << "CHECK\n";
        }
      }
      myexamplemap << " \n";
    }
    myexamplemap.close();

    cout << "\n\nBest map printed in file: BESTMAP with gnuplot format in "
            "columns 2, 3 and 4. \n\n\n";
  }
  return (0);
}

int bioem::calcross_cor(myfloat_t *localmap, myfloat_t &sum,
                        myfloat_t &sumsquare)
{
  // *********************** Routine to calculate Cross
  // correlations***********************

  sum = 0.0;
  sumsquare = 0.0;
  for (int i = 0; i < param.param_device.NumberPixels; i++)
  {
    for (int j = 0; j < param.param_device.NumberPixels; j++)
    {
      // Calculate Sum of pixels
      sum += localmap[i * param.param_device.NumberPixels + j];
      // Calculate Sum of pixels squared
      sumsquare += localmap[i * param.param_device.NumberPixels + j] *
                   localmap[i * param.param_device.NumberPixels + j];
    }
  }
  return (0);
}

int bioem::deviceInit() { return (0); }

int bioem::deviceStartRun() { return (0); }

int bioem::deviceFinishRun() { return (0); }

void *bioem::malloc_device_host(size_t size) { return (mallocchk(size)); }

void bioem::free_device_host(void *ptr) { free(ptr); }

void bioem::rebalanceWrapper(int workload)
{
  cuda_custom_timeslot("Rebalance workload", -1, workload, COLOR_WORKLOAD);
  rebalance(workload);
  cuda_custom_timeslot_end;
}

void bioem::rebalance(int workload) {}
