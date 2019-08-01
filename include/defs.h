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

#ifndef BIOEM_DEFS_H
#define BIOEM_DEFS_H

#define myError(error, ...)                                                    \
  {                                                                            \
    printf("!!!!!!!!!!!!!!!!!!!!!!!!\n");                                      \
    printf("Error - ");                                                        \
    printf((error), ##__VA_ARGS__);                                            \
    printf(" (%s: %d)\n", __FILE__, __LINE__);                                 \
    printf("!!!!!!!!!!!!!!!!!!!!!!!!\n");                                      \
    exit(1);                                                                   \
  }

#define myWarning(warning, ...)                                                \
  {                                                                            \
    printf("Warning - ");                                                      \
    printf((warning), ##__VA_ARGS__);                                          \
    printf(" (%s: %d)\n", __FILE__, __LINE__);                                 \
  }

#define mySscanf(count, buf, fmt, ...)                                         \
  {                                                                            \
    int rc = sscanf((buf), (fmt), ##__VA_ARGS__);                              \
    if (rc != count)                                                           \
      myError("line parsed by sscanf has wrong argument");                     \
  }

static const char FILE_COORDREAD[] = "COORDREAD";
static const char FILE_ANG_PROB[] = "ANG_PROB";
static const char FILE_BESTMAP[] = "BESTMAP";
static const char FILE_MAPS_DUMP[] = "maps.dump";
static const char FILE_MODEL_DUMP[] = "model.dump";

#define BIOEM_PROB_DOUBLE
//#define BIOEM_USE_DOUBLE
//#define DEBUG
//#define DEBUG_GPU
//#define DEBUG_PROB

#define READ_PARALLEL 1

#ifndef BIOEM_PROB_DOUBLE
typedef float myprob_t;
#define MY_MPI_FLOAT MPI_FLOAT
#else
typedef double myprob_t;
#define MY_MPI_FLOAT MPI_DOUBLE
#endif

#ifndef BIOEM_USE_DOUBLE
#define MIN_PROB -999999.
typedef float myfloat_t;
#define myfftw_malloc fftwf_malloc
#define myfftw_free fftwf_free
#define myfftw_destroy_plan fftwf_destroy_plan
#define myfftw_execute fftwf_execute
#define myfftw_execute_dft fftwf_execute_dft
#define myfftw_execute_dft_r2c fftwf_execute_dft_r2c
#define myfftw_execute_dft_c2r fftwf_execute_dft_c2r
#define myfftw_plan_dft_2d fftwf_plan_dft_2d
#define myfftw_plan_dft_r2c_2d fftwf_plan_dft_r2c_2d
#define myfftw_plan_dft_c2r_2d fftwf_plan_dft_c2r_2d
#define myfftw_plan fftwf_plan
#define myfftw_cleanup fftwf_cleanup
#define MY_CUFFT_C2R CUFFT_C2R
#define mycufftExecC2R cufftExecC2R
#define mycuComplex_t cuComplex
#else
typedef double myfloat_t;
#define MIN_PROB -999999.
#define myfftw_malloc fftw_malloc
#define myfftw_free fftw_free
#define myfftw_destroy_plan fftw_destroy_plan
#define myfftw_execute fftw_execute
#define myfftw_execute_dft fftw_execute_dft
#define myfftw_execute_dft_r2c fftw_execute_dft_r2c
#define myfftw_execute_dft_c2r fftw_execute_dft_c2r
#define myfftw_plan_dft_2d fftw_plan_dft_2d
#define myfftw_plan_dft_r2c_2d fftw_plan_dft_r2c_2d
#define myfftw_plan_dft_c2r_2d fftw_plan_dft_c2r_2d
#define myfftw_plan fftw_plan
#define myfftw_cleanup fftw_cleanup
#define mycufftExecC2R cufftExecZ2D
#define mycuComplex_t cuDoubleComplex
#define MY_CUFFT_C2R CUFFT_Z2D
#endif
typedef myfloat_t mycomplex_t[2];

#define BIOEM_FLOAT_3_PHYSICAL_SIZE 3 // Possible set to 4 for GPU

struct myfloat3_t
{
  myfloat_t pos[BIOEM_FLOAT_3_PHYSICAL_SIZE];
  myfloat_t quat4;
  //     myfloat_t prior;
};

/* myoptions
Structure for saving options, in order to mimic old Boost program_options
behaviour
*/
struct myoption_t
{
  const char *name;
  int arg;
  const char *desc;
  bool hidden;
};

/* comp_params
Put all parameters needed for each comparison in a single structure
This makes code cleaner and requires less GPU transfers
*/
struct myparam5_t
{
  myfloat_t amp;
  myfloat_t pha;
  myfloat_t env;
  myfloat_t sumC;
  myfloat_t sumsquareC;
};

/* comp_block
Put all parameters created by each inside-block comparison
This makes code cleaner
*/
// For GPUs
struct myblockGPU_t
{
  myprob_t logpro;
  int id;
  myprob_t sumExp;
  myprob_t sumAngles;
};
// For CPUs (easier to save value as well)
struct myblockCPU_t
{
  myprob_t logpro;
  int id;
  myprob_t sumExp;
  myfloat_t value;
};

#ifdef BIOEM_GPUCODE
#define myThreadIdxX threadIdx.x
#define myThreadIdxY threadIdx.y
#define myBlockDimX blockDim.x
#define myBlockDimY blockDim.y
#define myBlockIdxX blockIdx.x
#define myBlockIdxY blockIdx.y
#define myGridDimX gridDim.x
#else
#define __device__
#define __host__
#define myThreadIdxX 0
#define myThreadIdxY 0
#define myBlockDimX 1
#define myBlockDimY 1
#define myBlockIdxX 0
#define myBlockIdxY 0
#endif

#define OUTPUT_PRECISION 4

#define CUDA_THREAD_COUNT_ALGO1 256
#define CUDA_THREAD_COUNT_ALGO2 512
#define CUDA_THREAD_MAX 1024
#define CUDA_FFTS_AT_ONCE 1024

#define PIPELINE_LVL 2
#define MULTISTREAM_LVL 2
#define SPLIT_MAPS_LVL 2

/* Autotuning
   Autotuning algorithms:
    1. AlgoSimple = 1; Testing workload values between 100 and 30, all multiples
   of 5. Taking the value with the best timing.
    2. AlgoRatio = 2; Comparisons where GPU handles 100% or only 1% of the
   workload are timed, and then the optimal workload balance is computed.
    3. AlgoBisection = 3; Based on bisection, multiple workload values are
   tested until the optimal one is found.
 */
#define AUTOTUNING_ALGORITHM 3
/* Recalibrate every X projections. Put to a very high value, i.e., 99999, to de
 * facto disable recalibration */
#define RECALIB_FACTOR 200
/* After how many comparison iterations, comparison duration becomes stable */
#define FIRST_STABLE 7

static inline void *mallocchk(size_t size)
{
  void *ptr = malloc(size);
  if (ptr == 0)
  {
    myError("Memory allocation");
  }
  return (ptr);
}

static inline void *callocchk(size_t size)
{
  void *ptr = calloc(1, size);
  if (ptr == 0)
  {
    myError("Memory allocation and initialization to 0");
  }
  return (ptr);
}

static inline void *reallocchk(void *oldptr, size_t size)
{
  void *ptr = realloc(oldptr, size);
  if (ptr == 0)
  {
    myError("Memory reallocation");
  }
  return (ptr);
}

#ifndef WITH_OPENMP
#define omp_get_max_threads() 1
#define omp_get_thread_num() 0
#endif

extern int mpi_rank;
extern int mpi_size;

#endif
