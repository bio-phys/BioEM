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
    fprintf(stderr, "Error - MPI function %s: %d\n", __FILE__, __LINE__);      \
  }
#endif

#include <fenv.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _WIN32
#include <WinBase.h>
#include <Windows.h>
#endif

#include <algorithm>
#include <iostream>
#include <iterator>

#include "bioem.h"
#include "bioem_cuda.h"

#ifdef WITH_OPENMP
#include <omp.h>
#endif

#ifdef WITH_MPI
int mpi_rank;
int mpi_size;
#else
int mpi_rank = 0;
int mpi_size = 1;
#endif

#include "timer.h"

int main(int argc, char *argv[])
{
  // **************************************************************************************
  // *********************************  Main BioEM code
  // **********************************
  // ************************************************************************************

#ifdef WITH_MPI
  MPI_CHK(MPI_Init(&argc, &argv));
  MPI_CHK(MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank));
  MPI_CHK(MPI_Comm_size(MPI_COMM_WORLD, &mpi_size));
#endif

#ifdef _MM_DENORMALS_ZERO_ON
#pragma omp parallel
  {
    _MM_SET_DENORMALS_ZERO_MODE(
        _MM_DENORMALS_ZERO_ON); // Flush denormals to zero in all OpenMP threads
  }
#endif
  HighResTimer timer;

  bioem *bio;
#ifdef WITH_CUDA
  if (getenv("GPU") && atoi(getenv("GPU")))
  {
    bio = bioem_cuda_create();
  }
  else
#endif
  {
    bio = new bioem;
  }

  // ************  Configuration and Pre-calculating necessary objects
  // *****************
  if (mpi_rank == 0)
    printf("Configuring\n");
  if (bio->configure(argc, argv) == 0)
  {
    if (bio->needToPrintModel())
    {
      if (mpi_size == 1)
      {
        bio->printModel();
      }
      else
      {
        myError("Model printing can be performed only if using a single "
                "MPI process. Please change your execution to use a single MPI "
                "process or no MPI at all");
      }
    }
    else
    {
      // *******************************  Run BioEM routine
      // ******************************
      if (mpi_rank == 0)
        printf("Running\n");
      timer.Start();
      bio->run();
      timer.Stop();

      // ************************************ End
      // **********************************
      printf("The code ran for %f seconds (rank %d).\n", timer.GetElapsedTime(),
             mpi_rank);
      bio->cleanup();
    }
  }
  delete bio;

#ifdef WITH_MPI
  MPI_Finalize();
#endif

  return (0);
}
