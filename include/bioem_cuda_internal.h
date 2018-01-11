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

#ifndef BIOEM_CUDA_INTERNAL_H
#define BIOEM_CUDA_INTERNAL_H

#include <cuda.h>
#include <cufft.h>

// Hack to make nvcc compiler accept fftw.h, float128 is not used anyway
#define __float128 double
#include <fftw3.h>
#undef __float128

#include "bioem_cuda.h"

class bioem_cuda : public bioem
{
public:
  bioem_cuda();
  virtual ~bioem_cuda();

  virtual int compareRefMaps(int iPipeline, int iOrient, int iConv,
                             int maxParallelConv, mycomplex_t *localmultFFT,
                             myparam5_t *comp_params, const int startMap = 0);
  virtual void *malloc_device_host(size_t size);
  virtual void free_device_host(void *ptr);
  virtual void rebalance(int workload); // Rebalance GPUWorkload

protected:
  virtual int deviceInit();
  virtual int deviceStartRun();
  virtual int deviceFinishRun();
  int deviceExit();

private:
  int selectCudaDevice();

  int deviceInitialized;

  cudaStream_t cudaStream[PIPELINE_LVL + 1]; // Streams are used for both
                                             // PIPELINE and MULTISTREAM control
  cudaEvent_t cudaEvent[PIPELINE_LVL + 1];
  cudaEvent_t cudaFFTEvent[MULTISTREAM_LVL];
  bioem_RefMap *gpumap;
  bioem_Probability *pProb_host;
  bioem_Probability pProb_device;
  void *pProb_memory;

  mycomplex_t *pRefMapsFFT;
  mycomplex_t *pConvMapFFT;
  mycomplex_t *pConvMapFFT_Host;
  mycuComplex_t *pFFTtmp2[MULTISTREAM_LVL];
  myfloat_t *pFFTtmp[MULTISTREAM_LVL];
  cufftHandle plan[SPLIT_MAPS_LVL][MULTISTREAM_LVL];

  myparam5_t *pTmp_comp_params;

  myblockGPU_t *pTmp_comp_blocks;
  int Ncomp_blocks;

  bool *initialized_const; // In order to make sure Constoadd is initialized to
                           // the first value

  myfloat_t *sum, *sumsquare;

  int GPUAsync; // Run GPU Asynchronously, do the convolutions on the host in
                // parallel.
  int GPUDualStream; // Use two streams to improve paralelism
  int GPUWorkload;   // Percentage of workload to perform on GPU. Default 100.
                     // Rest is done on processor in parallel.

  int maxRef;
};

#endif
