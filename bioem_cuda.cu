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

#define BIOEM_GPUCODE

#if defined(_WIN32)
#include <windows.h>
#endif

#include <iostream>
using namespace std;

#include "bioem_cuda_internal.h"
//#include "helper_cuda.h"

#include "bioem_algorithm.h"

#define checkCudaErrors(error)                                                 \
  {                                                                            \
    if ((error) != cudaSuccess)                                                \
    {                                                                          \
      printf("CUDA Error %d / %s (%s: %d)\n", error,                           \
             cudaGetErrorString(error), __FILE__, __LINE__);                   \
      exit(1);                                                                 \
    }                                                                          \
  }

#ifdef DEBUG_GPU
#define printCudaDebugStart()                                                  \
  float time;                                                                  \
  time = 0.;                                                                   \
  cudaEvent_t start, stop;                                                     \
  checkCudaErrors(cudaEventCreate(&start));                                    \
  checkCudaErrors(cudaEventCreate(&stop));                                     \
  checkCudaErrors(cudaEventRecord(start, 0));
#define printCudaDebug(msg)                                                    \
  checkCudaErrors(cudaEventRecord(stop, 0));                                   \
  checkCudaErrors(cudaEventSynchronize(stop));                                 \
  checkCudaErrors(cudaEventElapsedTime(&time, start, stop));                   \
  printf("\t\t\tGPU: %s %1.6f sec\n", msg, time / 1000);                       \
  checkCudaErrors(cudaEventRecord(start, 0));

#else
#define printCudaDebugStart()
#define printCudaDebug(msg)
#endif

static const char *cufftGetErrorStrung(cufftResult error)
{
  switch (error)
  {
    case CUFFT_SUCCESS:
      return "CUFFT_SUCCESS";

    case CUFFT_INVALID_PLAN:
      return "CUFFT_INVALID_PLAN";

    case CUFFT_ALLOC_FAILED:
      return "CUFFT_ALLOC_FAILED";

    case CUFFT_INVALID_TYPE:
      return "CUFFT_INVALID_TYPE";

    case CUFFT_INVALID_VALUE:
      return "CUFFT_INVALID_VALUE";

    case CUFFT_INTERNAL_ERROR:
      return "CUFFT_INTERNAL_ERROR";

    case CUFFT_EXEC_FAILED:
      return "CUFFT_EXEC_FAILED";

    case CUFFT_SETUP_FAILED:
      return "CUFFT_SETUP_FAILED";

    case CUFFT_INVALID_SIZE:
      return "CUFFT_INVALID_SIZE";

    case CUFFT_UNALIGNED_DATA:
      return "CUFFT_UNALIGNED_DATA";
  }
  return "UNKNOWN";
}

/* Handing CUDA Driver errors */

#define cuErrorCheck(call)                                                     \
  do                                                                           \
  {                                                                            \
    CUresult __error__;                                                        \
    if ((__error__ = (call)) != CUDA_SUCCESS)                                  \
    {                                                                          \
      printf("CUDA Driver Error %d / %s (%s %d)\n", __error__,                 \
             cuGetError(__error__), __FILE__, __LINE__);                       \
      return __error__;                                                        \
    }                                                                          \
  } while (false)

static const char *cuGetError(CUresult result)
{
  switch (result)
  {
    case CUDA_SUCCESS:
      return "No errors";
    case CUDA_ERROR_INVALID_VALUE:
      return "Invalid value";
    case CUDA_ERROR_OUT_OF_MEMORY:
      return "Out of memory";
    case CUDA_ERROR_NOT_INITIALIZED:
      return "Driver not initialized";
    case CUDA_ERROR_DEINITIALIZED:
      return "Driver deinitialized";
    case CUDA_ERROR_PROFILER_DISABLED:
      return "Profiler disabled";
    case CUDA_ERROR_PROFILER_NOT_INITIALIZED:
      return "Profiler not initialized";
    case CUDA_ERROR_PROFILER_ALREADY_STARTED:
      return "Profiler already started";
    case CUDA_ERROR_PROFILER_ALREADY_STOPPED:
      return "Profiler already stopped";
    case CUDA_ERROR_NO_DEVICE:
      return "No CUDA-capable device available";
    case CUDA_ERROR_INVALID_DEVICE:
      return "Invalid device";
    case CUDA_ERROR_INVALID_IMAGE:
      return "Invalid kernel image";
    case CUDA_ERROR_INVALID_CONTEXT:
      return "Invalid context";
    case CUDA_ERROR_CONTEXT_ALREADY_CURRENT:
      return "Context already current";
    case CUDA_ERROR_MAP_FAILED:
      return "Map failed";
    case CUDA_ERROR_UNMAP_FAILED:
      return "Unmap failed";
    case CUDA_ERROR_ARRAY_IS_MAPPED:
      return "Array is mapped";
    case CUDA_ERROR_ALREADY_MAPPED:
      return "Already mapped";
    case CUDA_ERROR_NO_BINARY_FOR_GPU:
      return "No binary for GPU";
    case CUDA_ERROR_ALREADY_ACQUIRED:
      return "Already acquired";
    case CUDA_ERROR_NOT_MAPPED:
      return "Not mapped";
    case CUDA_ERROR_NOT_MAPPED_AS_ARRAY:
      return "Not mapped as array";
    case CUDA_ERROR_NOT_MAPPED_AS_POINTER:
      return "Not mapped as pointer";
    case CUDA_ERROR_ECC_UNCORRECTABLE:
      return "Uncorrectable ECC error";
    case CUDA_ERROR_UNSUPPORTED_LIMIT:
      return "Unsupported CUlimit";
    case CUDA_ERROR_CONTEXT_ALREADY_IN_USE:
      return "Context already in use";
    case CUDA_ERROR_INVALID_SOURCE:
      return "Invalid source";
    case CUDA_ERROR_FILE_NOT_FOUND:
      return "File not found";
    case CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND:
      return "Shared object symbol not found";
    case CUDA_ERROR_SHARED_OBJECT_INIT_FAILED:
      return "Shared object initialization failed";
    case CUDA_ERROR_OPERATING_SYSTEM:
      return "Operating System call failed";
    case CUDA_ERROR_INVALID_HANDLE:
      return "Invalid handle";
    case CUDA_ERROR_NOT_FOUND:
      return "Not found";
    case CUDA_ERROR_NOT_READY:
      return "CUDA not ready";
    case CUDA_ERROR_LAUNCH_FAILED:
      return "Launch failed";
    case CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES:
      return "Launch exceeded resources";
    case CUDA_ERROR_LAUNCH_TIMEOUT:
      return "Launch exceeded timeout";
    case CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING:
      return "Launch with incompatible texturing";
    case CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED:
      return "Peer access already enabled";
    case CUDA_ERROR_PEER_ACCESS_NOT_ENABLED:
      return "Peer access not enabled";
    case CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE:
      return "Primary context active";
    case CUDA_ERROR_CONTEXT_IS_DESTROYED:
      return "Context is destroyed";
    case CUDA_ERROR_ASSERT:
      return "Device assert failed";
    case CUDA_ERROR_TOO_MANY_PEERS:
      return "Too many peers";
    case CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED:
      return "Host memory already registered";
    case CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED:
      return "Host memory not registered";
    case CUDA_ERROR_UNKNOWN:
      return "Unknown error";
    default:
      return "Unknown error code";
  }
}

bioem_cuda::bioem_cuda()
{
  deviceInitialized = 0;
  GPUAsync = getenv("GPUASYNC") == NULL ? 1 : atoi(getenv("GPUASYNC"));
  GPUWorkload =
      getenv("GPUWORKLOAD") == NULL ? 100 : atoi(getenv("GPUWORKLOAD"));
  if (GPUWorkload == -1)
    GPUWorkload = 100;
  GPUDualStream =
      getenv("GPUDUALSTREAM") == NULL ? 1 : atoi(getenv("GPUDUALSTREAM"));
}

bioem_cuda::~bioem_cuda() { deviceExit(); }

__global__ void multComplexMap(const mycomplex_t *convmap,
                               const mycomplex_t *refmap, mycuComplex_t *out,
                               const int MapSize, const int maxParallelConv,
                               const int NumberRefMaps, const int Offset)
{
  int myConv = myBlockIdxX / NumberRefMaps;
  int myRef = myBlockIdxX - myConv * NumberRefMaps + Offset;
  const mycuComplex_t *myin = (mycuComplex_t *) &refmap[myRef * MapSize];
  const mycuComplex_t *myconv = (mycuComplex_t *) &convmap[myConv * MapSize];
  mycuComplex_t *myout = &out[myBlockIdxX * MapSize];
  for (int i = myThreadIdxX; i < MapSize; i += myBlockDimX)
  {
    mycuComplex_t val;
    const mycuComplex_t conv = myconv[i];
    const mycuComplex_t in = myin[i];

    val.x = conv.x * in.x + conv.y * in.y;
    val.y = conv.y * in.x - conv.x * in.y;
    myout[i] = val;
  }
}

__global__ void
cuDoRefMapsFFT(const int iOrient, const int iConv, const myfloat_t *lCC,
               const myparam5_t *comp_params, bioem_Probability pProb,
               const bioem_param_device param, const bioem_RefMap RefMap,
               const int maxRef, const int Offset)
{
  if (myBlockIdxX * myBlockDimX + myThreadIdxX >= maxRef)
    return;
  const int iRefMap = myBlockIdxX * myBlockDimX + myThreadIdxX + Offset;
  const myfloat_t *mylCC = &lCC[(myBlockIdxX * myBlockDimX + myThreadIdxX) *
                                param.NumberPixels * param.NumberPixels];
  doRefMapFFT(iRefMap, iOrient, iConv, comp_params->amp, comp_params->pha,
              comp_params->env, comp_params->sumC, comp_params->sumsquareC,
              mylCC, pProb, param, RefMap);
}

__global__ void
doRefMap_GPU_Parallel(const int iRefMap, const int iOrient, const int iConv,
                      const int maxParallelConv, const myfloat_t *lCC,
                      const myparam5_t *comp_params, myblockGPU_t *comp_block,
                      bioem_Probability pProb, const bioem_param_device param,
                      const bioem_RefMap RefMap, const int maxRef,
                      const int dispC)
{
  int myGlobalId = myBlockIdxX * myBlockDimX + myThreadIdxX;
  if (myGlobalId >= maxParallelConv * param.NtotDisp)
    return;
  int myConv = myGlobalId / param.NtotDisp;
  myGlobalId -= myConv * param.NtotDisp;
  int myX = myGlobalId / param.NxDisp;
  myGlobalId -= myX * param.NxDisp;
  int myY = myGlobalId;
  myGlobalId = myBlockIdxX * myBlockDimX + myThreadIdxX;

  int cent_x = (myX * param.GridSpaceCenter + dispC) % param.NumberPixels;
  int cent_y = (myY * param.GridSpaceCenter + dispC) % param.NumberPixels;
  int address = (myConv * maxRef * param.NumberPixels * param.NumberPixels) +
                (cent_x * param.NumberPixels + cent_y);
  myfloat_t value = (myfloat_t) lCC[address] /
                    (myfloat_t)(param.NumberPixels * param.NumberPixels);

  __shared__ myprob_t bestLogpro[CUDA_THREAD_MAX];
  __shared__ int bestId[CUDA_THREAD_MAX];
  __shared__ myprob_t sumExp[CUDA_THREAD_MAX];
  __shared__ myprob_t sumAngles[CUDA_THREAD_MAX];

  int nTotalThreads =
      ((maxParallelConv * param.NtotDisp) < ((myBlockIdxX + 1) * myBlockDimX)) ?
          ((maxParallelConv * param.NtotDisp) - (myBlockIdxX * myBlockDimX)) :
          myBlockDimX;
  int halfPoint = (nTotalThreads + 1) >> 1; // divide by two

  bioem_Probability_map &pProbMap = pProb.getProbMap(iRefMap);

  bestLogpro[myThreadIdxX] =
      calc_logpro(param, comp_params[myConv].amp, comp_params[myConv].pha,
                  comp_params[myConv].env, comp_params[myConv].sumC,
                  comp_params[myConv].sumsquareC, value,
                  RefMap.sum_RefMap[iRefMap], RefMap.sumsquare_RefMap[iRefMap]);
#ifdef DEBUG_PROB
  printf("\t\t\tProb: iRefMap %d, iOrient %d, iConv %d, "
         "cent_x %d, cent_y %d, address %d, value %f, logpro %f\n",
         iRefMap, iOrient, iConv, cent_x, cent_y, address, value,
         bestLogpro[myThreadIdxX]);
#endif
  bestId[myThreadIdxX] = myGlobalId;
  sumExp[myThreadIdxX] = exp(bestLogpro[myThreadIdxX] - pProbMap.Constoadd);
  if (param.writeAngles)
  {
    bioem_Probability_angle &pProbAngle = pProb.getProbAngle(iRefMap, iOrient);
    sumAngles[myThreadIdxX] =
        exp(bestLogpro[myThreadIdxX] - pProbAngle.ConstAngle);
  }
  __syncthreads();

  // Total number of active threads
  while (nTotalThreads > 1)
  {
    if (myThreadIdxX < (nTotalThreads >> 1))
    {
      // Get the shared value stored by another thread
      myprob_t temp = bestLogpro[myThreadIdxX + halfPoint];
      if (temp > bestLogpro[myThreadIdxX])
      {
        bestLogpro[myThreadIdxX] = temp;
        bestId[myThreadIdxX] = bestId[myThreadIdxX + halfPoint];
      }
      sumExp[myThreadIdxX] += sumExp[myThreadIdxX + halfPoint];
      if (param.writeAngles)
      {
        sumAngles[myThreadIdxX] += sumAngles[myThreadIdxX + halfPoint];
      }
    }
    __syncthreads();
    nTotalThreads = halfPoint;            // divide by two.
    halfPoint = (nTotalThreads + 1) >> 1; // divide by two
    // only the first half of the threads will be active.
  }
  if (myThreadIdxX == 0)
  {
    comp_block[myBlockIdxX].logpro = bestLogpro[0];
    comp_block[myBlockIdxX].id = bestId[0];
    comp_block[myBlockIdxX].sumExp = sumExp[0];
    if (param.writeAngles)
    {
      comp_block[myBlockIdxX].sumAngles = sumAngles[0];
    }
#ifdef DEBUG_PROB
    printf("\t\t\tProb block: iRefMap %d, iOrient %d, iConv %d, "
           "bestlogpro %f, bestId %d, sumExp %f\n",
           iRefMap, iOrient, iConv, bestLogpro[0], bestId[0], sumExp[0]);
#endif
  }
}

__global__ void
doRefMap_GPU_Reduce(const int iRefMap, const int iOrient, const int iConv,
                    const int maxParallelConv, const myfloat_t *lCC,
                    const myparam5_t *comp_params,
                    const myblockGPU_t *comp_block, bioem_Probability pProb,
                    const bioem_param_device param, const bioem_RefMap RefMap,
                    const int maxRef, const int dispC)
{

  __shared__ myprob_t bestLogpro[CUDA_THREAD_MAX];
  __shared__ int bestId[CUDA_THREAD_MAX];
  __shared__ myprob_t sumExp[CUDA_THREAD_MAX];
  __shared__ myprob_t sumAngles[CUDA_THREAD_MAX];

  // if it is the last block
  int nTotalThreads = myBlockDimX;
  int halfPoint = (nTotalThreads + 1) >> 1; // divide by two

  bioem_Probability_map &pProbMap = pProb.getProbMap(iRefMap);

  bestLogpro[myThreadIdxX] = comp_block[myThreadIdxX].logpro;
  bestId[myThreadIdxX] = comp_block[myThreadIdxX].id;
  sumExp[myThreadIdxX] = comp_block[myThreadIdxX].sumExp;
  if (param.writeAngles)
  {
    sumAngles[myThreadIdxX] = comp_block[myThreadIdxX].sumAngles;
  }
  __syncthreads();
  while (nTotalThreads > 1)
  {
    if (myThreadIdxX < (nTotalThreads >> 1))
    {
      // Get the shared value stored by another thread
      myfloat_t temp = bestLogpro[myThreadIdxX + halfPoint];
      if (temp > bestLogpro[myThreadIdxX])
      {
        bestLogpro[myThreadIdxX] = temp;
        bestId[myThreadIdxX] = bestId[myThreadIdxX + halfPoint];
      }
      sumExp[myThreadIdxX] += sumExp[myThreadIdxX + halfPoint];
      if (param.writeAngles)
      {
        sumAngles[myThreadIdxX] += sumAngles[myThreadIdxX + halfPoint];
      }
    }
    __syncthreads();
    nTotalThreads = halfPoint;            // divide by two.
    halfPoint = (nTotalThreads + 1) >> 1; // divide by two
    // only the first half of the threads will be active.
  }

  if (myThreadIdxX == 0)
  {
    pProbMap.Total += sumExp[0];
    if (pProbMap.Constoadd < bestLogpro[0])
    {
      pProbMap.Total *= exp(-bestLogpro[0] + pProbMap.Constoadd);
      pProbMap.Constoadd = bestLogpro[0];

      // ********** Getting parameters that maximize the probability ***********
      int myGlobalId = bestId[0];
      int myConv = myGlobalId / param.NtotDisp;
      myGlobalId -= myConv * param.NtotDisp;
      int myX = myGlobalId / param.NxDisp;
      myGlobalId -= myX * param.NxDisp;
      int myY = myGlobalId;

      int cent_x = (myX * param.GridSpaceCenter + dispC) % param.NumberPixels;
      int cent_y = (myY * param.GridSpaceCenter + dispC) % param.NumberPixels;
      int address =
          (myConv * maxRef * param.NumberPixels * param.NumberPixels) +
          (cent_x * param.NumberPixels + cent_y);
      myfloat_t value = (myfloat_t) lCC[address] /
                        (myfloat_t)(param.NumberPixels * param.NumberPixels);

      pProbMap.max.max_prob_cent_x =
          -((myX * param.GridSpaceCenter + dispC) - param.NumberPixels);
      pProbMap.max.max_prob_cent_y =
          -((myY * param.GridSpaceCenter + dispC) - param.NumberPixels);
      pProbMap.max.max_prob_orient = iOrient;
      pProbMap.max.max_prob_conv = iConv + myConv;
      pProbMap.max.max_prob_norm =
          -(-comp_params[myConv].sumC * RefMap.sum_RefMap[iRefMap] +
            param.Ntotpi * value) /
          (comp_params[myConv].sumC * comp_params[myConv].sumC -
           comp_params[myConv].sumsquareC * param.Ntotpi);
      pProbMap.max.max_prob_mu =
          -(-comp_params[myConv].sumC * value +
            comp_params[myConv].sumsquareC * RefMap.sum_RefMap[iRefMap]) /
          (comp_params[myConv].sumC * comp_params[myConv].sumC -
           comp_params[myConv].sumsquareC * param.Ntotpi);

#ifdef DEBUG_PROB
      printf("\tProbabilities change: iRefMap %d, iOrient %d, iConv %d, "
             "Total %f, Const %f, bestlogpro %f, sumExp %f, bestId %d\n",
             iRefMap, iOrient, iConv + myConv, pProbMap.Total,
             pProbMap.Constoadd, bestLogpro[0], sumExp[0], bestId[0]);
      printf("\tParameters: myConv %d, myX %d, myY %d, cent_x %d, cent_y %d, "
             "probX %d, probY %d\n",
             myConv, myX, myY, cent_x, cent_y, pProbMap.max.max_prob_cent_x,
             pProbMap.max.max_prob_cent_y);
#endif
    }
#ifdef DEBUG_PROB
    printf("\t\tProbabilities after Reduce: iRefMap %d, iOrient %d, iConv "
           "%d, Total %f, Const %f, bestlogpro %f, sumExp %f, bestId %d\n",
           iRefMap, iOrient, iConv, pProbMap.Total, pProbMap.Constoadd,
           bestLogpro[0], sumExp[0], bestId[0]);
#endif

    if (param.writeAngles)
    {
      bioem_Probability_angle &pProbAngle =
          pProb.getProbAngle(iRefMap, iOrient);
      pProbAngle.forAngles += sumAngles[0];
      if (pProbAngle.ConstAngle < bestLogpro[0])
      {
        pProbAngle.forAngles *= exp(-bestLogpro[0] + pProbAngle.ConstAngle);
        pProbAngle.ConstAngle = bestLogpro[0];
      }
    }
  }
}

__global__ void
init_Constoadd(const int iRefMap, const int iOrient, const myfloat_t *lCC,
               const myparam5_t *comp_params, bioem_Probability pProb,
               const bioem_param_device param, const bioem_RefMap RefMap,
               const int initialized_const)
{
  myfloat_t value =
      (myfloat_t) lCC[0] / (myfloat_t)(param.NumberPixels * param.NumberPixels);

  myfloat_t logpro =
      calc_logpro(param, comp_params->amp, comp_params->pha, comp_params->env,
                  comp_params->sumC, comp_params->sumsquareC, value,
                  RefMap.sum_RefMap[iRefMap], RefMap.sumsquare_RefMap[iRefMap]);

  bioem_Probability_map &pProbMap = pProb.getProbMap(iRefMap);

  // Needed only once, in the first projection
  if (!initialized_const)
  {
    pProbMap.Constoadd = logpro;
  }
  // Needed for every projection
  if (param.writeAngles)
  {
    bioem_Probability_angle &pProbAngle = pProb.getProbAngle(iRefMap, iOrient);
    pProbAngle.ConstAngle = logpro;
  }

#ifdef DEBUG_GPU
  printf("\tInitialized pProbMap.Constoadd of refmap %d to %f\n", iRefMap,
         pProbMap.Constoadd);
#endif
}

template <class T> static inline T divup(T num, T divider)
{
  return ((num + divider - 1) / divider);
}

int bioem_cuda::compareRefMaps(int iPipeline, int iOrient, int iConv,
                               int maxParallelConv, mycomplex_t *conv_mapsFFT,
                               myparam5_t *comp_params, const int startMap)
{
  if (startMap)
  {
    cout << "Error startMap not implemented for GPU Code\n";
    exit(1);
  }
  printCudaDebugStart();
  if (GPUAsync)
  {
    checkCudaErrors(cudaEventSynchronize(cudaEvent[iPipeline & 1]));
    printCudaDebug("time to synch projections");
  }

  int k = (iPipeline & 1) * param.nTotParallelConv;
  memcpy(&pConvMapFFT_Host[k * param.FFTMapSize],
         conv_mapsFFT[k * param.FFTMapSize],
         param.FFTMapSize * maxParallelConv * sizeof(mycomplex_t));
  printCudaDebug("time for memcpy");
  checkCudaErrors(
      cudaMemcpyAsync(&pConvMapFFT[k * param.FFTMapSize],
                      &pConvMapFFT_Host[k * param.FFTMapSize],
                      param.FFTMapSize * maxParallelConv * sizeof(mycomplex_t),
                      cudaMemcpyHostToDevice, cudaStream[GPUAsync ? 2 : 0]));
  // If one wants just a single tranfer, without memcpy:
  // checkCudaErrors(cudaMemcpyAsync(&pConvMapFFT[k * param.FFTMapSize],
  // conv_mapsFFT[k * param.FFTMapSize], param.FFTMapSize * maxParallelConv *
  // sizeof(mycomplex_t), cudaMemcpyHostToDevice, cudaStream[GPUAsync ? 2 :
  // 0]));
  checkCudaErrors(cudaMemcpyAsync(&pTmp_comp_params[k], &comp_params[k],
                                  maxParallelConv * sizeof(myparam5_t),
                                  cudaMemcpyHostToDevice,
                                  cudaStream[GPUAsync ? 2 : 0]));
  printCudaDebug("time for asyncmemcpy");
  if (GPUAsync)
  {
    checkCudaErrors(cudaEventRecord(cudaEvent[2], cudaStream[2]));
    checkCudaErrors(cudaStreamWaitEvent(cudaStream[0], cudaEvent[2], 0));
  }
  if (GPUDualStream)
  {
    checkCudaErrors(cudaEventRecord(cudaFFTEvent[0], cudaStream[0]));
    checkCudaErrors(cudaStreamWaitEvent(cudaStream[1], cudaFFTEvent[0], 0));
  }
  for (int offset = 0, stream = 0; offset < maxRef;
       offset += param.nTotParallelMaps, stream++)
  {
    if (!GPUDualStream)
      stream = 0;
    const int nRef = min(param.nTotParallelMaps, maxRef - offset);
    multComplexMap<<<maxParallelConv * nRef, CudaThreadCount, 0,
                     cudaStream[stream & 1]>>>(
        &pConvMapFFT[k * param.FFTMapSize], pRefMapsFFT, pFFTtmp2[stream & 1],
        param.FFTMapSize, maxParallelConv, nRef, offset);
    printCudaDebug("time for multComplexMap kernel");
    cufftResult err = mycufftExecC2R(offset + param.nTotParallelMaps > maxRef ?
                                         plan[1][stream & 1] :
                                         plan[0][stream & 1],
                                     pFFTtmp2[stream & 1], pFFTtmp[stream & 1]);
    if (err != CUFFT_SUCCESS)
    {
      cout << "Error running CUFFT " << cufftGetErrorStrung(err) << "\n";
      exit(1);
    }
    printCudaDebug("time for mycufftExecC2R kernel");
    if (BioEMAlgo == 1)
    {
      for (int conv = 0; conv < maxParallelConv; conv++)
      {
        cuDoRefMapsFFT<<<divup(nRef, CudaThreadCount), CudaThreadCount, 0,
                         cudaStream[stream & 1]>>>(
            iOrient, iConv + conv,
            pFFTtmp[stream & 1] +
                conv * nRef * param.param_device.NumberPixels *
                    param.param_device.NumberPixels,
            &pTmp_comp_params[k + conv], pProb_device, param.param_device,
            *gpumap, nRef, offset);
        printCudaDebug("time for cuDoRefMapsFFT kernel");
      }
    }
    else
    {
      for (int refmap = offset; refmap < nRef + offset; refmap++)
      {
        // First iteration needs to initialize Constoadd with the first valid
        // value to avoid overflow due to high sumExp values
        if ((initialized_const[refmap] == false) ||
            (param.param_device.writeAngles && iConv == 0))
        {
          init_Constoadd<<<1, 1, 0, cudaStream[stream & 1]>>>(
              refmap, iOrient,
              pFFTtmp[stream & 1] +
                  (refmap - offset) * param.param_device.NumberPixels *
                      param.param_device.NumberPixels,
              &pTmp_comp_params[k], pProb_device, param.param_device, *gpumap,
              (int) initialized_const[refmap]);
          initialized_const[refmap] = true;
          printCudaDebug("time for init_Constoadd kernel");
        }

        doRefMap_GPU_Parallel<<<divup(maxParallelConv *
                                          param.param_device.NtotDisp,
                                      CudaThreadCount),
                                CudaThreadCount, 0, cudaStream[stream & 1]>>>(
            refmap, iOrient, iConv, maxParallelConv,
            pFFTtmp[stream & 1] +
                (refmap - offset) * param.param_device.NumberPixels *
                    param.param_device.NumberPixels,
            &pTmp_comp_params[k], &pTmp_comp_blocks[refmap * Ncomp_blocks],
            pProb_device, param.param_device, *gpumap, nRef,
            param.param_device.NumberPixels -
                param.param_device.maxDisplaceCenter);
        printCudaDebug("time for doRefMaps_GPU_Parallel kernel");

        doRefMap_GPU_Reduce<<<1, divup(maxParallelConv *
                                           param.param_device.NtotDisp,
                                       CudaThreadCount),
                              0, cudaStream[stream & 1]>>>(
            refmap, iOrient, iConv, maxParallelConv,
            pFFTtmp[stream & 1] +
                (refmap - offset) * param.param_device.NumberPixels *
                    param.param_device.NumberPixels,
            &pTmp_comp_params[k], &pTmp_comp_blocks[refmap * Ncomp_blocks],
            pProb_device, param.param_device, *gpumap, nRef,
            param.param_device.NumberPixels -
                param.param_device.maxDisplaceCenter);
        printCudaDebug("time for doRefMaps_GPU_Reduce kernel");
      }
    }
  }
  checkCudaErrors(cudaPeekAtLastError());

  if (GPUDualStream)
  {
    checkCudaErrors(cudaEventRecord(cudaFFTEvent[1], cudaStream[1]));
    checkCudaErrors(cudaStreamWaitEvent(cudaStream[0], cudaFFTEvent[1], 0));
  }

  if ((BioEMAlgo == 1) && (GPUWorkload < 100))
  {
    bioem::compareRefMaps(iPipeline, iOrient, iConv, maxParallelConv,
                          conv_mapsFFT, comp_params, maxRef);
    printCudaDebug("time to run OMP");
  }
  if (GPUAsync)
  {
    checkCudaErrors(cudaEventRecord(cudaEvent[iPipeline & 1], cudaStream[0]));
  }
  else
  {
    checkCudaErrors(cudaStreamSynchronize(cudaStream[0]));
    printCudaDebug("time to synch at the end");
  }
  return (0);
}

int bioem_cuda::selectCudaDevice()
{
  int count;
  int bestDevice = 0;
  cudaDeviceProp deviceProp;

  /* Initializing CUDA driver API */
  cuErrorCheck(cuInit(0));

  /* Get number of available CUDA devices */
  checkCudaErrors(cudaGetDeviceCount(&count));
  if (count == 0)
  {
    printf("No CUDA device detected\n");
    return (1);
  }

  /* Find the best GPU */
  long long int bestDeviceSpeed = -1, deviceSpeed = -1;
  for (int i = 0; i < count; i++)
  {
    cudaGetDeviceProperties(&deviceProp, i);
    deviceSpeed = (long long int) deviceProp.multiProcessorCount *
                  (long long int) deviceProp.clockRate *
                  (long long int) deviceProp.warpSize;
    if (deviceSpeed > bestDeviceSpeed)
    {
      bestDevice = i;
      bestDeviceSpeed = deviceSpeed;
    }
  }

  /* Get user-specified GPU choice */
  if (getenv("GPUDEVICE"))
  {
    int device = atoi(getenv("GPUDEVICE"));
    if (device > count)
    {
      printf("Invalid CUDA device specified, max device number is %d\n", count);
      exit(1);
    }
#ifdef WITH_MPI
    if (device == -1)
    {
      device = mpi_rank % count;
    }
#endif
    if (device < 0)
    {
      printf("Negative CUDA device specified: %d, invalid!\n", device);
      exit(1);
    }
    bestDevice = device;
  }

  /* Set CUDA processes to appropriate devices */
  cudaGetDeviceProperties(&deviceProp, bestDevice);
  if (deviceProp.computeMode == 0)
  {
    checkCudaErrors(cudaSetDevice(bestDevice));
  }
  else
  {
    if (DebugOutput >= 1)
    {
      printf("CUDA device %d is not set in DEFAULT mode, make sure that CUDA "
             "processes are pinned as planned!\n",
             bestDevice);
      printf("Pinning process %d to CUDA device %d\n", mpi_rank, bestDevice);
    }
    checkCudaErrors(cudaSetDevice(bestDevice));
    /* This synchronization is needed in order to detect bogus silent errors
     * from cudaSetDevice call */
    checkCudaErrors(cudaDeviceSynchronize());
  }

  /* Debugging information about CUDA devices used by the current process */
  if (DebugOutput >= 2)
  {
    printf("Using CUDA Device %s with Properties:\n", deviceProp.name);
    printf("totalGlobalMem = %lld\n",
           (unsigned long long int) deviceProp.totalGlobalMem);
    printf("sharedMemPerBlock = %lld\n",
           (unsigned long long int) deviceProp.sharedMemPerBlock);
    printf("regsPerBlock = %d\n", deviceProp.regsPerBlock);
    printf("warpSize = %d\n", deviceProp.warpSize);
    printf("memPitch = %lld\n", (unsigned long long int) deviceProp.memPitch);
    printf("maxThreadsPerBlock = %d\n", deviceProp.maxThreadsPerBlock);
    printf("maxThreadsDim = %d %d %d\n", deviceProp.maxThreadsDim[0],
           deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
    printf("maxGridSize = %d %d %d\n", deviceProp.maxGridSize[0],
           deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
    printf("totalConstMem = %lld\n",
           (unsigned long long int) deviceProp.totalConstMem);
    printf("major = %d\n", deviceProp.major);
    printf("minor = %d\n", deviceProp.minor);
    printf("clockRate = %d\n", deviceProp.clockRate);
    printf("memoryClockRate = %d\n", deviceProp.memoryClockRate);
    printf("multiProcessorCount = %d\n", deviceProp.multiProcessorCount);
    printf("textureAlignment = %lld\n",
           (unsigned long long int) deviceProp.textureAlignment);
    printf("computeMode = %d\n", deviceProp.computeMode);
#if CUDA_VERSION > 3010
    size_t free, total;
#else
    unsigned int free, total;
#endif
    if (deviceProp.computeMode == 0)
    {
      CUdevice tmpDevice;
      cuErrorCheck(cuDeviceGet(&tmpDevice, bestDevice));
      CUcontext tmpContext;
      cuErrorCheck(cuCtxCreate(&tmpContext, 0, tmpDevice));
      cuErrorCheck(cuMemGetInfo(&free, &total));
      cuErrorCheck(cuCtxDestroy(tmpContext));
    }
    else
    {
      cuErrorCheck(cuMemGetInfo(&free, &total));
    }
    printf("free memory = %lld; total memory = %lld\n", free, total);
  }

  if (DebugOutput >= 1)
  {
    printf("BioEM for CUDA initialized (MPI Rank %d), %d GPUs found, using GPU "
           "%d\n",
           mpi_rank, count, bestDevice);
  }

  return (0);
}

int bioem_cuda::deviceInit()
{
  deviceExit();

  selectCudaDevice();

  gpumap = new bioem_RefMap;
  memcpy(gpumap, &RefMap, sizeof(bioem_RefMap));

  checkCudaErrors(cudaMalloc(&sum, sizeof(myfloat_t) * RefMap.ntotRefMap));
  checkCudaErrors(cudaMemcpy(sum, RefMap.sum_RefMap,
                             sizeof(myfloat_t) * RefMap.ntotRefMap,
                             cudaMemcpyHostToDevice));
  checkCudaErrors(
      cudaMalloc(&sumsquare, sizeof(myfloat_t) * RefMap.ntotRefMap));
  checkCudaErrors(cudaMemcpy(sumsquare, RefMap.sumsquare_RefMap,
                             sizeof(myfloat_t) * RefMap.ntotRefMap,
                             cudaMemcpyHostToDevice));
  gpumap->sum_RefMap = sum;
  gpumap->sumsquare_RefMap = sumsquare;

  checkCudaErrors(
      cudaMalloc(&pProb_memory,
                 pProb_device.get_size(RefMap.ntotRefMap, param.nTotGridAngles,
                                       param.param_device.writeAngles)));

  for (int i = 0; i < PIPELINE_LVL; i++)
  {
    checkCudaErrors(cudaStreamCreate(&cudaStream[i]));
    checkCudaErrors(cudaEventCreate(&cudaEvent[i]));
  }
  for (int i = 0; i < MULTISTREAM_LVL; i++)
  {
    checkCudaErrors(cudaEventCreate(&cudaFFTEvent[i]));
  }
  if (GPUAsync)
  {
    checkCudaErrors(cudaStreamCreate(&cudaStream[2]));
    checkCudaErrors(cudaEventCreate(&cudaEvent[2]));
  }

  checkCudaErrors(
      cudaMalloc(&pRefMapsFFT,
                 RefMap.ntotRefMap * param.FFTMapSize * sizeof(mycomplex_t)));
  checkCudaErrors(
      cudaMalloc(&pFFTtmp2[0], param.nTotParallelConv * param.nTotParallelMaps *
                                   param.FFTMapSize * MULTISTREAM_LVL *
                                   sizeof(mycomplex_t)));
  checkCudaErrors(
      cudaMalloc(&pFFTtmp[0], param.nTotParallelConv * param.nTotParallelMaps *
                                  param.param_device.NumberPixels *
                                  param.param_device.NumberPixels *
                                  MULTISTREAM_LVL * sizeof(myfloat_t)));
  for (int i = 1; i < MULTISTREAM_LVL; i++)
  {
    pFFTtmp2[i] =
        pFFTtmp2[0] +
        i * param.nTotParallelConv * param.nTotParallelMaps * param.FFTMapSize;
    pFFTtmp[i] = pFFTtmp[0] +
                 i * param.nTotParallelConv * param.nTotParallelMaps *
                     param.param_device.NumberPixels *
                     param.param_device.NumberPixels;
  }
  checkCudaErrors(cudaMalloc(&pConvMapFFT, param.nTotParallelConv *
                                               param.FFTMapSize * PIPELINE_LVL *
                                               sizeof(mycomplex_t)));
  checkCudaErrors(cudaHostAlloc(&pConvMapFFT_Host,
                                param.nTotParallelConv * param.FFTMapSize *
                                    PIPELINE_LVL * sizeof(mycomplex_t),
                                0));
  checkCudaErrors(
      cudaMemcpy(pRefMapsFFT, RefMap.RefMapsFFT,
                 RefMap.ntotRefMap * param.FFTMapSize * sizeof(mycomplex_t),
                 cudaMemcpyHostToDevice));
  checkCudaErrors(
      cudaMalloc(&pTmp_comp_params,
                 param.nTotParallelConv * PIPELINE_LVL * sizeof(myparam5_t)));
  Ncomp_blocks = divup(param.nTotParallelConv * param.param_device.NtotDisp,
                       CudaThreadCount);
  if (Ncomp_blocks > CudaThreadCount)
  {
    cout << "Error with input parameters. Check CudaThreadCount, "
            "displacements and max number of parallel comparisons\n";
    exit(1);
  }
  checkCudaErrors(
      cudaMalloc(&pTmp_comp_blocks,
                 Ncomp_blocks * RefMap.ntotRefMap * sizeof(myblockGPU_t)));

  initialized_const = new bool[RefMap.ntotRefMap];
  for (int i = 0; i < RefMap.ntotRefMap; i++)
    initialized_const[i] = false;

  deviceInitialized = 1;
  return (0);
}

int bioem_cuda::deviceExit()
{
  if (deviceInitialized == 0)
    return (0);

  cudaFree(pProb_memory);
  cudaFree(sum);
  cudaFree(sumsquare);
  for (int i = 0; i < PIPELINE_LVL; i++)
  {
    cudaStreamDestroy(cudaStream[i]);
    cudaEventDestroy(cudaEvent[i]);
  }
  for (int i = 0; i < MULTISTREAM_LVL; i++)
  {
    cudaEventDestroy(cudaFFTEvent[i]);
  }

  cudaFree(pRefMapsFFT);
  cudaFree(pConvMapFFT);
  cudaFreeHost(pConvMapFFT_Host);
  cudaFree(pFFTtmp[0]);
  cudaFree(pFFTtmp2[0]);
  cudaFree(pTmp_comp_params);
  cudaFree(pTmp_comp_blocks);

  if (GPUAsync)
  {
    cudaStreamDestroy(cudaStream[2]);
    cudaEventDestroy(cudaEvent[2]);
  }

  delete gpumap;
  delete initialized_const;
  cudaDeviceReset();

  deviceInitialized = 0;
  return (0);
}

int bioem_cuda::deviceStartRun()
{
  if (GPUWorkload >= 100)
  {
    maxRef = RefMap.ntotRefMap;
    pProb_host = &pProb;
  }
  else
  {
    maxRef = ((size_t) RefMap.ntotRefMap * (size_t) GPUWorkload / 100) < 1 ?
                 (size_t) RefMap.ntotRefMap :
                 (size_t) RefMap.ntotRefMap * (size_t) GPUWorkload / 100;
    pProb_host = new bioem_Probability;
    pProb_host->init(maxRef, param.nTotGridAngles, *this);
    pProb_host->copyFrom(&pProb, *this);
  }

  pProb_device = *pProb_host;
  pProb_device.ptr = pProb_memory;
  pProb_device.set_pointers();
  checkCudaErrors(
      cudaMemcpyAsync(pProb_device.ptr, pProb_host->ptr,
                      pProb_host->get_size(maxRef, param.nTotGridAngles,
                                           param.param_device.writeAngles),
                      cudaMemcpyHostToDevice, cudaStream[0]));

  if (maxRef / (param.nTotParallelMaps * param.nTotParallelConv) >
      (double) SPLIT_MAPS_LVL)
  {
    cout << "Error planning CUFFT dimensions\n";
    exit(1);
  }
  for (int j = 0; j < MULTISTREAM_LVL; j++)
  {
    for (int i = 0; i < SPLIT_MAPS_LVL; i++)
    {
      if (i && maxRef % param.nTotParallelMaps == 0)
        continue;
      int n[2] = {param.param_device.NumberPixels,
                  param.param_device.NumberPixels};
      if (cufftPlanMany(
              &plan[i][j], 2, n, NULL, 1, param.FFTMapSize, NULL, 1, 0,
              MY_CUFFT_C2R,
              i ? ((maxRef % param.nTotParallelMaps) * param.nTotParallelConv) :
                  (param.nTotParallelMaps * param.nTotParallelConv)) !=
          CUFFT_SUCCESS)
      {
        cout << "Error planning CUFFT\n";
        exit(1);
      }
      if (cufftSetStream(plan[i][j], cudaStream[j]) != CUFFT_SUCCESS)
      {
        cout << "Error setting CUFFT stream\n";
        exit(1);
      }
    }
    if (!GPUDualStream)
      break;
  }

  return (0);
}

int bioem_cuda::deviceFinishRun()
{
  if (GPUAsync)
    cudaStreamSynchronize(cudaStream[0]);
  checkCudaErrors(
      cudaMemcpyAsync(pProb_host->ptr, pProb_device.ptr,
                      pProb_host->get_size(maxRef, param.nTotGridAngles,
                                           param.param_device.writeAngles),
                      cudaMemcpyDeviceToHost, cudaStream[0]));

  for (int j = 0; j < MULTISTREAM_LVL; j++)
  {
    for (int i = 0; i < SPLIT_MAPS_LVL; i++)
    {
      if (i && maxRef % param.nTotParallelMaps == 0)
        continue;
      cufftDestroy(plan[i][j]);
    }
    if (!GPUDualStream)
      break;
  }

  cudaDeviceSynchronize();
  if (GPUWorkload < 100)
  {
    pProb.copyFrom(pProb_host, *this);
    free_device_host(pProb_host->ptr);
    delete[] pProb_host;
  }

  return (0);
}

void *bioem_cuda::malloc_device_host(size_t size)
{
  void *ptr;
  checkCudaErrors(cudaHostAlloc(&ptr, size, 0));
  return (ptr);
}

void bioem_cuda::free_device_host(void *ptr) { cudaFreeHost(ptr); }

void bioem_cuda::rebalance(int workload)
{
  if ((workload < 0) || (workload > 100) || (workload == GPUWorkload))
    return;

  deviceFinishRun();

  if (DebugOutput >= 2)
  {
    printf("\t\tSetting GPU workload to %d%% (rank %d)\n", workload, mpi_rank);
  }

  GPUWorkload = workload;
  maxRef = (size_t) RefMap.ntotRefMap * (size_t) GPUWorkload / 100;

  deviceStartRun();
}

bioem *bioem_cuda_create()
{
  int count;

  if (cudaGetDeviceCount(&count) != cudaSuccess)
    count = 0;
  if (count == 0)
  {
    printf("No CUDA device available, using fallback to CPU version\n");
    return new bioem;
  }

  return new bioem_cuda;
}
