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

#ifndef BIOEM_H
#define BIOEM_H

#include "bioem.h"
#include "defs.h"
#include "map.h"
#include "model.h"
#include "param.h"

class bioem
{
  friend class bioem_RefMap;
  friend class bioem_Probability;

public:
  bioem();
  virtual ~bioem();

  void printOptions(myoption_t *myoptions, int myoptions_length);
  int readOptions(int ac, char *av[]);
  int configure(int ac, char *av[]);
  void cleanup(); // Cleanup everything happening during configure

  int precalculate(); // Is it better to pass directly the input File names?
  inline int needToPrintModel() { return param.printModel; }
  int printModel();
  int run();
  int doProjections(int iMap);
  int createConvolutedProjectionMap(int iOreint, int iMap, mycomplex_t *lproj,
                                    mycomplex_t *localmultFFT, myfloat_t &sumC,
                                    myfloat_t &sumsquareC);
  int createConvolutedProjectionMap_noFFT(mycomplex_t *lproj,
                                          myfloat_t *Mapconv,
                                          mycomplex_t *localmultFFT,
                                          myfloat_t &sumC,
                                          myfloat_t &sumsquareC);

  virtual int compareRefMaps(int iPipeline, int iOrient, int iConv,
                             int maxParallelConv, mycomplex_t *localmultFFT,
                             myparam5_t *comp_params, const int startMap = 0);

  virtual void *malloc_device_host(size_t size);
  virtual void free_device_host(void *ptr);
  virtual void rebalance(int workload); // Rebalance GPUWorkload
  void rebalanceWrapper(int workload);  // Rebalance wrapper

  int createProjection(int iMap, mycomplex_t *map);
  int calcross_cor(myfloat_t *localmap, myfloat_t &sum, myfloat_t &sumsquare);
  void calculateCCFFT(int iMap, mycomplex_t *localConvFFT,
                      mycomplex_t *localCCT, myfloat_t *lCC);
  void doRefMap_CPU_Parallel(int iRefMap, int iOrient, int iConv,
                             myfloat_t *lCC, myparam5_t *comp_params,
                             myblockCPU_t *comp_block);
  void doRefMap_CPU_Reduce(int iRefMap, int iOrient, int iConvStart,
                           int maxParallelConv, myparam5_t *comp_params,
                           myblockCPU_t *comp_block);

  bioem_Probability pProb;

  string OutfileName;

protected:
  virtual int deviceInit();
  virtual int deviceStartRun();
  virtual int deviceFinishRun();

  bioem_param param;
  bioem_model Model;
  bioem_RefMap RefMap;

  int nReferenceMaps;      // Maps in memory at a time
  int nReferenceMapsTotal; // Maps in total

  int nProjectionMaps;      // Maps in memory at a time
  int nProjectionMapsTotal; // Maps in total

  int BioEMAlgo;          // BioEM algorithm used to do comparison (Default 1)
  int CudaThreadCount;    // Number of CUDA threads used in each block (Default
                          // depends on the BioEM algorithm)
  int DebugOutput;        // Debug Output Level (Default 0)
  int nProjectionsAtOnce; // Number of projections to do at once via OpenMP
                          // (Default number of OMP threads)
  bool Autotuning; // Do the autotuning of the load-balancing between CPUs and
  // GPUs (Default 1, if GPUs are used and GPUWORKLOAD is not specified)
};

#endif
