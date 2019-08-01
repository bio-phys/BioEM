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

#ifndef BIOEM_PARAM_H
#define BIOEM_PARAM_H

#include "defs.h"
#include "map.h"
#include <complex>
#include <fftw3.h>
#include <math.h>

using namespace std;

class bioem_param_device
{
public:
  // Grids in center assuming equidistance from 0,0
  int maxDisplaceCenter;
  int GridSpaceCenter;
  int NumberPixels;
  int NumberFFTPixels1D;
  int NxDisp;
  int NtotDisp;

  myfloat_t Ntotpi;
  myfloat_t volu;
  myfloat_t sigmaPriorbctf;
  myfloat_t sigmaPriordefo;
  myfloat_t Priordefcent;
  myfloat_t sigmaPrioramp;
  myfloat_t Priorampcent;
  // If to write Probabilities of Angles from Model
  int writeAngles;
  bool tousepsf;
};

class bioem_param
{
public:
  bioem_param();
  ~bioem_param();

  int readParameters(const char *fileinput);
  int CalculateGridsParam(const char *fileangles);
  int CalculateRefCTF();
  int forprintBest(const char *fileinput);
  void PrepareFFTs();
  bool doaaradius;
  bool writeCTF;
  bool nocentermass;
  bool printrotmod;
  bool readquatlist;
  bool showrotatemod;
  bool notnormmap;
  bool usepsf;
  bool ignorePDB;

  myfloat_t elecwavel;

  bioem_param_device param_device;

  int FFTMapSize;
  int Alignment;
  mycomplex_t *refCTF;
  myfloat3_t *CtfParam;
  size_t getRefCtfCount() { return nTotCTFs * FFTMapSize; }
  size_t getCtfParamCount() { return nTotCTFs; }

  myfloat_t pixelSize;
  // Priors
  myfloat_t priorMod;
  bool yespriorAngles;
  myfloat_t *angprior;

  // Grid Points in Euler angles, assuming uniform sampling d_alpha=d_gamma (in
  // 2pi) & cos(beta)=-1,1
  int angleGridPointsAlpha;
  int angleGridPointsBeta;

  int GridPointsQuatern;
  bool doquater;

  myfloat_t voluang;
  bool notuniformangles;
  int NotUn_angles;

  bool withnoise;
  myfloat_t stnoise;
  //        std::string inanglef;
  //	std::string quatfile;

  int numberGridPointsDisplaceCenter;
  // Grid sampling for the convolution kernel

  //        CTF
  myfloat_t startBfactor, endBfactor;
  int numberBfactor;
  myfloat_t startDefocus, endDefocus;
  int numberDefocus;

  // ENVELOPE
  myfloat_t startGridEnvelop;
  myfloat_t endGridEnvelop;
  int numberGridPointsEnvelop;
  myfloat_t gridEnvelop;
  // CTF=Amp*cos(phase*x)-sqrt(1-Amp**2)*sin(phase*x)
  myfloat_t startGridCTF_phase;
  myfloat_t endGridCTF_phase;
  int numberGridPointsCTF_phase;
  myfloat_t gridCTF_phase;
  myfloat_t startGridCTF_amp;
  myfloat_t endGridCTF_amp;
  int numberGridPointsCTF_amp;
  myfloat_t gridCTF_amp;
  // Others
  myfloat3_t *angles;
  int nTotGridAngles;
  int nTotCTFs;
  int shiftX, shiftY;

  int nTotParallelConv;
  int nTotParallelMaps;

  bool printModel;
  bool BestmapCalcCC;

  int fft_plans_created;
  myfftw_plan fft_plan_c2c_forward, fft_plan_c2c_backward, fft_plan_r2c_forward,
      fft_plan_c2r_backward;

  mycomplex_t **fft_scratch_complex;
  myfloat_t **fft_scratch_real;

  bool dumpMap, loadMap;

  int ddx, ddy;
  myfloat_t bestnorm, bestoff;

private:
  void releaseFFTPlans();
};

#endif
