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

#ifndef BIOEM_MAP_H
#define BIOEM_MAP_H

#include "defs.h"
#include "param.h"
#include <complex>
#include <math.h>

class bioem_param;
class bioem;

class bioem_RefMap
{
public:
  bioem_RefMap()
  {
    maps = NULL;
    RefMapsFFT = NULL;
    sum_RefMap = NULL;
    sumsquare_RefMap = NULL;
  }

  void freePointers()
  {
    if (maps)
      free(maps);
    if (sum_RefMap)
      free(sum_RefMap);
    if (sumsquare_RefMap)
      free(sumsquare_RefMap);
    if (RefMapsFFT)
      delete[] RefMapsFFT;
    maps = NULL;
    sum_RefMap = NULL;
    sumsquare_RefMap = NULL;
    RefMapsFFT = NULL;
  }
  int readRefMaps(bioem_param &param, const char *filemap);
  int precalculate(bioem_param &param, bioem &bio);
  int PreCalculateMapsFFT(bioem_param &param);

  int read_int(int *currlong, FILE *fin, int swap);
  int read_float(float *currfloat, FILE *fin, int swap);
  int read_float_empty(FILE *fin);
  int read_char_float(float *currfloat, FILE *fin);
  int test_mrc(const char *vol_file, int swap);
  int read_MRC(const char *filename, bioem_param &param);

  mycomplex_t *RefMapsFFT;

  bool readMRC, readMultMRC;

  int ntotRefMap;
  int numPixels;
  int refMapSize;
  myfloat_t *maps;
  myfloat_t *sum_RefMap;
  myfloat_t *sumsquare_RefMap;

  __host__ __device__ inline myfloat_t get(int map, int x, int y) const
  {
    return (maps[map * refMapSize + x * numPixels + y]);
  }
  __host__ __device__ inline const myfloat_t *getp(int map, int x, int y) const
  {
    return (&maps[map * refMapSize + x * numPixels]);
  }
  __host__ __device__ inline myfloat_t *getmap(int map)
  {
    return (&maps[map * refMapSize]);
  }
};

class bioem_RefMap_Mod : public bioem_RefMap
{
public:
  __host__ __device__ inline myfloat_t get(int map, int x, int y) const
  {
    return (maps[(x * numPixels + y) * ntotRefMap + map]);
  }

  void init(const bioem_RefMap &map)
  {
    maps = (myfloat_t *) malloc(map.refMapSize * map.ntotRefMap *
                                sizeof(myfloat_t));
#pragma omp parallel for
    for (int i = 0; i < map.ntotRefMap; i++)
    {
      for (int j = 0; j < map.numPixels; j++)
      {
        for (int k = 0; k < map.numPixels; k++)
        {
          maps[(j * map.numPixels + k) * map.ntotRefMap + i] = map.get(i, j, k);
        }
      }
    }
  }
};

class bioem_Probability_map
{
public:
  myprob_t Total;
  myprob_t Constoadd;

  class bioem_Probability_map_max
  {
  public:
    int max_prob_cent_x, max_prob_cent_y, max_prob_orient, max_prob_conv;
    myfloat_t max_prob_norm, max_prob_mu;
  } max;
};

class bioem_Probability_angle
{
public:
  myprob_t forAngles;
  myprob_t ConstAngle;
};

class bioem_Probability
{
public:
  int nMaps;
  int nAngles;

  __device__ __host__ bioem_Probability_map &getProbMap(int map)
  {
    return (ptr_map[map]);
  }
  __device__ __host__ bioem_Probability_angle &getProbAngle(int map, int angle)
  {
    return (ptr_angle[angle * nMaps + map]);
  }

  void *ptr;
  bioem_Probability_map *ptr_map;
  bioem_Probability_angle *ptr_angle;

  static size_t get_size(size_t maps, size_t angles, int writeAngles)
  {
    size_t size = sizeof(bioem_Probability_map);
    if (writeAngles)
      size += angles * sizeof(bioem_Probability_angle);
    return (maps * size);
  }

  void init(size_t maps, size_t angles, bioem &bio);
  void copyFrom(bioem_Probability *from, bioem &bio);

  void set_pointers()
  {
    ptr_map = (bioem_Probability_map *) ptr;
    ptr_angle = (bioem_Probability_angle *) (&ptr_map[nMaps]);
  }
};

#endif
