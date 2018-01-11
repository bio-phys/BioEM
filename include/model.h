/* ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
   < BioEM software for Bayesian inference of Electron Microscopy images>
   Copyright (C) 2016 Pilar Cossio, David Rohr, Fabio Baruffa, Markus Rampp,
        Volker Lindenstruth and Gerhard Hummer.
   Max Planck Institute of Biophysics, Frankfurt, Germany.
   Frankfurt Institute for Advanced Studies, Goethe University Frankfurt,
   Germany.
   Max Planck Computing and Data Facility, Garching, Germany.

   Released under the GNU Public License, v3.
   See license statement for terms of distribution.

   ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/

#ifndef BIOEM_MODEL_H
#define BIOEM_MODEL_H

#include "defs.h"
#include "param.h"

class bioem_model
{
public:
  class bioem_model_point
  {
  public:
    myfloat3_t point;
    myfloat_t radius;
    myfloat_t density;
  };

  bioem_model();
  ~bioem_model();

  int readModel(bioem_param &param, const char *filemodel);

  bool readPDB;

  myfloat_t getAminoAcidRad(char *name);
  myfloat_t getAminoAcidDensity(char *name);
  myfloat_t NormDen;

  int nPointsModel;
  bioem_model_point *points;
};

#endif
