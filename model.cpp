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

#include <cstring>
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#include "model.h"
#include "param.h"

using namespace std;

bioem_model::bioem_model() { points = NULL; }

bioem_model::~bioem_model()
{
  if (points)
    free(points);
}

int bioem_model::readModel(bioem_param &param, const char *filemodel)
{
  // **************************************************************************************
  // ***************Reading reference Models either PDB or x,y,z,r,d
  // format****************
  // **************************************************************************************

  ofstream exampleReadCoor;
  exampleReadCoor.open("COORDREAD");

  exampleReadCoor << "Text --- Number ---- x ---- y ---- z ---- radius ---- "
                     "number of electron\n";
  int allocsize = 0;

  std::ifstream input(filemodel);
  if (readPDB)
  {
    //************** Reading PDB files **********************

    ifstream input(filemodel);
    if (!input.good())
    {
      cout << "PDB Failed to open file"
           << endl; // pdbfilename << " ("<<filename<<")\n";
      exit(1);
    }

    char line[512] = {0};
    char tmpLine[512] = {0};
    int numres = 0;
    NormDen = 0.0;

    string strfilename(filemodel);

    size_t foundpos = strfilename.find(".pdb");
    size_t endpos = strfilename.find_last_not_of(" \t");

    if (foundpos > endpos)
    {
      cout << "Warining:::: .pdb extension NOT dectected in file name \n";
      cout << "Warining::::  Are you sure you want to read a PDB? \n";
    }

    //  cout << " HERE	" << filemodel ;
    // for eachline in the file
    while (!input.eof())
    {
      input.getline(line, 511);

      strncpy(tmpLine, line, strlen(line));
      char *token = strtok(tmpLine, " ");

      if (strcmp(token, "ATOM") ==
          0) // Optional,Mandatory if standard residues exist
      {
        /*
          1-6 			"ATOM  "
          7 - 11		 Integer		 serial		Atom
          serial
          number.
          13 - 16		Atom			name		  Atom
          name.
          17			 Character	   altLoc Alternate
          location indicator.
          18 - 20		Residue name	resName	   Residue name.
          22			 Character	   chainID	   Chain
          identifier.
          23 - 26		Integer		 resSeq		Residue
          sequence number.
          27			 AChar		   iCode		 Code
          for
          insertion of residues.
          31 - 38		Real(8.3)	   x Orthogonal
          coordinates for X in
          39 - 46		Real(8.3)	   y Orthogonal
          coordinates for Y in
          47 - 54		Real(8.3)	   z Orthogonal
          coordinates for Z in
        */

        char name[5] = {0};
        char resName[4] = {0};
        float x = 0.0;
        float y = 0.0;
        float z = 0.0;
        char tmp[6] = {0};

        // parse name
        strncpy(tmp, line + 12, 4);
        sscanf(tmp, "%s", name);

        // parse resName
        strncpy(tmp, line + 17, 3);
        sscanf(tmp, "%s", resName);

        // parse x, y, z
        char tmpVals[36] = {0};

        strncpy(tmpVals, line + 30, 8);
        sscanf(tmpVals, "%f", &x);

        strncpy(tmpVals, line + 38, 8);
        sscanf(tmpVals, "%f", &y);

        strncpy(tmpVals, line + 46, 8);
        sscanf(tmpVals, "%f", &z);

        if (strcmp(name, "CA") == 0)
        {
          if (allocsize == 0)
          {
            allocsize = 64;
            points = (bioem_model_point *) mallocchk(sizeof(bioem_model_point) *
                                                     allocsize);
          }
          else if (numres + 1 >= allocsize)
          {
            allocsize *= 2;
            points = (bioem_model_point *) reallocchk(
                points, sizeof(bioem_model_point) * allocsize);
          }

          // Getting residue Radius and electron density
          points[numres].radius = getAminoAcidRad(resName);
          points[numres].density = getAminoAcidDensity(resName);
          NormDen += points[numres].density;

          // Getting the coordinates
          points[numres].point.pos[0] = (myfloat_t) x;
          points[numres].point.pos[1] = (myfloat_t) y;
          points[numres].point.pos[2] = (myfloat_t) z;
          exampleReadCoor << "RESIDUE " << numres << " "
                          << points[numres].point.pos[0] << " "
                          << points[numres].point.pos[1] << " "
                          << points[numres].point.pos[2] << " "
                          << points[numres].radius << " "
                          << points[numres].density << "\n";
          numres++;
        }
      }
    }
    nPointsModel = numres;
    cout << "Protein structure read from PDB\n";
  }
  else // Reading model from FILE FORMAT x,y,z,rad,density
  {
    //**************** Reading Text FILES ***********************

    char line[128];
    int numres = 0;
    NormDen = 0.0;

    string strfilename(filemodel);

    size_t foundpos = strfilename.find(".pdb");
    size_t endpos = strfilename.find_last_not_of(" \t");

    if (foundpos < endpos)
    {
      cout << "Warining:::: .pdb dectected in file name whilst using text read "
              "\n";
      cout << "Warining::::  Are you sure you do not need --ReadPDB? \n";
      cout << "If so then you must include the keyword IGNORE_PDB in "
              "inputfile\n";
      if (not param.ignorePDB)
        exit(1);
    }

    FILE *file = fopen(filemodel, "r");
    if (file == NULL)
    {
      cout << "Error opening file " << filemodel << "\n";
      exit(1);
    }
    while (fgets(line, sizeof line, file) != NULL)
    {
      if (allocsize == 0)
      {
        allocsize = 64;
        points = (bioem_model_point *) mallocchk(sizeof(bioem_model_point) *
                                                 allocsize);
      }
      else if (numres + 1 >= allocsize)
      {
        allocsize *= 2;
        points = (bioem_model_point *) reallocchk(
            points, sizeof(bioem_model_point) * allocsize);
      }

      float tmpval[5];
      sscanf(line, "%f %f %f %f %f", &tmpval[0], &tmpval[1], &tmpval[2],
             &tmpval[3], &tmpval[4]);
      points[numres].point.pos[0] = (myfloat_t) tmpval[0];
      points[numres].point.pos[1] = (myfloat_t) tmpval[1];
      points[numres].point.pos[2] = (myfloat_t) tmpval[2];
      points[numres].radius = (myfloat_t) tmpval[3];
      points[numres].density = (myfloat_t) tmpval[4];

      exampleReadCoor << "RESIDUE " << numres << " "
                      << points[numres].point.pos[0] << " "
                      << points[numres].point.pos[1] << " "
                      << points[numres].point.pos[2] << " "
                      << points[numres].radius << " " << points[numres].density
                      << "\n";
      NormDen += points[numres].density;
      numres++;
    }
    fclose(file);
    nPointsModel = numres;
    cout << "Protein structure read from Standard File\n";
  }
  points = (bioem_model_point *) reallocchk(points, sizeof(bioem_model_point) *
                                                        nPointsModel);
  cout << "Total Number of Voxels " << nPointsModel;
  cout << "\nTotal Number of Electrons " << NormDen;
  cout << "\n+++++++++++++++++++++++++++++++++++++++++ \n";

  exampleReadCoor.close();
  //******************** Moving to Model to its center of density mass:
  myfloat3_t r_cm;

  if (not(param.nocentermass))
  { // by default it is normally done

    for (int n = 0; n < 3; n++)
      r_cm.pos[n] = 0.0;

    for (int n = 0; n < nPointsModel; n++)
    {
      r_cm.pos[0] += points[n].point.pos[0] * points[n].density;
      r_cm.pos[1] += points[n].point.pos[1] * points[n].density;
      r_cm.pos[2] += points[n].point.pos[2] * points[n].density;
    }
    r_cm.pos[0] = r_cm.pos[0] / NormDen;
    r_cm.pos[1] = r_cm.pos[1] / NormDen;
    r_cm.pos[2] = r_cm.pos[2] / NormDen;

    for (int n = 0; n < nPointsModel; n++)
    {
      points[n].point.pos[0] -= r_cm.pos[0];
      points[n].point.pos[1] -= r_cm.pos[1];
      points[n].point.pos[2] -= r_cm.pos[2];
    }
  }
  return (0);
}

myfloat_t bioem_model::getAminoAcidRad(char *name)
{
  // *************** Function that gets the radius for each amino acid
  // ****************
  myfloat_t iaa = 0;

  if (std::strcmp(name, "CYS") == 0)
    iaa = 2.75;
  else if (std::strcmp(name, "PHE") == 0)
    iaa = 3.2;
  else if (std::strcmp(name, "LEU") == 0)
    iaa = 3.1;
  else if (std::strcmp(name, "TRP") == 0)
    iaa = 3.4;
  else if (std::strcmp(name, "VAL") == 0)
    iaa = 2.95;
  else if (std::strcmp(name, "ILE") == 0)
    iaa = 3.1;
  else if (std::strcmp(name, "MET") == 0)
    iaa = 3.1;
  else if (std::strcmp(name, "HIS") == 0)
    iaa = 3.05;
  else if (std::strcmp(name, "TYR") == 0)
    iaa = 3.25;
  else if (std::strcmp(name, "ALA") == 0)
    iaa = 2.5;
  else if (std::strcmp(name, "GLY") == 0)
    iaa = 2.25;
  else if (std::strcmp(name, "PRO") == 0)
    iaa = 2.8;
  else if (std::strcmp(name, "ASN") == 0)
    iaa = 2.85;
  else if (std::strcmp(name, "THR") == 0)
    iaa = 2.8;
  else if (std::strcmp(name, "SER") == 0)
    iaa = 2.6;
  else if (std::strcmp(name, "ARG") == 0)
    iaa = 3.3;
  else if (std::strcmp(name, "GLN") == 0)
    iaa = 3.0;
  else if (std::strcmp(name, "ASP") == 0)
    iaa = 2.8;
  else if (std::strcmp(name, "LYS") == 0)
    iaa = 3.2;
  else if (std::strcmp(name, "GLU") == 0)
    iaa = 2.95;

  if (iaa == 0)
  {
    cout << "PROBLEM WITH AMINO ACID " << name << endl;
    exit(1);
  }
  return iaa;
}

myfloat_t bioem_model::getAminoAcidDensity(char *name)
{
  // *************** Function that gets the number of electrons for each amino
  // acid ****************
  myfloat_t iaa = 0.0;

  if (std::strcmp(name, "CYS") == 0)
    iaa = 64.0;
  else if (std::strcmp(name, "PHE") == 0)
    iaa = 88.0;
  else if (std::strcmp(name, "LEU") == 0)
    iaa = 72.0;
  else if (std::strcmp(name, "TRP") == 0)
    iaa = 108.0;
  else if (std::strcmp(name, "VAL") == 0)
    iaa = 64.0;
  else if (std::strcmp(name, "ILE") == 0)
    iaa = 72.0;
  else if (std::strcmp(name, "MET") == 0)
    iaa = 80.0;
  else if (std::strcmp(name, "HIS") == 0)
    iaa = 82.0;
  else if (std::strcmp(name, "TYR") == 0)
    iaa = 96.0;
  else if (std::strcmp(name, "ALA") == 0)
    iaa = 48.0;
  else if (std::strcmp(name, "GLY") == 0)
    iaa = 40.0;
  else if (std::strcmp(name, "PRO") == 0)
    iaa = 62.0;
  else if (std::strcmp(name, "ASN") == 0)
    iaa = 66.0;
  else if (std::strcmp(name, "THR") == 0)
    iaa = 64.0;
  else if (std::strcmp(name, "SER") == 0)
    iaa = 56.0;
  else if (std::strcmp(name, "ARG") == 0)
    iaa = 93.0;
  else if (std::strcmp(name, "GLN") == 0)
    iaa = 78.0;
  else if (std::strcmp(name, "ASP") == 0)
    iaa = 59.0;
  else if (std::strcmp(name, "LYS") == 0)
    iaa = 79.0;
  else if (std::strcmp(name, "GLU") == 0)
    iaa = 53.0;

  if (iaa == 0.0)
  {
    cout << "PROBLEM WITH AMINO ACID " << name << endl;
    exit(1);
  }
  return iaa;
}
