/* ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
   < BioEM software for Bayesian inference of Electron Microscopy images>
   Copyright (C) 2016 Pilar Cossio, David Rohr, Fabio Baruffa, Markus Rampp,
        Luka Stanisic, Volker Lindenstruth and Gerhard Hummer.
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
#include <vector>

#ifdef WITH_OPENMP
#include <omp.h>
#endif

#include "model.h"
#include "mrc.h"
#include "param.h"

using namespace std;

bioem_model::bioem_model() { points = NULL; }

bioem_model::~bioem_model()
{
  if (points)
    free(points);
}

//************** Loading Model from Binary file *******
void bioem_model::readBinaryFile()
{
  FILE *fp = fopen(FILE_MODEL_DUMP, "rb");
  if (fp == NULL)
  {
    myError("Opening dump file");
  }
  size_t elements_read;
  elements_read = fread(&NormDen, sizeof(NormDen), 1, fp);
  if (elements_read != 1)
  {
    myError("Reading file");
  }
  elements_read = fread(&nPointsModel, sizeof(nPointsModel), 1, fp);
  if (elements_read != 1)
  {
    myError("Reading file");
  }
  points =
      (bioem_model_point *) mallocchk(sizeof(bioem_model_point) * nPointsModel);
  elements_read = fread(points, sizeof(bioem_model_point), nPointsModel, fp);
  if (elements_read != (size_t) nPointsModel)
  {
    myError("Reading file");
  }
  fclose(fp);
  cout << "Protein structure read from model dump\n";
}

//************* Dumping model *********************
void bioem_model::writeBinaryFile()
{
  FILE *fp = fopen(FILE_MODEL_DUMP, "w+b");
  if (fp == NULL)
  {
    myError("Opening dump file");
  }
  fwrite(&NormDen, sizeof(NormDen), 1, fp);
  fwrite(&nPointsModel, sizeof(nPointsModel), 1, fp);
  fwrite(points, sizeof(bioem_model_point), nPointsModel, fp);
  fclose(fp);
}

//************** Reading PDB files **********************
void bioem_model::readPDBFile(const char *filemodel)
{
  /*
    1-6    "ATOM  "
    7-11   Integer serial Atom serial number.
    13-16  Atom name Atom name.
    17     Character altLoc Alternate location indicator.
    18-20  Residue name resName Residue name.
    22     Character chainID Chain identifier.
    23-26  Integer resSeq Residue sequence number.
    27     AChar iCode Code for insertion of residues.
    31-38  Real(8.3) x Orthogonal coordinates for X in
    39-46  Real(8.3) y Orthogonal coordinates for Y in
    47-54  Real(8.3) z Orthogonal coordinates for Z in
  */
  int numres = 0;
  NormDen = 0.0;

  // PDB file name detection
  string strfilename(filemodel);
  size_t foundpos = strfilename.find(".pdb");
  size_t endpos = strfilename.find_last_not_of(" \t");
  if (foundpos > endpos)
  {
    myWarning("PDB extension NOT detected in file name: %s. "
              "Are you sure you want to read a PDB?",
              filemodel);
  }

  if (READ_PARALLEL) // reading PDB file in parallel
  {
    FILE *file = fopen(filemodel, "r");
    if (file == NULL)
    {
      myError("Opening file: %s", filemodel);
    }

    long lSize, size;
    char *buffer;

    // Obtain file size
    fseek(file, 0, SEEK_END);
    lSize = ftell(file);
    rewind(file);

    // Allocate memory to contain the whole file
    buffer = (char *) mallocchk(sizeof(char) * lSize);
    if (buffer == NULL)
    {
      myError("Memory error");
    }

    // Copy the file into the buffer
    size = fread(buffer, 1, lSize, file);
    if (size != lSize)
    {
      myError("Reading error")
    }

    // Parallel read
    int nthreads;
    std::vector<long> *lineStarts;
    int *nlines;
    int *offsets;
#pragma omp parallel
    {
      nthreads = omp_get_max_threads();
      const int num = omp_get_thread_num();
      myfloat_t localNormDen = 0.0;
      char type[7] = {0};
      char name[5] = {0};
// Allocate local data structures
#pragma omp single
      {
        lineStarts = new vector<long>[nthreads];
        nlines = (int *) callocchk(sizeof(int) * nthreads);
        offsets = (int *) mallocchk(sizeof(int) * nthreads);

        // First line is special
        strncpy(type, buffer, 6);
        mySscanf(1, type, "%s", type);
        strncpy(name, buffer + 12, 4);
        mySscanf(1, name, "%s", name);
        if (strcmp(type, "ATOM") == 0 && strcmp(name, "CA") == 0)
        {
          nlines[0] = 1;
          lineStarts[0].push_back(17);
        }
      }
// Get the number of lines and position of resName
#pragma omp for private(type, name)
      for (long i = 0; i < size - 1; i++)
      {
        if (buffer[i] == '\n')
        {
          strncpy(type, buffer + i + 1, 6);
          mySscanf(1, type, "%s", type);
          strncpy(name, buffer + i + 13, 4);
          // It is not an error if "name" doesnt exist, just ignore it
          // This can occur with comments and/or last line
          sscanf(name, "%s", name);
          if (strcmp(type, "ATOM") == 0 && strcmp(name, "CA") == 0)
          {
            nlines[num]++;
            lineStarts[num].push_back(i + 18);
          }
        }
      }
// Allocate points and compute starting position for each thread
#pragma omp single
      {
        for (int i = 0; i < nthreads; i++)
        {
          offsets[i] = numres;
          numres += nlines[i];
        }
        points =
            (bioem_model_point *) mallocchk(sizeof(bioem_model_point) * numres);
      }
// Parallel parsing of the input file
#pragma omp for
      for (int i = 0; i < nthreads; i++)
      {
        char resName[4] = {0};
        char tmpCoor[25] = {0};
        double x = 0.0;
        double y = 0.0;
        double z = 0.0;
        int lPos = offsets[i];
        for (std::vector<long>::const_iterator j = lineStarts[i].begin();
             j != lineStarts[i].end(); j++)
        {
          strncpy(resName, buffer + *j, 3);
          mySscanf(1, resName, "%s", resName);

          strncpy(tmpCoor, buffer + *j + 13, 24);
          mySscanf(3, tmpCoor, "%lf %lf %lf", &x, &y, &z);

          points[lPos].point.pos[0] = (myfloat_t) x;
          points[lPos].point.pos[1] = (myfloat_t) y;
          points[lPos].point.pos[2] = (myfloat_t) z;
          points[lPos].radius = getAminoAcidRad(resName);
          points[lPos].density = getAminoAcidDensity(resName);

          localNormDen += points[lPos].density;
          lPos++;
        }
      }
#pragma omp critical
      NormDen += localNormDen;
    }
    // Cleanup
    free(buffer);
    delete[] lineStarts;
    free(nlines);
    free(offsets);

    fclose(file);
  }
  else // reading PDB file sequentially
  {
    std::ifstream input(filemodel);
    if (!input.good())
    {
      myError("PDB failed to open file: %s", filemodel);
    }

    int allocsize = 0;
    char line[512] = {0};
    char tmpLine[512] = {0};

    while (input.getline(line, 512))
    {
      strncpy(tmpLine, line, strlen(line));
      char *token = strtok(tmpLine, " ");

      if (strcmp(token, "ATOM") ==
          0) // Optional,Mandatory if standard residues exist
      {
        char name[5] = {0};
        char tmp[6] = {0};

        // parse name
        strncpy(tmp, line + 12, 4);
        // It is not an error if "name" doesnt exist, just ignore it
        // This can occur with comments and/or last line
        sscanf(tmp, "%s", name);

        if (strcmp(name, "CA") == 0)
        {
          char resName[4] = {0};
          double x = 0.0;
          double y = 0.0;
          double z = 0.0;

          // parse resName
          strncpy(tmp, line + 17, 3);
          mySscanf(1, tmp, "%s", resName);

          // parse x, y, z
          char tmpVals[36] = {0};

          strncpy(tmpVals, line + 30, 8);
          mySscanf(1, tmpVals, "%lf", &x);

          strncpy(tmpVals, line + 38, 8);
          mySscanf(1, tmpVals, "%lf", &y);

          strncpy(tmpVals, line + 46, 8);
          mySscanf(1, tmpVals, "%lf", &z);

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
          numres++;
        }
      }
    }
  }

#ifdef DEBUG
  printf("NormDen %lf, numres %d\n", NormDen, numres);
#endif
  nPointsModel = numres;
  cout << "Protein structure read from PDB\n";
}

//************** Reading MRC file **********************
void bioem_model::readMRCFile(bioem_param &param, const char *filename)
{
  int numres = 0;
  NormDen = 0.0;

  // Check if filename ends with .mrc
  string strfilename(filename);
  size_t foundpos = strfilename.find(".mrc");
  size_t endpos = strfilename.find_last_not_of(" \t");
  if (foundpos > endpos)
  {
    myWarning("MRC extension NOT detected in file name: %s. "
              "Are you sure you want to read an MRC?",
              filename);
  }

  // Check if MRC has a good format
  int nx, ny, nz, swap, nsymbt;
  check_one_MRC(filename, &swap, &nx, &ny, &nz, &nsymbt);
  int total = nx * ny * nz;

  // Open mrc and go to the first data
  FILE *fin;
  float currfloat;
  fin = fopen(filename, "rb");
  if (fin == NULL)
  {
    myError("Opening MRC: %s", filename);
  }
  for (int count = 0; count < 256; ++count)
  {
    if (read_float_empty(fin) == 0)
    {
      myError("Converting Data: %s", filename);
    }
  }
  for (long count = 0; count < (long) nsymbt; ++count)
  {
    if (read_char_float(&currfloat, fin) == 0)
    {
      myError("Converting Data: %s", filename);
    }
  }

  // Actual reading of the data
  points = (bioem_model_point *) mallocchk(sizeof(bioem_model_point) * total);
  for (int i = 1; i <= nx; i++)
  {
    for (int j = 1; j <= ny; j++)
    {
      for (int k = 1; k <= nz; k++)
      {
        if (read_float(&currfloat, fin, swap) == 0)
        {
          myError("Converting Data: %s", filename);
        }
        else
        {
          // Getting the coordinates
          points[numres].point.pos[0] = (i - nx / 2.0) * param.pixelSize;
          points[numres].point.pos[1] = (j - ny / 2.0) * param.pixelSize;
          points[numres].point.pos[2] = (k - nz / 2.0) * param.pixelSize;

          points[numres].radius = 2.0 * param.pixelSize;
          points[numres].density = (myfloat_t) currfloat;
          NormDen += points[numres].density;

          numres++;
        }
      }
    }
  }

  if (numres != total)
  {
    myError("Number of points in .mrc file is %d, while %d was expected",
            numres, total);
  }

#ifdef DEBUG
  printf("NormDen %lf, numres %d, nx*ny*nz %d\n", NormDen, numres, total);
#endif
  nPointsModel = numres;
  cout << "Protein structure read from MRC\n";
}

//**************** Reading Text FILES ***********************
void bioem_model::readTextFile(bioem_param &param, const char *filemodel)
{
  cout << "Note: Reading model in simple text format\n";
  cout << "----  x   y   z  radius  density ------- \n";

  double tmpval[5];
  int numres = 0;
  NormDen = 0.0;

  // PDB file name detection
  string strfilename(filemodel);
  size_t foundpos = strfilename.find(".pdb");
  size_t endpos = strfilename.find_last_not_of(" \t");
  if (foundpos < endpos)
  {
    myWarning("PDB detected in file name: %s. "
              "Are you sure you do not need --ReadPDB? "
              "If so then you must include the keyword "
              "IGNORE_PDB in inputfile",
              filemodel);
    if (not param.ignorePDB)
    {
      myError("PDB is not ignored");
    }
  }

  FILE *file = fopen(filemodel, "r");
  if (file == NULL)
  {
    myError("Opening file: %s", filemodel);
  }

  if (READ_PARALLEL) // reading textual file in parallel
  {
    long lSize, size;
    char *buffer;

    // Obtain file size
    fseek(file, 0, SEEK_END);
    lSize = ftell(file);
    rewind(file);

    // Allocate memory to contain the whole file
    buffer = (char *) mallocchk(sizeof(char) * lSize);
    if (buffer == NULL)
    {
      myError("Memory error");
    }

    // Copy the file into the buffer
    size = fread(buffer, 1, lSize, file);
    if (size != lSize)
    {
      myError("Reading error")
    }

    // Parallel read
    int nthreads;
    std::vector<long> *lineStarts;
    int *nlines;
    int *offsets;
#pragma omp parallel
    {
      nthreads = omp_get_max_threads();
      const int num = omp_get_thread_num();
      myfloat_t localNormDen = 0.0;
// Allocate local data structures
#pragma omp single
      {
        lineStarts = new vector<long>[nthreads];
        nlines = (int *) callocchk(sizeof(int) * nthreads);
        offsets = (int *) mallocchk(sizeof(int) * nthreads);

        lineStarts[0].push_back(0);
        nlines[0] = 1;
      }
// Get the number of lines and end of line position
#pragma omp for
      for (long i = 0; i < size - 1; i++)
      {
        if (buffer[i] == '\n')
        {
          nlines[num]++;
          lineStarts[num].push_back(i + 1);
          buffer[i] = '\0';
        }
      }
// Allocate points and compute starting position for each thread
#pragma omp single
      {
        for (int i = 0; i < nthreads; i++)
        {
          offsets[i] = numres;
          numres += nlines[i];
        }
        points =
            (bioem_model_point *) mallocchk(sizeof(bioem_model_point) * numres);
      }
// Parallel parsing of the input file
#pragma omp for private(tmpval)
      for (int i = 0; i < nthreads; i++)
      {
        char *localBuff;
        int lPos = offsets[i];
        for (std::vector<long>::const_iterator j = lineStarts[i].begin();
             j != lineStarts[i].end(); j++)
        {
          localBuff = &buffer[*j];
          mySscanf(5, localBuff, "%lf %lf %lf %lf %lf", &tmpval[0], &tmpval[1],
                   &tmpval[2], &tmpval[3], &tmpval[4]);

          if (tmpval[3] < 0)
          {
            myError("Radius must be positive");
          }

          points[lPos].point.pos[0] = (myfloat_t) tmpval[0];
          points[lPos].point.pos[1] = (myfloat_t) tmpval[1];
          points[lPos].point.pos[2] = (myfloat_t) tmpval[2];
          points[lPos].radius = (myfloat_t) tmpval[3];
          points[lPos].density = (myfloat_t) tmpval[4];

          localNormDen += points[lPos].density;
          lPos++;
        }
      }
#pragma omp critical
      NormDen += localNormDen;
    }
    // Cleanup
    free(buffer);
    delete[] lineStarts;
    free(nlines);
    free(offsets);
  }
  else // reading textual file sequentially
  {
    char line[128];
    int allocsize = 0;
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

      mySscanf(5, line, "%lf %lf %lf %lf %lf", &tmpval[0], &tmpval[1],
               &tmpval[2], &tmpval[3], &tmpval[4]);

      if (tmpval[3] < 0)
      {
        myError("Radius must be positive");
      }

      points[numres].point.pos[0] = (myfloat_t) tmpval[0];
      points[numres].point.pos[1] = (myfloat_t) tmpval[1];
      points[numres].point.pos[2] = (myfloat_t) tmpval[2];
      points[numres].radius = (myfloat_t) tmpval[3];
      points[numres].density = (myfloat_t) tmpval[4];
      NormDen += points[numres].density;
      numres++;
    }

    points = (bioem_model_point *) reallocchk(
        points, sizeof(bioem_model_point) * numres);
  }

#ifdef DEBUG
  printf("NormDen %lf, numres %d\n", NormDen, numres);
#endif

  nPointsModel = numres;
  fclose(file);
  cout << "Protein structure read from Standard File\n";
}

//******************** Moving to Model to its center of density mass
void bioem_model::centerDensityMass()
{
  myfloat3_t r_cm;
  r_cm.pos[0] = 0.0;
  r_cm.pos[1] = 0.0;
  r_cm.pos[2] = 0.0;

  if (READ_PARALLEL) // in parallel
  {
#pragma omp parallel shared(r_cm)
    {
      myfloat3_t l_r_cm;
      l_r_cm.pos[0] = 0.0;
      l_r_cm.pos[1] = 0.0;
      l_r_cm.pos[2] = 0.0;
#pragma omp for
      for (int n = 0; n < nPointsModel; n++)
      {
        l_r_cm.pos[0] += points[n].point.pos[0] * points[n].density;
        l_r_cm.pos[1] += points[n].point.pos[1] * points[n].density;
        l_r_cm.pos[2] += points[n].point.pos[2] * points[n].density;
      }
#pragma omp critical
      {
        r_cm.pos[0] += l_r_cm.pos[0];
        r_cm.pos[1] += l_r_cm.pos[1];
        r_cm.pos[2] += l_r_cm.pos[2];
      }
#pragma omp barrier
#pragma omp single
      {
        r_cm.pos[0] /= NormDen;
        r_cm.pos[1] /= NormDen;
        r_cm.pos[2] /= NormDen;
      }
#ifdef DEBUG
      printf("Centering: NormDen %lf r_cm[0] %lf r_cm[1] %lf r_cm[2] %lf\n",
             NormDen, r_cm.pos[0], r_cm.pos[1], r_cm.pos[2]);
#endif
#pragma omp for
      for (int n = 0; n < nPointsModel; n++)
      {
        points[n].point.pos[0] -= r_cm.pos[0];
        points[n].point.pos[1] -= r_cm.pos[1];
        points[n].point.pos[2] -= r_cm.pos[2];
      }
    }
  }
  else // sequential
  {
    for (int n = 0; n < nPointsModel; n++)
    {
      r_cm.pos[0] += points[n].point.pos[0] * points[n].density;
      r_cm.pos[1] += points[n].point.pos[1] * points[n].density;
      r_cm.pos[2] += points[n].point.pos[2] * points[n].density;
    }

    r_cm.pos[0] /= NormDen;
    r_cm.pos[1] /= NormDen;
    r_cm.pos[2] /= NormDen;

    for (int n = 0; n < nPointsModel; n++)
    {
      points[n].point.pos[0] -= r_cm.pos[0];
      points[n].point.pos[1] -= r_cm.pos[1];
      points[n].point.pos[2] -= r_cm.pos[2];
    }
  }
}

int bioem_model::readModel(bioem_param &param, const char *filemodel)
{
  // ***Reading reference Models either PDB or x,y,z,r,d format***

  if (loadModel)
  {
    readBinaryFile();
  }
  else if (readPDB)
  {
    readPDBFile(filemodel);
  }
  else if (readModelMRC)
  {
    readMRCFile(param, filemodel);
  }
  else
  {
    readTextFile(param, filemodel);
  }

  if (dumpModel)
  {
    writeBinaryFile();
  }

  cout << "Total Number of Voxels " << nPointsModel;
  cout << "\nTotal Number of Electrons " << NormDen;
  cout << "\n+++++++++++++++++++++++++++++++++++++++++ \n";

  if (not(param.nocentermass))
  { // by default it is normally done
    centerDensityMass();
  }

  return (0);
}

int bioem_model::printCOOR()
{
  cout << "Note - Look at file " << FILE_COORDREAD
       << " to confirm that the Model coordinates are correct\n";

  ofstream exampleReadCoor;
  exampleReadCoor.open(FILE_COORDREAD);

  exampleReadCoor << "Text --- Number ---- x ---- y ---- z ---- radius ---- "
                     "number of electron\n";

  for (int numres = 0; numres < nPointsModel; numres++)
  {
    exampleReadCoor << "RESIDUE " << numres << " "
                    << points[numres].point.pos[0] << " "
                    << points[numres].point.pos[1] << " "
                    << points[numres].point.pos[2] << " "
                    << points[numres].radius << " " << points[numres].density
                    << "\n";
  }

  exampleReadCoor.close();

  return 0;
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
    myError("Amino acid: %s", name);
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
    myError("Amino acid: %s", name);
  }
  return iaa;
}
