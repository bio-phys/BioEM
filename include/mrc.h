/* ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
   < BioEM software for Bayesian inference of Electron Microscopy images>
   Copyright (C) 2019 Pilar Cossio, Markus Rampp, Luka Stanisic and Gerhard
   Hummer. Max Planck Institute of Biophysics, Frankfurt, Germany. Frankfurt
   Institute for Advanced Studies, Goethe University Frankfurt, Germany. Max
   Planck Computing and Data Facility, Garching, Germany.

   Released under the GNU Public License, v3.
   See license statement for terms of distribution.

   ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/

#include "defs.h"

static inline int read_float(float *currfloat, FILE *fin, int swap)
{
  unsigned char *cptr, tmp;

  if (fread(currfloat, 4, 1, fin) != 1)
    return 0;
  if (swap == 1)
  {
    cptr = (unsigned char *) currfloat;
    tmp = cptr[0];
    cptr[0] = cptr[3];
    cptr[3] = tmp;
    tmp = cptr[1];
    cptr[1] = cptr[2];
    cptr[2] = tmp;
  }
  return 1;
}

static inline int read_int(int *currlong, FILE *fin, int swap)
{
  unsigned char *cptr, tmp;

  if (fread(currlong, 4, 1, fin) != 1)
    return 0;
  if (swap == 1)
  {
    cptr = (unsigned char *) currlong;
    tmp = cptr[0];
    cptr[0] = cptr[3];
    cptr[3] = tmp;
    tmp = cptr[1];
    cptr[1] = cptr[2];
    cptr[2] = tmp;
  }
  return 1;
}

static inline int read_float_empty(FILE *fin)
{
  float currfloat;

  if (fread(&currfloat, 4, 1, fin) != 1)
    return 0;
  return 1;
}

static inline int read_char_float(float *currfloat, FILE *fin)
{
  char currchar;

  if (fread(&currchar, 1, 1, fin) != 1)
    return 0;
  *currfloat = (float) currchar;
  return 1;
}

static int test_mrc(const char *vol_file, int swap)
{
  FILE *fin;
  int nc, nr, ns, mx, my, mz;
  int mode, ncstart, nrstart, nsstart;
  float xlen, ylen, zlen;
  int i, header_ok = 1, n_range_viols = 0;
  int mapc, mapr, maps_local;
  float alpha, beta, gamma;
  float dmin, dmax, dmean, dummy, xorigin, yorigin, zorigin;

  fin = fopen(vol_file, "rb");
  if (fin == NULL)
  {
    myError("Opening MRC: %s", vol_file);
  }

  //* read header info
  header_ok *= read_int(&nc, fin, swap);
  header_ok *= read_int(&nr, fin, swap);
  header_ok *= read_int(&ns, fin, swap);
  header_ok *= read_int(&mode, fin, swap);
  header_ok *= read_int(&ncstart, fin, swap);
  header_ok *= read_int(&nrstart, fin, swap);
  header_ok *= read_int(&nsstart, fin, swap);
  header_ok *= read_int(&mx, fin, swap);
  header_ok *= read_int(&my, fin, swap);
  header_ok *= read_int(&mz, fin, swap);
  header_ok *= read_float(&xlen, fin, swap);
  header_ok *= read_float(&ylen, fin, swap);
  header_ok *= read_float(&zlen, fin, swap);
  header_ok *= read_float(&alpha, fin, swap);
  header_ok *= read_float(&beta, fin, swap);
  header_ok *= read_float(&gamma, fin, swap);
  header_ok *= read_int(&mapc, fin, swap);
  header_ok *= read_int(&mapr, fin, swap);
  header_ok *= read_int(&maps_local, fin, swap);
  header_ok *= read_float(&dmin, fin, swap);
  header_ok *= read_float(&dmax, fin, swap);
  header_ok *= read_float(&dmean, fin, swap);
  for (i = 23; i < 50; ++i)
    header_ok *= read_float(&dummy, fin, swap);
  header_ok *= read_float(&xorigin, fin, swap);
  header_ok *= read_float(&yorigin, fin, swap);
  header_ok *= read_float(&zorigin, fin, swap);
  fclose(fin);
  if (header_ok == 0)
  {
    myError("Reading MRC header: %s", vol_file);
  }

  n_range_viols += (nc > 5000);
  n_range_viols += (nc < 0);
  n_range_viols += (nr > 5000);
  n_range_viols += (nr < 0);
  n_range_viols += (ns > 5000);
  n_range_viols += (ns < 0);
  n_range_viols += (ncstart > 5000);
  n_range_viols += (ncstart < -5000);
  n_range_viols += (nrstart > 5000);
  n_range_viols += (nrstart < -5000);
  n_range_viols += (nsstart > 5000);
  n_range_viols += (nsstart < -5000);
  n_range_viols += (mx > 5000);
  n_range_viols += (mx < 0);
  n_range_viols += (my > 5000);
  n_range_viols += (my < 0);
  n_range_viols += (mz > 5000);
  n_range_viols += (mz < 0);
  n_range_viols += (alpha > 360.0f);
  n_range_viols += (alpha < -360.0f);
  n_range_viols += (beta > 360.0f);
  n_range_viols += (beta < -360.0f);
  n_range_viols += (gamma > 360.0f);
  n_range_viols += (gamma < -360.0f);

  return n_range_viols;
}

static int check_one_MRC(const char *filename, int *swap, int *nc, int *nr,
                         int *ns, int *nsymbt)
{
  // Partially:
  /*     subroutine "read_MRC" of the Situs 2.7.2 program.
         Ref: Willy Wriggers. Using Situs for the Integration of
     Multi-Resolution Structures.
         Biophysical Reviews, 2010, Vol. 2, pp. 21-27.*/

  FILE *fin;
  int header_ok = 1;
  float xlen, ylen, zlen;
  int mode, ncstart, nrstart, nsstart, ispg, lskflg;
  float a_tmp, b_tmp, g_tmp;
  int mx, my, mz, mapc, mapr, maps_local;
  float dmin, dmax, dmean;
  int n_range_viol0, n_range_viol1;

  fin = fopen(filename, "rb");
  if (fin == NULL)
  {
    myError("Opening MRC: %s", filename);
  }

  // Setting up swap
  n_range_viol0 = test_mrc(filename, 0);
  n_range_viol1 = test_mrc(filename, 1);
  if (n_range_viol0 < n_range_viol1)
  { //* guess endianism
    *swap = 0;
    if (n_range_viol0 > 0)
    {
      myWarning("%i header field range violations detected in file %s",
                n_range_viol0, filename);
    }
  }
  else
  {
    *swap = 1;
    if (n_range_viol1 > 0)
    {
      myWarning("%i header field range violations detected in file %s",
                n_range_viol1, filename);
    }
  }

  // Reading header
  header_ok *= read_int(nc, fin, *swap);
  header_ok *= read_int(nr, fin, *swap);
  header_ok *= read_int(ns, fin, *swap);
  header_ok *= read_int(&mode, fin, *swap);
  header_ok *= read_int(&ncstart, fin, *swap);
  header_ok *= read_int(&nrstart, fin, *swap);
  header_ok *= read_int(&nsstart, fin, *swap);
  header_ok *= read_int(&mx, fin, *swap);
  header_ok *= read_int(&my, fin, *swap);
  header_ok *= read_int(&mz, fin, *swap);
  header_ok *= read_float(&xlen, fin, *swap);
  header_ok *= read_float(&ylen, fin, *swap);
  header_ok *= read_float(&zlen, fin, *swap);
  header_ok *= read_float(&a_tmp, fin, *swap);
  header_ok *= read_float(&b_tmp, fin, *swap);
  header_ok *= read_float(&g_tmp, fin, *swap);
  header_ok *= read_int(&mapc, fin, *swap);
  header_ok *= read_int(&mapr, fin, *swap);
  header_ok *= read_int(&maps_local, fin, *swap);
  header_ok *= read_float(&dmin, fin, *swap);
  header_ok *= read_float(&dmax, fin, *swap);
  header_ok *= read_float(&dmean, fin, *swap);
  header_ok *= read_int(&ispg, fin, *swap);
  header_ok *= read_int(nsymbt, fin, *swap);
  header_ok *= read_int(&lskflg, fin, *swap);

  // Verify header values
  if (header_ok == 0)
  {
    myError("Reading MRC header: %s", filename);
  }

  if (mode != 2)
  {
    myError("MRC mode: %d. Currently mode 2 is the only one allowed", mode);
  }

  fclose(fin);
  return (0);
}
