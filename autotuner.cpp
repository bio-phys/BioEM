/* ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
   < BioEM software for Bayesian inference of Electron Microscopy images>
   Copyright (C) 2017 Pilar Cossio, Markus Rampp, Luka Stanisic and Gerhard
   Hummer.
   Max Planck Institute of Biophysics, Frankfurt, Germany.
   Max Planck Computing and Data Facility, Garching, Germany.

   Released under the GNU Public License, v3.
   See license statement for terms of distribution.

   ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/

#include "autotuner.h"

void Autotuner::Reset()
{
  stopTuning = false;
  workload = 100;

  best_time = 0.;
  best_workload = 0;

  a = 1;
  b = 50;
  c = 100;
  x = 50;
  limit = 1;
  fb = 0.;
  fx = 0.;

  if (algo == 3)
    workload = 50;
}

bool Autotuner::Needed(int iteration)
{
  if (stopTuning)
    return false;

  switch (algo)
  {
    case 1:
    case 3:
      return iteration % (stable + 1) == stable;
    case 2:
      return (iteration == (int) stable / 2) || (iteration == stable);
    default: /* Should never happen */;
  }
  return false;
}

bool Autotuner::Finished()
{
  switch (algo)
  {
    case 1:
      if (workload < 30)
      {
        workload = best_workload;
        return stopTuning = true;
      }
      break;
    case 2:
      if (best_workload != 0)
        return stopTuning = true;
      break;
    case 3:
      if ((c - b == limit) && (b - a == limit))
        return stopTuning = true;
      break;
    default: /* Should never happen */;
  }
  return false;
}

void Autotuner::Tune(double compTime)
{
  switch (algo)
  {
    case 1:
      AlgoSimple(compTime);
      break;
    case 2:
      AlgoRatio(compTime);
      break;
    case 3:
      AlgoBisection(compTime);
      break;
    default: /* Should never happen */;
  }
}

void Autotuner::AlgoSimple(double compTime)
{
  if (best_time == 0. || compTime < best_time)
  {
    best_time = compTime;
    best_workload = workload;
  }

  workload -= 5;
}

void Autotuner::AlgoRatio(double compTime)
{
  if (best_time == 0.)
  {
    best_time = compTime;
    workload = 1;
  }
  else
  {
    best_workload = (int) 100 * (compTime / (best_time + compTime));
    workload = best_workload;
  }
}

void Autotuner::AlgoBisection(double compTime)
{
  if (fb == 0.)
  {
    fb = compTime;
    x = 75;
    workload = x;
    return;
  }

  fx = compTime;

  if (fx < fb)
  {
    if (x < b)
      c = b;
    else
      a = b;
    b = x;
    fb = fx;
  }
  else
  {
    if (x < b)
      a = x;
    else
      c = x;
  }

  x = (c - b > b - a) ? (int) (b + (c - b) / 2) : (int) (a + (b - a + 1) / 2);
  workload = x;
}
