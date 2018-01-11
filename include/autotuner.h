/* ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
   < BioEM software for Bayesian inference of Electron Microscopy images>
   Copyright (C) 2017 Pilar Cossio, Markus Rampp, Luka Stanisic and Gerhard
   Hummer.
   Max Planck Institute of Biophysics, Frankfurt, Germany.
   Max Planck Computing and Data Facility, Garching, Germany.

   Released under the GNU Public License, v3.
   See license statement for terms of distribution.

   ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/

#ifndef AUTOTUNER_H
#define AUTOTUNER_H

class Autotuner
{

public:
  Autotuner() { stopTuning = true; }

  /* Setting variables to initial values */
  inline void Initialize(int alg = 3, int st = 7)
  {
    algo = alg;
    stable = st;
    Reset();
  }

  /* Resetting variables to initial values */
  void Reset();

  /* Check if autotuning is needed, depending on which comparison is finished */
  bool Needed(int iteration);

  /* Check if optimal workload value has been computed */
  bool Finished();

  /* Set a new workload value to test, depending on the algorithm */
  void Tune(double compTime);

  /* Return workload value */
  inline int Workload() { return workload; }

private:
  int algo;
  int stable;

  bool stopTuning;
  int workload;

  /* Variables needed for AlgoSimple and AlgoRatio */
  double best_time;
  int best_workload;

  /* Variables needed for AlgoBisection */
  int a;
  int b;
  int c;
  int x;
  int limit;
  double fb, fx;

  /* Autotuning algorithms */
  void AlgoSimple(double compTime);
  void AlgoRatio(double compTime);
  void AlgoBisection(double compTime);
};

#endif
