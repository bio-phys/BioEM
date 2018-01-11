#!/bin/bash -l
# Standard output and error:
#SBATCH -o ./tjob_hybrid.out.%j
#SBATCH -e ./tjob_hybrid.err.%j
# Initial working directory:
#SBATCH -D ./
# Job Name:
#SBATCH -J bioem-test-array
# Queue (Partition):
#SBATCH --partition=gpu
# Node feature:
#SBATCH --constraint="gpu"
# Number of nodes and MPI tasks per node:
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
# for OpenMP:
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
# Submitting an array
#SBATCH --array=1-20

set -e

echo "Hello: I'm task $SLURM_ARRAY_TASK_ID. SLURM_ARRAY_TASK_MIN=$SLURM_ARRAY_TASK_MIN SLURM_ARRAY_TASK_MAX=$SLURM_ARRAY_TASK_MAX SLURM_ARRAY_TASK_STEP=$SLURM_ARRAY_TASK_STEP"
x=$SLURM_ARRAY_TASK_ID

# The following line is only needed for the older Slurm versions
SLURM_ARRAY_TASK_COUNT=$((($SLURM_ARRAY_TASK_MAX - $SLURM_ARRAY_TASK_MIN + 1) / $SLURM_ARRAY_TASK_STEP))

##########################################################
##########################################################
##########################################################
# Creating a list of quaternions for a specific particle image
# Code similar to the create_gridOr.sh script, only for a single particle image

# Loading Python
module purge
module load anaconda/3/4.3.1

# Variables
#TMPDIR="/tmp/"
M=M1
MODEL=MODEL_1
GRID=smallGrid_125

PYTHON=python3.6
INPUTPROB=Output_${MODEL}
ORIENTATIONS=${TMPDIR}/Quaternion_List_${M}_P${x}
TMP1=${TMPDIR}/base${x}
TMP2=${TMPDIR}/ll${x}
OUTPUTPROB=${TMPDIR}/Output_${MODEL}_P${x}
FINALOUTTMP=Output_Tmp
FINALOUTPROB=Output_${MODEL}_Round2

# Creating list of quaternions
grep Maxi ${INPUTPROB} | grep -v Notat | awk -v y=${x} '{if(NR==y)print $6,$8,$10,$12}' > ${TMP1}

# Using the python numpy library to multiply the quaternions
$PYTHON ./multiply_quat.py ${TMP1} ${GRID} > ${TMP2}

# Generating the list of quaternions around the best orientation from base
echo 125 > ${ORIENTATIONS}

sed 's/,/ /g' ${TMP2} |  sed 's/)/ /g' |  sed 's/(/ /g' | awk '{printf"%12.8f%12.8f%12.8f%12.8f\n",$1,$2,$3,$4}' >> ${ORIENTATIONS}

rm -f ${TMP1} ${TMP2}

##########################################################
##########################################################
##########################################################
# BioEM part Round 2

# Loading necessary modules for Intel compilers
module load intel/17.0
module load impi/2017.3
module load fftw/3.3.6
module load cuda/8.0

# Environment variables
export OMP_NUM_THREADS=16
export OMP_PLACES=cores
export GPUDEVICE=-1
export GPUWORKLOAD=100

# Environment variable to tune
export BIOEM_CUDA_THREAD_COUNT=1024
export BIOEM_DEBUG_OUTPUT=1
export BIOEM_ALGO=2
export GPU=1
export BIOEM_PROJ_CONV_AT_ONCE=16

# Path to your BioEM installation
BIOEM=${HOME}/BioEM_project/build/bioEM

# Running BioEM
mpiexec -perhost 2 ${BIOEM} --Modelfile ${MODEL} --Particlesfile Particles/Particle_$x --Inputfile Param_Input_ModelComparision --ReadOrientation ${ORIENTATIONS} --OutputFile ${OUTPUTPROB} 

##########################################################
##########################################################
##########################################################
# Writing results with a proper text in a proper order
# Note that the order in which jobs are going to finish is undetermined, but the results are written in the good order at the end

# Writing to the temporary shared file
echo Job: ${x} $(tail -2 ${OUTPUTPROB} | head -1 | sed 's/RefMap: 0/RefMap: '${x}'/') >> ${FINALOUTTMP}
echo Job: ${x} $(tail -1 ${OUTPUTPROB} | sed 's/RefMap: 0/RefMap: '${x}'/') >> ${FINALOUTTMP}

# Cleanup
rm -f ${ORIENTATIONS} ${OUTPUTTPROB}

y=2
# If this was the last job, sort everything and remove JobID
if [[ $(wc -l < ${FINALOUTTMP}) == $(( $y * $SGE_TASK_COUNT )) ]]
then
    sort -n -k 2 ${FINALOUTTMP} | sed 's/Job: [0-9]\+ //' > ${FINALOUTPROB}
    rm -rf ${FINALOUTTMP}
fi
