### run in /bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -N bioem-test-array
#$ -l h_rt=00:30:00
### Allocating nodes with 2 GPUs each, on "phys" cluster
#$ -pe impi_hydra 24
#$ -P gpu
#$ -l use_gpus=1
#$ -l type_gpus=gtx1080
### Allocation job-array for 20 jobs
#$ -t 1-20

set -e

# Local ID of the job
x=$SGE_TASK_ID
SGE_TASK_COUNT=$((($SGE_TASK_LAST - $SGE_TASK_FIRST + 1) / $SGE_TASK_STEPSIZE))

# Loading necessary libraries
module purge
module load intel impi cuda python33/python/3.3 python33/scipy/2015.10

# Variables
#TMPDIR="/tmp/"
M=M2
MODEL=MODEL_2
GRID=smallGrid_125

PYTHON=python3.3
INPUTPROB=Output_${MODEL}
ORIENTATIONS=${TMPDIR}/Quaternion_List_${M}_P${x}
TMP1=${TMPDIR}/base${x}
TMP2=${TMPDIR}/ll${x}
OUTPUTPROB=${TMPDIR}/Output_${MODEL}_P${x}
FINALOUTTMP=Output_Tmp
FINALOUTPROB=Output_${MODEL}_Round2

##########################################################
##########################################################
##########################################################
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

# Path to your BioEM installation
BIOEM=${HOME}/BioEM_project/build/bioEM

# Running BioEM
OMP_NUM_THREADS=12 GPU=1 GPUDEVICE=-1 BIOEM_DEBUG_OUTPUT=0 BIOEM_ALGO=2 mpiexec -perhost 2 ${BIOEM}  --Modelfile ${MODEL} --Particlesfile Particles/Particle_$x --Inputfile Param_Input_ModelComparision --ReadOrientation ${ORIENTATIONS} --OutputFile ${OUTPUTPROB}

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

# If this was the last job, sort everything and remove JobID
y=2
if [[ $(wc -l < ${FINALOUTTMP}) == $(( $y * $SGE_TASK_COUNT )) ]]
then
    sort -n -k 2 ${FINALOUTTMP} | sed 's/Job: [0-9]\+ //' > ${FINALOUTPROB}
    rm -rf ${FINALOUTTMP}
fi
