### to run:
# ./create_gridOr.sh $1 $2

## $1 == Output file from previous BioEM round
## $2 == Name to put on lists

# necessary python libraries:

module load python33/python/3.3 python33/scipy/2015.10

PYTHON=python3.3
TMPDIR=/tmp/
INDIR=${TMPDIR}/Quaternion_Lists_${2}
mkdir -p ${INDIR}
OUTPUTFILE=${INDIR}/Quaternion_List_${2}_P
TMP1=${TMPDIR}/base
TMP2=${TMPDIR}/ll

#Change the variable for the total number of images
numim=20 

for((y=1;y<${numim}+1;y++))
do

#Extracting the best orientation from the output file from column 1
grep Maxi $1 | grep -v Notat | awk -v x=$y '{if(NR==x)print $6,$8,$10,$12}' > ${TMP1}

#Using the python numpy library to multiply the quaternions
$PYTHON ./multiply_quat.py ${TMP1} smallGrid_125 > ${TMP2}

#generating the list of quaternions around the best orientation from base
echo 125 > ${OUTPUTFILE}${y}

sed 's/,/ /g' ${TMP2} |  sed 's/)/ /g' |  sed 's/(/ /g' | awk '{printf"%12.8f%12.8f%12.8f%12.8f\n",$1,$2,$3,$4}' >> ${OUTPUTFILE}${y}

echo "Finished ${y}/${numim}"

rm ${TMP1} ${TMP2}
done
