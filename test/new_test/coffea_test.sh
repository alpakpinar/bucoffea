#!/bin/bash

# First, run the VBF processor to get coffea output via run_quick.py script
# cd ../../bucoffea/scripts;
# echo "===================="
# echo "Running run_quick.py"
# echo "===================="
# ./run_quick.py vbfhinv

while IFS=" " read -r DATASET REMOTE_PATH REMAINDER
do
    FNAME=${DATASET}.root
    if [[ ! -f $FNAME ]]; then
        wget ${REMOTE_PATH}/${FNAME}
    fi
    echo ${FNAME} > files.txt
    buexec vbfhinv --outpath ./output_newtest/ --jobs 1 worker --dataset ${DATASET} --filelist files.txt --chunk 0
done < "testfiles_new.txt"

# Get the coffea output from run_quick, feed it into the checking script

# outputfilename="run_quick_output.txt"
# run_quick_output=$(head -n 1 ${outputfilename})
# Then run the python testing script on the output
echo "===================="
echo "Running coffea checker"
echo "===================="
cd output_newtest;
ls;
for filename in ./*.coffea
do
    python ../check_coffea_histograms.py $(basename ${filename})
done


