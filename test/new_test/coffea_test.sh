#!/bin/bash

# First, run the VBF processor to get coffea output via run_quick.py script
cd ../../bucoffea/scripts;
echo "===================="
echo "Running run_quick.py"
echo "===================="
./run_quick.py vbfhinv

# Get the coffea output from run_quick, feed it into the checking script

outputfilename="run_quick_output.txt"
run_quick_output=$(head -n 1 ${outputfilename})
# Then run the python testing script on the output
echo "===================="
echo "Running coffea checker"
echo "===================="
echo "I am here:"
cd -;
py.test 
# python ./check_coffea_histograms.py ${run_quick_output}


