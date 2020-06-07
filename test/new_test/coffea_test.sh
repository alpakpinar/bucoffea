#!/bin/bash

# First, run the VBF processor to get coffea output via run_quick.py script
cd bucoffea/scripts;
echo "===================="
echo "Running run_quick.py"
echo "===================="
./run_quick.py vbfhinv

# Then run the python testing script on the output
cd -;
cd test/new_test;
python ./check_coffea_histograms.py


