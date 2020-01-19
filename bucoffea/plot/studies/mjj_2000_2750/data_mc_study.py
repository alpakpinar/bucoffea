#!/usr/bin/env python

######################
# Generate .txt files by calling
# excess_data_events.py for different
# distributions over a dataset.
######################

import sys
import subprocess

def main():
    inpath, region = sys.argv[1:3]
    distributions = [
        'mjj',
        'detajj',
        'dphijj',
        'recoil',
        'ak4_pt',
        'ak4_pt0',
        'ak4_eta',
        'ak4_eta0',
        'ak4_phi',
        'photon_pt0'
    ]

    for dist in distributions:
        cmd = ['./excess_data_events.py', '-i', inpath, '-r', region, '-d', dist]

        p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
        print(p.communicate())
        


if __name__ == '__main__':
    main()



