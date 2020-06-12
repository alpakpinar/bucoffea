#!/usr/bin/env python

import os
import sys
import re
import argparse
from klepto.archives import dir_archive
from collections import OrderedDict

from pprint import pprint

def compare_cutflows(acc_old, acc_new, dataset_tag, year, region='sr'):
    '''Compare the cutflows of two datasets which are ran with different processor versions.'''
    # Load the cutflows for comparison
    acc_old.load(f'cutflow_{region}_vbf')
    c_old = acc_old[f'cutflow_{region}_vbf']
    acc_new.load(f'cutflow_{region}_vbf')
    c_new = acc_new[f'cutflow_{region}_vbf']

    # Get the values for the right datasets 
    cutflow_old = {dataset: c_old[dataset] for dataset in c_old.keys() if dataset.startswith(dataset_tag) and str(year) in dataset}
    cutflow_new = {dataset: c_new[dataset] for dataset in c_new.keys() if dataset.startswith(dataset_tag) and str(year) in dataset}

    # Combine the HT bins and/or extensions
    grouped_cutflow_old = {} 
    grouped_cutflow_new = {} 

    for cutflow in cutflow_old.values():
        for cut in cutflow:
            if cut in grouped_cutflow_old.keys():
                grouped_cutflow_old[cut] += cutflow[cut]
            else:
                grouped_cutflow_old[cut] = cutflow[cut]

    for cutflow in cutflow_new.values():
        for cut in cutflow:
            if cut in grouped_cutflow_new.keys():
                grouped_cutflow_new[cut] += cutflow[cut]
            else:
                grouped_cutflow_new[cut] = cutflow[cut]

    # Sort the dictionaries!
    sorted_list_old = sorted(grouped_cutflow_old.items(), key=lambda item: item[1], reverse=True)
    grouped_cutflow_old_sorted = OrderedDict()
    for cut, val in sorted_list_old:
        grouped_cutflow_old_sorted[cut] = val

    sorted_list_new = sorted(grouped_cutflow_new.items(), key=lambda item: item[1], reverse=True)
    grouped_cutflow_new_sorted = OrderedDict()
    for cut, val in sorted_list_new:
        grouped_cutflow_new_sorted[cut] = val

    pprint(grouped_cutflow_old_sorted)
    pprint(grouped_cutflow_new_sorted)

def main():
    inpath_old = sys.argv[1]
    inpath_new = sys.argv[2]

    acc_old = dir_archive(
                    inpath_old,
                    serialized=True,
                    compression=0,
                    memsize=1e3
                    )

    acc_new = dir_archive(
                    inpath_new,
                    serialized=True,
                    compression=0,
                    memsize=1e3
                    )

    compare_cutflows(acc_old, acc_new, year=2017, dataset_tag='ZJetsToNuNu')
    compare_cutflows(acc_old, acc_new, year=2018, dataset_tag='ZJetsToNuNu')

if __name__ == '__main__':
    main()
