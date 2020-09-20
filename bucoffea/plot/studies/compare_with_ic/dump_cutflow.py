#!/usr/bin/env python

import os
import sys
import re
import tabulate
import numpy as np
from klepto.archives import dir_archive
from pprint import pprint

pjoin = os.path.join

def dump_cutflow(acc, dataset):
    '''Dump the cutflow for the signal region cuts out of the given accumulator, into an output txt file'''
    acc.load('cutflow_sr_vbf')
    cf = acc['cutflow_sr_vbf'][dataset]

    # Tabulate the cutflow
    table = []
    for cut, count in sorted(cf.items(), key=lambda x:x[1], reverse=True):
        table.append([cut, count])

    text = tabulate.tabulate(table, headers=["Cut", "Passing events"], floatfmt=".1f")

    # Save the table into an output file
    outdir = './output/cutflow'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    outpath = pjoin(outdir, 'bu_cutflow.txt')

    with open(outpath, 'w+') as f:
        f.write(f'{dataset}\n')
        f.write('\n')
        f.write(text)
    
    print(f'MSG% Cutflow saved in: {outpath}')

def main():
    inpath = sys.argv[1]
    acc = dir_archive(inpath)

    dataset = 'ZJetsToNuNu_HT-200To400-mg_2017'

    dump_cutflow(acc, dataset)

if __name__ == '__main__':
    main()
