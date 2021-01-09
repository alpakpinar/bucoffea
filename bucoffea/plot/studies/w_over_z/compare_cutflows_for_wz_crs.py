#!/usr/bin/env python

import os
import sys
import re
import numpy as np

from coffea import hist
from coffea.processor import defaultdict_accumulator
from matplotlib import pyplot as plt
from klepto.archives import dir_archive
from tabulate import tabulate
from pprint import pprint

pjoin = os.path.join

def merge_cutflows(cflow_z, cflow_w, year):
    # Merge DYJetsToLL cutflows
    combined_cflow_z = defaultdict_accumulator(int)
    for dataset, cutflow in cflow_z.items():
        if not re.match(f'DYJetsToLL.*{year}', dataset):
            continue
        combined_cflow_z.add(cutflow)

    # Merge WJetsToLNu cutflows
    combined_cflow_w = defaultdict_accumulator(int)
    for dataset, cutflow in cflow_w.items():
        if not re.match(f'WJetsToLNu.*{year}', dataset):
            continue
        combined_cflow_w.add(cutflow)

    return combined_cflow_z, combined_cflow_w

def dump_cutflows_for_wz_crs(acc, outtag, channel='muons', year=2017):
    '''For the given year and channel, dump the cutflows for W and Z control regions into an output txt file.'''
    regions = {
        'muons' : ['cr_2m_vbf', 'cr_1m_vbf'],
        'electrons' : ['cr_2e_vbf', 'cr_1e_vbf'],
    }
    acc.load(f'cutflow_{regions[channel][0]}')
    acc.load(f'cutflow_{regions[channel][1]}')

    cflow_z = acc[f'cutflow_{regions[channel][0]}']
    cflow_w = acc[f'cutflow_{regions[channel][1]}']

    combined_cflow_z, combined_cflow_w = merge_cutflows(cflow_z, cflow_w, year)

    outdir = f'./output/{outtag}/from_acc/cutflow_comparison'
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    outfile = pjoin(outdir, f'cutflow_{channel}_{year}.txt')
    with open(outfile, 'w+') as f:

        z_table = []
        z_counts = []
        f.write('---\n')
        f.write(f'DYJetsToLL {year}\n')
        f.write('---\n')
        for idx, (cut, count) in enumerate(sorted(combined_cflow_z.items(), key=lambda x:x[1], reverse=True)):
            z_counts.append(count)
            z_table.append([cut, count, (z_counts[idx] / z_counts[idx-1])*100 ])

        f.write(tabulate(z_table, headers=["Cut", "Passing events", "Efficiency (%)"], floatfmt=(".1f", ".1f", ".2f")))
        f.write('\n')

        w_table = []
        w_counts = []
        f.write('---\n')
        f.write(f'WJetsToLNu {year}\n')
        f.write('---\n')
        for idx, (cut, count) in enumerate(sorted(combined_cflow_w.items(), key=lambda x:x[1], reverse=True)):
            w_counts.append(count)
            w_table.append([cut, count, (w_counts[idx] / w_counts[idx-1])*100])

        f.write(tabulate(w_table, headers=["Cut", "Passing events", "Efficiency (%)"], floatfmt=(".1f", ".1f", ".2f")))
    
    print(f'File saved: {outfile}')

def main():
    inpath = sys.argv[1]
    acc = dir_archive(inpath)
    acc.load('sumw')
    acc.load('sumw2')

    outtag = re.findall('merged_.*', inpath)[0].replace('/', '')

    dump_cutflows_for_wz_crs(acc, outtag)

if __name__ == '__main__':
    main()