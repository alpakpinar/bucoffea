#!/usr/bin/env python

import os
import re
import argparse
import numpy as np
from matplotlib import pyplot as plt
from klepto.archives import dir_archive
from pprint import pprint 
from coffea import hist
from bucoffea.plot.util import merge_datasets, scale_xs_lumi, merge_extensions

pjoin = os.path.join

fig_title = {
    'DR' : 'GJets_DR-0p4_HT_2016',
    'nonDR' : 'GJets_HT_2016',
    'nlo' : 'G1Jet_Pt_2016'
}

regex = {
    'DR' : 'GJets_DR-0p4.*2016',
    'nonDR' : 'GJets_HT.*2016',
    'nlo' : 'G1Jet.*'
}

def parse_commandline():
    parser = argparse.ArgumentParser()
    parser.add_argument('inpath', help='The path containing input .coffea files.')
    parser.add_argument('-v', '--version', help='Distribution version (inclusive, noDRreq or full')
    parser.add_argument('-s', '--sample', help='Type of GJets sample: DR, nonDR or NLO')
    parser.add_argument('--scale', help='Scale values w.r.t. xs and lumi.', action='store_true')
    args = parser.parse_args()
    return args

args = parse_commandline()

inpath = args.inpath
version = args.version
sample_type = args.sample
scale = args.scale

acc = dir_archive(inpath)

outtag = inpath.split('/')[-1]

acc.load('sumw')
acc.load('sumw2')

if version != 'full':
    dist = f'lhe_mindr_g_parton_stat1_{version}'
else:
    dist = f'lhe_mindr_g_parton_stat1'

acc.load(dist)
h = acc[dist]

h = h[re.compile(regex[sample_type])]
new_dr_bin = hist.Bin('dr', r'$\deltaR_{j,\gamma}$', 50, 0, 5)
h = h.rebin('dr', new_dr_bin)

# Merge extensions
h = merge_extensions(h, acc, reweight_pu=False)
# Scale w.r.t. xs and lumi only if requested
if scale:
    scale_xs_lumi(h)

fig, ax = plt.subplots()
for data, vals in h.values().items():
    dataset_name = data[0]
    if sample_type in ['DR', 'nonDR']:
        match = re.findall(r'HT-\d+To.*', dataset_name)
        label = match[0].replace('-MLM', '').replace('_2016', '')
    elif sample_type == 'nlo':
        match = re.findall(r'Pt-\d+To.*', dataset_name)
        label = match[0].replace('-amcatnlo', '').replace('_2016', '')
        
    ax.step(h.axes()[1].edges()[:-1], vals, where='post', label=label)

ax.legend(ncol=2)
ax.set_xlabel(r'$\Delta R_{j,\gamma}$')
ax.set_xlim(0,5)
if scale:
    ax.set_ylabel('Scaled Counts')
else:
    ax.set_ylabel('Unscaled Counts')

ax.set_title(f'{fig_title[sample_type]} ({version})')

outdir = f'./output/{outtag}/gjets_samples_unmerged'
if not os.path.exists(outdir):
    os.makedirs(outdir)

scale_suffix = '_scaled' if scale else '_unscaled'

outpath = pjoin(outdir, f'gjets_{sample_type}_comp_{version}{scale_suffix}.pdf')

fig.savefig(outpath)
print(f'File saved: {outpath}')

