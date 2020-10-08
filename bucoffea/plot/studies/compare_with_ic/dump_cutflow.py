#!/usr/bin/env python

import os
import sys
import re
import tabulate
import numpy as np
from coffea.util import load
from pprint import pprint

pjoin = os.path.join

pretty_names_for_cuts = {
    'all' : 'All Events',
    'trig_met' : 'MET trigger requirement',
    'metphihemextveto' : 'MET phi HEM veto for 2018',
    'hornveto' : 'Jet horn veto',
    'veto_ele' : 'Electron veto',
    'veto_muo' : 'Muon veto',
    'filt_met' : 'MET cleaning filters',
    'mindphijr' : 'min(Delta phi jet MET) > 0.5',
    'recoil' : 'Recoil > 250 GeV',
    'two_jets' : 'Exactly two jets requirement',
    'leadak4_pt_eta' : 'Leading jet pt&eta cut',
    'leadak4_id' : 'Leading jet ID cut',
    'trailak4_pt_eta' : 'Trailing jet pt&eta cut',
    'trailak4_id' : 'Trailing jet ID cut',
    'hemisphere' : 'Opposite hemispheres for two leading jets',
    'mjj' : 'mjj cut',
    'dphijj' : 'dphijj cut',
    'detajj' : 'detajj cut',
    'veto_photon' : 'Photon veto',
    'veto_tau' : 'Tau veto',
    'veto_b' : 'b-jet veto',
    'dpfcalo_sr' : '(PF MET - Calo MET) cut',
    'eemitigation' : 'VecB/VecDPhi cuts',
    'max_neEmEF' : 'Jet EM energy fracion cut',
    'veto_hfhf' : 'HF-HF veto'
}

def dump_cutflow(acc, outtag, dataset):
    '''Dump the cutflow for the signal region cuts out of the given accumulator, into an output txt file'''
    cf = acc['cutflow_sr_vbf'][dataset]

    # Tabulate the cutflow
    table = []
    for cut, count in sorted(cf.items(), key=lambda x:x[1], reverse=True):
        table.append([pretty_names_for_cuts[cut], count])

    text = tabulate.tabulate(table, headers=["Cut", "Passing events"], floatfmt=".1f")

    # Save the table into an output file
    outdir = f'./output/cutflow/{outtag}'
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
    acc = load(inpath)

    outtag = inpath.split('/')[-2]

    # Figure out the dataset from the filename
    filename = inpath.split('/')[-1]
    dataset = filename.replace('tree_', '').replace('.root', '')

    print(f'File being used: {filename}')
    print(f'Dataset: {dataset}')

    dump_cutflow(acc, outtag, dataset)

if __name__ == '__main__':
    main()
