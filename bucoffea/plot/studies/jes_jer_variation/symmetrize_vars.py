#!/usr/bin/env python

import os
import sys
import re
import uproot
import numpy as np

from matplotlib import pyplot as plt

pjoin = os.path.join

def symmetrize_variations(infile, outfile):
    for year in [2017, 2018]:
        nuisances = [
            'jer',
            'jesAbsolute',
            f'jesAbsolute_{year}',
            'jesBBEC1',
            f'jesBBEC1_{year}',
            'jesEC2',
            f'jesEC2_{year}',
            'jesFlavorQCD',
            'jesHF',
            f'jesHF_{year}',
            'jesRelativeBal',
            f'jesRelativeSample_{year}',
        ]

        year_tag = f'{int(year)-2000}'
        transfer_factors = [
            f'znunu_over_wlnu{year_tag}',
            f'znunu_over_zmumu{year_tag}',
            f'znunu_over_zee{year_tag}',
            f'znunu_over_gjets{year_tag}',
            f'wlnu_over_wenu{year_tag}',
            f'wlnu_over_wmunu{year_tag}',
        ]

        for transfer_factor in transfer_factors:
            for nuisance in nuisances:
                tag = f'{transfer_factor}_qcd_{nuisance}'
                orig_up = infile[f'{tag}Up'].values[0]
                orig_down = infile[f'{tag}Down'].values[0]

                edges = infile[f'{tag}Up'].edges

                # Decide which one is bigger (i.e. "up" variation)
                if abs(orig_up-1) > abs(orig_down-1):
                    direction = 1
                else:
                    direction = -1

                # New uncertainty is the symmetrized one
                new_unc = 0.5 * (abs(orig_up-1) + abs(orig_down-1))

                outfile[f'{tag}Up'] = (np.array([1 + new_unc * direction]), edges)

                outfile[f'{tag}Down'] = (np.array([1 - new_unc * direction]), edges)

                assert np.isclose(abs(outfile[f'{tag}Up'].values-1), abs(outfile[f'{tag}Down'].values-1))

def main():
    # Input ROOT file containing original variations
    inpath = './output/merged_2020-09-29_monojet_splitJECuncs_25Aug20/splitJEC/monojet/root/monojet_jes_jer_tf_uncs.root'
    infile = uproot.open(inpath)

    # Specify the output root file, where the new variations will be saved
    outpath = './output/merged_2020-09-29_monojet_splitJECuncs_25Aug20/splitJEC/monojet/root/monojet_jes_jer_tf_uncs_symmetrized.root'
    outfile = uproot.recreate(outpath)

    symmetrize_variations(infile, outfile)

if __name__ == '__main__':
    main()