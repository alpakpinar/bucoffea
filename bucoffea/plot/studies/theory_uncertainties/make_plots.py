#!/usr/bin/env python

from coffea.hist.plot import plot1d
from matplotlib import pyplot as plt
import uproot
import re
import sys
import sys
import numpy as np
import os
from pprint import pprint

pjoin = os.path.join

labels = {
    'num_varied_nlo_muf_up_denom_varied_nlo_muf_down' : r'$\mu_F up$',
    'num_varied_nlo_muf_down_denom_varied_nlo_muf_up' : r'$\mu_F down$',
    'num_varied_nlo_mur_up_denom_varied_nlo_mur_down' : r'$\mu_R up$',
    'num_varied_nlo_mur_down_denom_varied_nlo_mur_up' : r'$\mu_R down$'
}

def make_pretty_plots(infile):
    '''Make pretty plots'''
    f = uproot.open(infile)
    for year in [2017, 2018]:
        fig, ax = plt.subplots()

        for key in f.keys():
            name = key.decode('utf-8')
            if f'{year}' not in name:
                continue

            vals = f[key].values
            edges = f[key].edges
            centers = ((edges + np.roll(edges, -1))/2)[:-1]

            legend_label = None
            for histname, label in labels.items():
                if histname in name:
                    legend_label = label
                    break

            ax.plot(centers, vals, marker='o', label=legend_label)
        
        ax.set_xlabel(r'$M_{jj} \ (GeV)$')
        ax.set_ylabel('Combined Z/W scale unc')
        ax.legend()

        fig.savefig(f'test_{year}.pdf')


def main():
    infile = sys.argv[1]
    make_pretty_plots(infile)

if __name__ == '__main__':
    main()