#!/usr/bin/env python

import os
import sys
import re
import warnings
import numpy as np
from bucoffea.plot.util import fig_ratio
from coffea import hist
from matplotlib import pyplot as plt
from klepto.archives import dir_archive
from pprint import pprint

pjoin = os.path.join

warnings.filterwarnings('ignore')

def get_dataset_names(tag1, tag2):
    '''Given the two dataset tags to compare, return the names of these.'''
    datasets_to_compare = {
        'ggh_inc_2016'  : 'GluGlu_HToInvisible_M125_pow_pythia8_2016',
        'ggh_filt_2016' : 'GluGlu_HToInvisible_M125_TuneCP5_pow_pythia8_2016',
        'ggh_filt_2017' : 'GluGlu_HToInvisible_M125_HiggspTgt190_TuneCP5_pow_pythia8_2017',
        'vbf_inc_2016'  : 'VBF_HToInvisible_M125_pow_pythia8_2016',
        'vbf_filt_2017' : 'VBF_HToInvisible_M125_TuneCP5_pow_pythia8_2017'
    }

    return datasets_to_compare[tag1], datasets_to_compare[tag2] 

def get_dataset_regex(tag, year):
    '''Given the dataset tag, get the dataset regex to look for.'''
    tag_to_regex = {
        'vbf' : 'VBF_HToInv.*M125.*{}',
        'ggh' : 'GluGlu_HToInv.*M125_(TuneCP5|HiggspTgt190).*{}',
    }

    return re.compile( tag_to_regex[tag].format(year) )

def get_title(tag):
    tag_to_title = {
        'vbf' : 'VBF H(inv)',
        'ggh' : 'ggH(inv)'
    }

    return tag_to_title[tag]

def do_rebinning(h):
    met_ax = hist.Bin('met',r'$p_{T}^{miss}$ (GeV)',list(range(0,500,50)) + list(range(500,1100,100)))
    h = h.rebin('met', met_ax)
    return h

def compare_datasets(acc, outtag, tag1, tag2, distribution='met', region='inclusive'):
    '''Compare the provided signal samples.'''
    acc.load(distribution)
    h = acc[distribution]

    h = h.integrate('region', region)
    h = do_rebinning(h)

    # Get the individual histograms for the two datasets
    dataset_names = get_dataset_names(tag1, tag2)
    h1 = h[dataset_names[0]]
    h2 = h[dataset_names[1]]


    # Plot the comparison!
    data_err_opts = {
        'linestyle':'none',
        'marker': '.',
        'markersize': 10.,
        'color':'k',
    }
    
    fig, ax, rax = fig_ratio()
    hist.plot1d(h1, ax=ax, overlay='dataset', overflow='over', density=True, error_opts=data_err_opts)
    hist.plot1d(h2, ax=ax, overlay='dataset', overflow='over', density=True, clear=False)

    ax.set_xlabel('')
    ax.set_yscale('log')
    ax.set_ylim(1e-5, 1e1)
    ax.set_ylabel('Normalized Counts')
    # ax.set_title( get_title(tag) )

    # Update legend labels: Use shorter names for datasets
    handles, labels = ax.get_legend_handles_labels()

    # Plot the 2016 / 2017 ratio on the bottom
    h1_integ = h1.integrate('dataset')
    h2_integ = h2.integrate('dataset')

    h1_integ.scale(1/np.sum(h1_integ.values()[()]) ),
    h2_integ.scale(1/np.sum(h2_integ.values()[()]) ),

    hist.plotratio(
        h1_integ,
        h2_integ,
        ax=rax,
        unc='num',
        overflow='over',
        error_opts=data_err_opts
    )

    ax.text(1., 1., region,
                fontsize=12,
                horizontalalignment='right',
                verticalalignment='bottom',
                transform=ax.transAxes
            )

    rax.set_ylabel('2016 / 2017')
    rax.set_ylim(0.5,1.5)
    rax.grid(True)
    rax.axhline(1, xmin=0, xmax=1, color='red')
    
    if distribution == 'met':
        rax.set_xlabel(r'Higgs $p_T$ (GeV)')
    else:
        rax.set_xlabel(r'Higgs GEN-$p_T$ (GeV)')

    # Save figure
    outdir = f'./output/{outtag}'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    outpath = pjoin(outdir, f'{tag1}_{tag2}_{distribution}_{region}.pdf')
    fig.savefig(outpath)
    plt.close(fig)

    print(f'File saved: {outpath}')

def main():
    inpath = sys.argv[1]
    # Accumulator containing the 2016 and 2017 ggH and VBF signal samples
    acc = dir_archive(inpath)
    acc.load('sumw')
    acc.load('sumw2')

    outtag = re.findall('merged_.*', inpath)[0].replace('/', '')

    tags_to_compare = [
        ('ggh_filt_2016', 'ggh_filt_2017'),
        ('vbf_inc_2016', 'vbf_filt_2017'),
        ('ggh_inc_2016', 'ggh_filt_2016'),
    ]

    regions = ['inclusive', 'sr_vbf']
    distributions = ['met', 'gen_met']
    for tag1, tag2 in tags_to_compare:
        for region in regions:
            for distribution in distributions:
                compare_datasets(acc, outtag, 
                            tag1=tag1, 
                            tag2=tag2,
                            distribution=distribution, 
                            region=region)

if __name__ == '__main__':
    main()