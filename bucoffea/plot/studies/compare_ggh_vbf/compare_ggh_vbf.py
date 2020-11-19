#!/usr/bin/env python

import os
import sys
import re
from bucoffea.plot.util import merge_datasets, merge_extensions, scale_xs_lumi, fig_ratio
from coffea import hist
from matplotlib import pyplot as plt
from klepto.archives import dir_archive
from pprint import pprint

pjoin = os.path.join

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
    met_ax = hist.Bin('met',r'$p_{T}^{miss}$ (GeV)',list(range(0,500,50)) + list(range(500,1000,100)) + list(range(1000,2000,250)))
    h = h.rebin('met', met_ax)
    return h

def compare_ggh_vbf(acc, outtag, tag='vbf', distribution='met', region='inclusive'):
    '''Compare signal samples between 2016 and 2017.'''
    acc.load(distribution)
    h = acc[distribution]

    h = merge_extensions(h, acc, reweight_pu=False)
    h = merge_datasets(h)

    h = h.integrate('region', region)
    h = do_rebinning(h)

    # Get 2016 and 2017 datasets
    h_2016 = h[ get_dataset_regex(tag, year=2016) ]
    h_2017 = h[ get_dataset_regex(tag, year=2017) ]

    # Plot the comparison!
    data_err_opts = {
        'linestyle':'none',
        'marker': '.',
        'markersize': 10.,
        'color':'k',
    }
    
    fig, ax, rax = fig_ratio()
    hist.plot1d(h_2016, ax=ax, overlay='dataset', error_opts=data_err_opts)
    hist.plot1d(h_2017, ax=ax, overlay='dataset', clear=False)

    ax.set_xlabel('')
    ax.set_yscale('log')
    ax.set_ylim(1e-5, 1e1)
    ax.set_title( get_title(tag) )

    # Plot the 2016 / 2017 ratio on the bottom
    hist.plotratio(
        h_2016.integrate('dataset'),
        h_2017.integrate('dataset'),
        ax=rax,
        unc='num',
        error_opts=data_err_opts
    )

    ax.text(1., 1., region,
                fontsize=10,
                horizontalalignment='right',
                verticalalignment='bottom',
                transform=ax.transAxes
            )

    rax.set_ylabel('2016 / 2017')
    rax.set_ylim(0.8,1.2)
    rax.grid(True)
    if distribution == 'met':
        rax.set_xlabel('MET (GeV)')
    else:
        rax.set_xlabel('GenMET (GeV)')

    # Save figure
    outdir = f'./output/{outtag}'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    outpath = pjoin(outdir, f'{tag}_{distribution}_{region}.pdf')
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

    regions = ['inclusive', 'sr_vbf']
    distributions = ['met', 'gen_met']
    for tag in ['ggh', 'vbf']:
        for region in regions:
            for distribution in distributions:
                compare_ggh_vbf(acc, outtag, tag=tag, distribution=distribution, region=region)

if __name__ == '__main__':
    main()