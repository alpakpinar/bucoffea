#!/usr/bin/env python

import os
import sys
import re
from bucoffea.plot.util import merge_datasets, merge_extensions, scale_xs_lumi, fig_ratio
from coffea import hist
from matplotlib import pyplot as plt
from klepto.archives import dir_archive

pjoin = os.path.join

def get_dataset_regex(tag, year):
    '''Given the dataset tag, get the dataset regex to look for.'''
    tag_to_regex = {
        'vbf' : 'VBF_HToInv.*M125.*{}',
        'ggh' : 'GluGlu_HToInv.*M125.*{}',
    }

    return re.compile( tag_to_regex[tag].format(year) )

def compare_ggh_vbf(acc, outtag, tag='vbf', distribution='met', region='inclusive'):
    '''Compare signal samples between 2016 and 2017.'''
    acc.load(distribution)
    h = acc[distribution]

    h = merge_extensions(h, acc, reweight_pu=False)
    scale_xs_lumi(h)
    h = merge_datasets(h)

    h = h.integrate('region', region)

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
    
    line_opts = {
        'color' : 'crimson'
    }

    fig, ax, rax = fig_ratio()
    hist.plot1d(h_2016, ax=ax, overlay='dataset', error_opts=data_err_opts)
    hist.plot1d(h_2017, ax=ax, overlay='dataset', line_opts=line_opts, clear=False)

    ax.set_xlabel('')
    ax.set_yscale('log')
    ax.set_ylim(1e-3, 1e5)

    # Plot the 2016 / 2017 ratio on the bottom
    hist.plotratio(
        h_2016.integrate('dataset'),
        h_2017.integrate('dataset'),
        ax=rax,
        error_opts=data_err_opts
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

    regions = ['inclusive', 'sr_vbf_no_veto_all']
    distributions = ['met', 'gen_met']
    for tag in ['ggh', 'vbf']:
        for region in regions:
            for distribution in distributions:
                compare_ggh_vbf(acc, outtag, tag=tag, distribution=distribution, region=region)

if __name__ == '__main__':
    main()