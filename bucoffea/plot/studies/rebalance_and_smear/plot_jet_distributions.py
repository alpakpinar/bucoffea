#!/usr/bin/env python

import os
import sys
import re
import numpy as np

from coffea import hist
from bucoffea.plot.util import merge_datasets, merge_extensions
from matplotlib import pyplot as plt
from klepto.archives import dir_archive
from pprint import pprint

pjoin = os.path.join

new_x_labels = {
    'ak4_pt0' : r'Leading Jet $p_T$ (GeV)',
    'ak4_pt1' : r'Trailing Jet $p_T$ (GeV)',
    'ak4_eta0' : r'Leading Jet $\eta$',
    'ak4_eta1' : r'Trailing Jet $\eta$',
}

def plot_jet_distributions(acc, outtag, distribution, trigger, years=[2017, 2018]):
    '''Plot the given distribution for the given trigger selection.'''
    acc.load(distribution)
    h = acc[distribution]

    h = merge_extensions(h, acc, reweight_pu=False)
    h = merge_datasets(h)

    # Output directory to save these plots
    outdir = f'./output/{outtag}/jet_distributions'
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # Take the region with the specific trigger
    h = h.integrate('region', f'r_{trigger}')

    for year in years:
        _h = h.integrate('dataset', f'JetHT_{year}')
        fig, ax = plt.subplots()

        hist.plot1d(_h, ax=ax)

        ax.set_yscale('log')
        ax.set_ylim(1e0, 1e5)

        ax.text(0., 1., trigger,
            fontsize=14,
            horizontalaligment='left',
            verticalalignment='bottom',
            transform=ax.transAxes
        )

        ax.text(1., 1., year,
            fontsize=14,
            horizontalaligment='left',
            verticalalignment='bottom',
            transform=ax.transAxes
        )

        # Save figure
        outpath = pjoin(outdir, f'{trigger}_{distribution}_{year}.pdf')
        fig.savefig(outpath)
        plt.close(fig)
        print(f'File saved: {outpath}')

def plot_jet_distributions_on_same_figure(acc, outtag, distribution, trigger_tag, trigger_regex, years=[2017,2018]):
    '''Plot the distribution of given quantity for the events passing the set of triggers given (one at a time).'''
    acc.load(distribution)
    h = acc[distribution]

    h = merge_extensions(h, acc, reweight_pu=False)
    h = merge_datasets(h)

    # Output directory to save these plots
    outdir = f'./output/{outtag}/jet_distributions'
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    for year in years:
        # Pick the dataset for the given year and relevant triggers (as specified by the regex)
        _h = h.integrate('dataset', f'JetHT_{year}')[re.compile(trigger_regex)]

        fig, ax = plt.subplots()
        hist.plot1d(_h, ax=ax, overlay='region')
    
        ax.set_yscale('log')
        ax.set_ylim(1e0,1e8)

        ax.text(0., 1., f'JetHT Dataset {year}',
            fontsize=14,
            horizontalalignment='left',
            verticalalignment='bottom',
            transform=ax.transAxes
        )

        # Update legend labels
        handles, labels = ax.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
            new_label = label.replace('r_', '')
            handle.set_label(new_label)

        ax.legend(title='Trigger', handles=handles)

        if distribution in new_x_labels.keys():
            ax.set_xlabel(new_x_labels[distribution])

        # Save figure
        outpath = pjoin(outdir, f'{trigger_tag}_{distribution}_{year}.pdf')
        fig.savefig(outpath)
        plt.close(fig)
        print(f'File saved: {outpath}')

def main():
    inpath = sys.argv[1]
    acc = dir_archive(inpath)
    acc.load('sumw')
    acc.load('sumw2')

    outtag = re.findall('merged_.*', inpath)[0].replace('/','')

    distributions = [
        'mjj',
        'detajj',
        'dphijj',
        'ak4_pt',
        'ak4_pt0',
        'ak4_pt1',
        'ak4_eta0',
        'ak4_eta1',
    ]

    # List of triggers used for event selection
    triggers = [
        'HLT_PFJet40',
        'HLT_PFJet60',
        'HLT_PFJet80',
        'HLT_PFJet140',
        'HLT_PFHT180',
        'HLT_PFHT250',
        'HLT_PFHT370',
    ]

    for distribution in distributions:
        trigger_regexes = {
            'hlt_pfjet' : 'r_HLT_PFJet.*',
            'hlt_pfht'  : 'r_HLT_PFHT.*',
        }
        for trigger_tag, trigger_regex in trigger_regexes.items():
            # Just 2017 for now
            plot_jet_distributions_on_same_figure(acc, outtag, 
                distribution=distribution,
                trigger_tag=trigger_tag,
                trigger_regex=trigger_regex,
                years=[2017]
            )
        # for trigger in triggers:
            # Run only on JetHT 2017 dataset for now
            # plot_jet_distributions(acc, outtag, distribution=distribution, trigger=trigger, years=[2017])

if __name__ == '__main__':
    main()