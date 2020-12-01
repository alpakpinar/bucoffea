#!/usr/bin/env python

import os
import sys
import re
import numpy as np
from coffea import hist
from bucoffea.plot.util import merge_datasets, merge_extensions, scale_xs_lumi, fig_ratio
from matplotlib import pyplot as plt
from klepto.archives import dir_archive
from pprint import pprint

pjoin = os.path.join

def get_title(region, year, plot='data'):
    mapping = {
        'sr_vbf' : r'{} {}: Signal Region',
        'cr_1m_vbf' : r'{} {}: $1\mu$ CR',
        'cr_1e_vbf' : r'{} {}: $1e$ CR',
        'cr_g_vbf' : r'{} {}: $\gamma$ CR',
    }

    dataset_tags = {
        'data' : 'Data',
        'mc' : 'Total Background (MC)'
    }

    for regex, titletemp in mapping.items():
        if re.match(regex, region):
            return titletemp.format(dataset_tags[plot], year)

    raise RuntimeError(f'Could not find title for: {region}, {year}')

def get_new_legend_label(oldlabel):
    '''Get prettier legend labels for the comparison plot.'''
    newlabels = {
        '(c|s)r_vbf(_no_veto_all)?$' : 'PU Inclusive',
        '(c|s)r_vbf(_no_veto_all)?_nvtx_lt_20' : r'$N_{PV} \leq 20$',
        '(c|s)r_vbf(_no_veto_all)?_nvtx_btw_30_60' : r'$30 \leq N_{PV} \leq 60$',
        '(c|s)r_vbf(_no_veto_all)?_nvtx_ht_30' : r'$N_{PV} \geq 30$',
    }

    for label, newlabel in newlabels.items():
        if re.match(label, oldlabel):
            return newlabel
    
    raise RuntimeError(f'Could not find legend label for: {oldlabel}')

def compare_low_pu_high_pu(acc, outtag, region='sr_vbf', distribution='mjj', plot='data'):
    '''For the given variable, compare the distribution at low PU and high PU (in data).'''
    acc.load(distribution)
    h = acc[distribution]

    h = merge_extensions(h, acc, reweight_pu=False)
    scale_xs_lumi(h)
    h = merge_datasets(h)

    if distribution == 'mjj':
        mjj_ax = hist.Bin('mjj', r'$M_{jj} \ (GeV)$', [200., 400., 600., 900., 1200., 1500.,2000., 2750., 3500., 5000.])
        h = h.rebin('mjj', mjj_ax)
            
    for year in [2017, 2018]:
        data = {
            'sr_vbf' : f'MET_{year}',
            'cr_1m_vbf' : f'MET_{year}',
            'cr_1e_vbf' : f'EGamma_{year}',
            'cr_g_vbf' : f'EGamma_{year}',
        }
        
        mc = {
            'sr_vbf' : re.compile(f'(ZJetsToNuNu.*|EW.*|Top_FXFX.*|Diboson.*|.*DYJetsToLL_M-50_HT_MLM.*|.*WJetsToLNu.*HT.*).*{year}'),
            'cr_1m_vbf' : re.compile(f'(EWKW.*|Top_FXFX.*|Diboson.*|.*DYJetsToLL_M-50_HT_MLM.*|.*WJetsToLNu.*HT.*).*{year}'),
            'cr_1e_vbf' : re.compile(f'(EWKW.*|Top_FXFX.*|Diboson.*|.*DYJetsToLL_M-50_HT_MLM.*|.*WJetsToLNu.*HT.*).*{year}'),
            'cr_g_vbf' : re.compile(f'(GJets_(DR-0p4|SM).*|QCD_data.*|WJetsToLNu.*HT.*).*{year}'),
        }

        if plot == 'data':
            dataset = data[region]
        elif plot == 'mc':
            dataset = mc[region]
        else:
            raise ValueError(f'Invalid value for plot argument: {plot}')
        
        # Region tag: For MC, get no veto region (SR)
        regiontag = region
        if region == 'sr_vbf':
            if plot == 'mc':
                regiontag = 'sr_vbf_no_veto_all'
        
        _h = h.integrate('dataset', dataset)[re.compile(f'{regiontag}(_nvtx)?.*(?<!all)$')]

        pprint(_h.values())

        fig, ax, rax = fig_ratio()
        hist.plot1d(_h, ax=ax, overlay='region', density=True)

        ax.set_yscale('log')
        ax.set_ylim(1e-7,1e1)
        ax.set_ylabel('Normalized Counts')
        ax.set_xlabel('')

        # Handle legend labels
        handles, labels = ax.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
            newlabel = get_new_legend_label(label)
            handle.set_label(newlabel)

        ax.legend(title='Pileup', handles=handles)

        ax.set_title( 
            get_title(region, year, plot=plot),
            fontsize=14
            )

        data_err_opts = {
            'linestyle':'none',
            'marker': '.',
            'markersize': 10.,
        }

        # Plot ratio w.r.t. PU inclusive case
        h_pu_inc = _h.integrate('region', regiontag)
        
        # Normalize the histograms to their integrals
        sumw = np.sum(h_pu_inc.values()[()])
        h_pu_inc.scale(1/sumw)

        for idx, putag in enumerate(['nvtx_btw_30_60', 'nvtx_ht_30', 'nvtx_lt_20']):
            data_err_opts['color'] = f'C{idx+1}'
            htemp = _h.integrate('region', f'{regiontag}_{putag}')
            sumwtemp = np.sum(htemp.values()[()])
            htemp.scale(1/sumwtemp)
            
            hist.plotratio(
                htemp,
                h_pu_inc,
                ax=rax,
                unc='num',
                error_opts=data_err_opts,
                clear=False
            )

        rax.set_xlabel(r'$M_{jj} \ (GeV)$')
        rax.set_ylabel('Ratio to PU inc.')
        rax.set_ylim(0.8,1.2)
        rax.grid(True)

        rax.axhline(1, xmin=0, xmax=1, color='black')

        # Save figure
        outdir = f'./output/{outtag}'
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        outpath = pjoin(outdir, f'{plot}_comp_{region}_{year}.pdf')
        fig.savefig(outpath)
        plt.close(fig)
        print(f'File saved: {outpath}')


def main():
    inpath = sys.argv[1]
    acc = dir_archive(inpath)
    acc.load('sumw')
    acc.load('sumw2')

    outtag = re.findall('merged_.*', inpath)[0].replace('/', '')

    for region in ['sr_vbf']:
        # Plot = 'data' will plot the comparison for data
        # Plot = 'mc' will plot the comparison for total MC in the region
        for plot in ['data', 'mc']:
            compare_low_pu_high_pu(acc, outtag, region=region, plot=plot)

if __name__ == '__main__':
    main()