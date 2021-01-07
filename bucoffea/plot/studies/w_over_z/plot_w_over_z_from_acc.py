#!/usr/bin/env python

import os
import sys
import re
import numpy as np

from coffea import hist
from bucoffea.plot.util import merge_datasets, merge_extensions, scale_xs_lumi, fig_ratio
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
from klepto.archives import dir_archive
from pprint import pprint

pjoin = os.path.join

REBIN = {
    'mjj' : hist.Bin('mjj', r'$M_{jj} \ (GeV)$', [200., 400., 600., 900., 1200., 1500., 2000., 2750., 3500., 5000.])
}

dataset_regex = {
    'qcdZ' : 'DYJetsToLL.*{}',
    'ewkZ' : 'EWKZ2Jets.*ZToLL.*{}',
    'qcdW' : 'WJetsToLNu.*{}',
    'ewkW' : 'EWKW2Jets.*{}',
}

regions = {
    'muons' : ('2m', '1m'),
    'electrons' : ('2e', '1e'),
}

data_err_opts = {
    'linestyle':'none',
    'marker': '.',
    'markersize': 10.,
}

def preprocess(h, acc, distribution):
    h = merge_extensions(h, acc, reweight_pu=False)
    scale_xs_lumi(h)
    h = merge_datasets(h)

    if distribution == 'mjj':
        h = h.rebin('mjj', REBIN['mjj'])

    return h

def get_ylabel(channel):
    ylabels = {
        'muons' : r'$Z(\mu\mu) \ / \ W(\mu\nu)$',
        'electrons' : r'$Z(ee) \ / \ W(e\nu)$',
    }
    return ylabels[channel]

def plot_z_over_w(acc, outtag, channel='muons', distribution='mjj'):
    '''Plot Z/W ratio from the accumulator itself.'''
    acc.load(distribution)
    h = preprocess(acc[distribution], acc, distribution)

    outdir = f'./output/{outtag}/from_acc'
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    for year in [2017, 2018]:
        h_qcdz = h.integrate('region', f'cr_{regions[channel][0]}_vbf').integrate('dataset', re.compile(dataset_regex['qcdZ'].format(year)))
        h_ewkz = h.integrate('region', f'cr_{regions[channel][0]}_vbf').integrate('dataset', re.compile(dataset_regex['ewkZ'].format(year)))
        h_qcdw = h.integrate('region', f'cr_{regions[channel][1]}_vbf').integrate('dataset', re.compile(dataset_regex['qcdW'].format(year)))
        h_ewkw = h.integrate('region', f'cr_{regions[channel][1]}_vbf').integrate('dataset', re.compile(dataset_regex['ewkW'].format(year)))
        
        fig, ax = plt.subplots()
        hist.plotratio(h_qcdz, h_qcdw, 
            ax=ax, 
            unc='num',
            label='QCD',
            error_opts=data_err_opts
            )

        hist.plotratio(h_ewkz, h_ewkw, 
            ax=ax,
            unc='num',
            label='EWK',
            error_opts=data_err_opts,
            clear=False
            )

        h_qcdz.add(h_ewkz)
        h_qcdw.add(h_ewkw)

        hist.plotratio(h_qcdz, h_qcdw,
            ax=ax,
            unc='num',
            label='QCD + EWK',
            error_opts=data_err_opts,
            clear=False
        )

        ax.set_xlim(0,5000)
        ax.set_ylim(0,0.2)
        ax.set_ylabel(get_ylabel(channel))
        ax.legend()

        loc1 = MultipleLocator(0.02)
        loc2 = MultipleLocator(0.01)
        ax.yaxis.set_major_locator(loc1)
        ax.yaxis.set_minor_locator(loc2)

        ax.yaxis.set_ticks_position('both')

        ax.text(0., 1., year,
            fontsize=14,
            horizontalalignment='left',
            verticalalignment='bottom',
            transform=ax.transAxes
        )

        # Save figure
        outpath = pjoin(outdir, f'z_over_w_{channel}_{year}.pdf')
        fig.savefig(outpath)
        plt.close(fig)
        print(f'File saved: {outpath}')

def compare_z_over_w_with_and_without_nucut(acc, outtag, proc='qcd', channel='muons', distribution='mjj'):
    '''Compare Z/W ratio as a function of mjj with and without the neutrino eta cut.'''
    acc.load(distribution)
    h = preprocess(acc[distribution], acc, distribution)

    outdir = f'./output/{outtag}/from_acc/with_without_nu_cut'
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    for year in [2017, 2018]:
        # Z and W yields with and without the neutrino eta cut
        h_z = h.integrate('region', f'cr_{regions[channel][0]}_vbf').integrate('dataset', re.compile(dataset_regex[f'{proc}Z'].format(year)))
        h_z_cn = h.integrate('region', f'cr_{regions[channel][0]}_vbf_central_nu').integrate('dataset', re.compile(dataset_regex[f'{proc}Z'].format(year)))
        
        h_w = h.integrate('region', f'cr_{regions[channel][1]}_vbf').integrate('dataset', re.compile(dataset_regex[f'{proc}W'].format(year)))
        h_w_cn = h.integrate('region', f'cr_{regions[channel][1]}_vbf_central_nu').integrate('dataset', re.compile(dataset_regex[f'{proc}W'].format(year)))

        # Plot the two ratios
        fig, ax, rax = fig_ratio()
        hist.plotratio(h_z, h_w,
            ax=ax,
            unc='num',
            label=r'$\nu$: $\eta$ Inclusive',
            error_opts=data_err_opts
        )

        hist.plotratio(h_z_cn, h_w_cn,
            ax=ax,
            unc='num',
            error_opts=data_err_opts,
            label=r'$\nu$: $|\eta| < 2.5$',
            clear=False
        )

        ax.text(0., 1., f'{proc.upper()} Z/W',
            fontsize=14,
            horizontalalignment='left',
            verticalalignment='bottom',
            transform=ax.transAxes
        )

        ax.set_xlim(0,5000)
        ax.set_ylim(0,0.2)
        ax.set_ylabel(get_ylabel(channel))
        ax.legend()

        # Plot the ratio of Z/Ws
        sumw_z, sumw2_z = h_z.values(sumw2=True)[()]
        sumw_w, _ = h_w.values(sumw2=True)[()]

        r = sumw_z / sumw_w
        r_err = np.sqrt(sumw2_z) / sumw_w

        r_with_nu_cut = h_z_cn.values()[()] / h_w_cn.values()[()]

        dratio = r / r_with_nu_cut
        dratio_err = r_err / r_with_nu_cut
        centers = h_z.axes()[0].centers()

        rax.errorbar(centers, dratio, yerr=dratio_err, color='k', **data_err_opts)
        
        rax.set_xlabel(r'$M_{jj} \ (GeV)$')
        rax.set_ylabel(r'Inclusive / $\eta$ cut')
        rax.set_ylim(0.8,1.2)
        rax.grid(True)

        outpath = pjoin(outdir, f'{proc}_z_over_w_{channel}_{year}.pdf')
        fig.savefig(outpath)
        plt.close(fig)
        print(f'File saved: {outpath}')

def plot_z_over_w_with_and_without_hfveto(acc, outtag, channel='muons', proc='qcd', distribution='mjj'):
    '''Plot Z/W ratio with and without HF-HF veto applied.'''
    acc.load(distribution)
    h = preprocess(acc[distribution], acc, distribution)

    outdir = f'./output/{outtag}/from_acc/with_without_hf_veto'
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    for year in [2017, 2018]:
        h_z = h.integrate('region', f'cr_{regions[channel][0]}_vbf').integrate('dataset', re.compile(dataset_regex[f'{proc}Z'].format(year)))
        h_z_nohfhf = h.integrate('region', f'cr_{regions[channel][0]}_vbf_nohfveto').integrate('dataset', re.compile(dataset_regex[f'{proc}Z'].format(year)))

        h_w = h.integrate('region', f'cr_{regions[channel][1]}_vbf').integrate('dataset', re.compile(dataset_regex[f'{proc}W'].format(year)))
        h_w_nohfhf = h.integrate('region', f'cr_{regions[channel][1]}_vbf_nohfveto').integrate('dataset', re.compile(dataset_regex[f'{proc}W'].format(year)))

        fig, ax = plt.subplots()
        hist.plotratio(h_z, h_w,
            ax=ax,
            unc='num',
            label='With HF-HF veto',
            error_opts=data_err_opts
        )

        hist.plotratio(h_z_nohfhf, h_w_nohfhf,
            ax=ax,
            unc='num',
            label='Without HF-HF veto',
            clear=False,
            error_opts=data_err_opts
        )

        ax.text(0., 1., f'{proc.upper()} Z/W',
            fontsize=14,
            horizontalalignment='left',
            verticalalignment='bottom',
            transform=ax.transAxes
        )

        ax.set_xlim(200,5000)
        ax.set_ylim(0,0.1)
        ax.set_ylabel(get_ylabel(channel))
        ax.legend()
        ax.grid(True)

        outpath = pjoin(outdir, f'{proc}_z_over_w_{channel}_{distribution}_{year}.pdf')
        fig.savefig(outpath)
        plt.close(fig)
        print(f'File saved: {outpath}')

def main():
    inpath = sys.argv[1]
    acc = dir_archive(inpath)
    acc.load('sumw')
    acc.load('sumw2')

    outtag = re.findall('merged_.*', inpath)[0].replace('/', '')

    for channel in ['electrons', 'muons']:
        plot_z_over_w(acc, outtag, channel=channel)

        try:
            for proc in ['qcd', 'ewk']:
                compare_z_over_w_with_and_without_nucut(acc, outtag, proc=proc, channel=channel)
        except KeyError:
            print(f'Cannot find region with neutrino eta cut in {inpath}, skipping.')

        for proc in ['qcd', 'ewk']:
            try:
                plot_z_over_w_with_and_without_hfveto(acc, outtag, proc=proc, channel=channel)
            except KeyError:
                print(f'Cannot find region without HF veto in {inpath}, skipping.')

if __name__ == '__main__':
    main()