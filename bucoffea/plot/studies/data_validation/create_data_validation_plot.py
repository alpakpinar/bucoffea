#!/usr/bin/env python

import os
import sys
import re
import numpy as np
import mplhep as hep
from coffea import hist
from bucoffea.plot.util import merge_datasets, merge_extensions, scale_xs_lumi
from matplotlib import pyplot as plt
from klepto.archives import dir_archive
from pprint import pprint

pjoin = os.path.join

xlabels = {
    'ak4_eta0' : r'Leading Jet $\eta$',
    'ak4_eta1' : r'Trailing Jet $\eta$',
}

ylabels = {
    'zmumu_over_zee'   : r'$Z(\mu\mu) \ / \ Z(ee)$',
    'wmunu_over_wenu'  : r'$W(\mu\nu) \ / \ W(e\nu)$',
    'zmumu_over_gjets' : r'$Z(\mu\mu) \ / \ \gamma + jets$',
    'zee_over_gjets'   : r'$Z(ee) \ / \ \gamma + jets$',
    'wmunu_over_gjets' : r'$W(\mu\nu) \ / \ \gamma + jets$',
    'wenu_over_gjets'  : r'$W(e\nu) \ / \ \gamma + jets$',
}

binning = {
    # 'ak4_eta0' : hist.Bin('jeteta', r'Leading Jet $\eta$', [-5, -4, -3] + list(np.arange(-2.8,3,0.4)) + [3, 4, 5]),
    # 'ak4_eta1' : hist.Bin('jeteta', r'Trailing Jet $\eta$', [-5, -4, -3] + list(np.arange(-2.8,3,0.4)) + [3, 4, 5]),
    'ak4_eta0' : hist.Bin('jeteta', r'Leading Jet $\eta$', np.arange(-5,6)),
    'ak4_eta1' : hist.Bin('jeteta', r'Trailing Jet $\eta$', np.arange(-5,6)),
}

def create_data_validation_plot(acc, tag, outtag, region1, region2, year, variable='ak4_eta0'):
    '''Create data validation plot, given the input accumulator and two regions being considered.'''
    acc.load(variable)
    h = acc[variable]

    # Preprocess the histogram
    h = merge_extensions(h, acc, reweight_pu=False)
    scale_xs_lumi(h)
    h = merge_datasets(h)

    # Rebin
    if variable in ['ak4_eta0', 'ak4_eta1']:
        h = h.rebin('jeteta', binning[variable])

    # Regular expressions for data in each region
    data = {
        'sr_vbf' : None,
        'cr_1m_vbf' : f'MET_{year}',
        'cr_2m_vbf' : f'MET_{year}',
        'cr_1e_vbf' : f'EGamma_{year}',
        'cr_2e_vbf' : f'EGamma_{year}',
        'cr_g_vbf' : f'EGamma_{year}',
    }

    # Regular expressions for MC in each region
    mc_lo = {
        'sr_vbf' : re.compile(f'(ZJetsToNuNu.*|EW.*|Top_FXFX.*|Diboson.*|QCD_HT.*|.*DYJetsToLL_M-50_HT_MLM.*|.*WJetsToLNu.*HT.*).*{year}'),
        'cr_1m_vbf' : re.compile(f'(EW.*|Top_FXFX.*|Diboson.*|QCD_HT.*|.*DYJetsToLL_M-50_HT_MLM.*|.*WJetsToLNu.*HT.*).*{year}'),
        'cr_1e_vbf' : re.compile(f'(EW.*|Top_FXFX.*|Diboson.*|QCD_HT.*|.*DYJetsToLL_M-50_HT_MLM.*|.*WJetsToLNu.*HT.*).*{year}'),
        'cr_2m_vbf' : re.compile(f'(EW.*|Top_FXFX.*|Diboson.*|QCD_HT.*|.*DYJetsToLL_M-50_HT_MLM.*).*{year}'),
        'cr_2e_vbf' : re.compile(f'(EW.*|Top_FXFX.*|Diboson.*|QCD_HT.*|.*DYJetsToLL_M-50_HT_MLM.*).*{year}'),
        'cr_g_vbf' : re.compile(f'(GJets_(DR-0p4|SM).*|QCD_HT.*|WJetsToLNu.*HT.*).*{year}'),
    }

    data_region1 = h.integrate('region', region1).integrate('dataset', data[region1]).values()[()]
    data_region2 = h.integrate('region', region2).integrate('dataset', data[region2]).values()[()]

    # Calculate the ratio of data in the two regions 
    data_ratio = data_region1 / data_region2
    # Guard against NaN values
    data_ratio[np.isinf(data_ratio) | np.isnan(data_ratio)] = 1.

    data_err_opts = {
        'linestyle':'none',
        'marker': '.',
        'markersize': 10.,
        'color':'k',
    }
    if variable in ['ak4_eta0', 'ak4_eta1']:
        bins = h.axis('jeteta').edges()

    # Plot ratio in data
    fig, (ax, rax) = plt.subplots(2, 1, figsize=(7,7), gridspec_kw={"height_ratios": (3, 1)}, sharex=True)
    hep.histplot(data_ratio, bins, ax=ax, histtype='errorbar', label=f'{ylabels[tag]} Data', **data_err_opts)

    mc_region1 = h.integrate('region', region1).integrate('dataset', mc_lo[region1]).values()[()]
    mc_region2 = h.integrate('region', region2).integrate('dataset', mc_lo[region2]).values()[()]

    # Calculate the ratio of MC in the two regions
    mc_ratio = mc_region1 / mc_region2
    # Guard against NaN values
    mc_ratio[np.isinf(mc_ratio) | np.isnan(mc_ratio)] = 1.

    mc_err_opts = {
        'linestyle':'-',
        'color':'crimson'
    }

    # Plot ratio in MC
    hep.histplot(mc_ratio, bins, ax=ax, histtype='step', label=f'{ylabels[tag]} MC', **mc_err_opts)

    # Aesthetics for the top panel
    ax.set_ylabel(ylabels[tag])
    ax.legend()

    # Calculate the double ratio: Data ratio / MC ratio
    dratio = data_ratio / mc_ratio
    dratio[np.isinf(dratio) | np.isnan(dratio)] = 1.

    # Plot the ratio
    hep.histplot(dratio, bins, ax=rax, histtype='errorbar', **data_err_opts)
    rax.set_xlabel(xlabels[variable])
    rax.set_ylabel('Ratio in Data / Ratio in MC')
    rax.grid(True)
    rax.set_ylim(0.5,1.5)

    xlim = rax.get_xlim()
    rax.plot(xlim, [1., 1.], 'r--')
    rax.set_xlim(xlim)

    # Save figure
    outdir = f'./output/{outtag}'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
        
    outpath = pjoin(outdir, f'{tag}_data_validation_{variable}.pdf')
    fig.savefig(outpath)
    plt.close(fig)
    print(f'MSG% File saved: {outpath}')

def main():
    inpath = sys.argv[1]

    if inpath.endswith('/'):
        outtag = inpath.split('/')[-2]
    else:
        outtag = inpath.split('/')[-1]

    acc = dir_archive(inpath)
    acc.load('sumw')

    create_data_validation_plot(acc, tag='zmumu_over_zee', outtag=outtag, region1='cr_2m_vbf', 
                region2='cr_2e_vbf', year=2017, variable='ak4_eta0'
                )

if __name__ == '__main__':
    main()



