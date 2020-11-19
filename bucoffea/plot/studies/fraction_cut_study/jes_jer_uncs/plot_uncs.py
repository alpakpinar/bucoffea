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

pjoin = os.path.join

def rebin(h, variable):
    new_axes = {
        'pt_ax' : hist.Bin('jetpt', r'Jet $p_T \ (GeV)$', [40,80,120,160,200,300])
    }

    # Rebin the jet pt axis
    h = h.rebin('jetpt', new_axes['pt_ax'])

    return h

def get_legend_label(variation):
    if variation == '':
        return 'Nominal'
    return variation.replace('_', '')

def plot_uncs(acc, outtag, variable='ak4_pt0_eta0', etaslice='pos'):
    '''As a function of the given variable, plot the JES/JER uncertainties.'''
    acc.load(variable)
    h = acc[variable]

    h = merge_extensions(h, acc, reweight_pu=False)
    scale_xs_lumi(h)
    h = merge_datasets(h)

    h = rebin(h, variable)
    
    # Integrate over the endcap: Positive or negative side, as specified in the function argument
    if etaslice == 'pos':
        h = h.integrate('jeteta', slice(2.5,3.0))
        fig_title = r'Jet: $2.5 < \eta < 3.0$'
    else:
        h = h.integrate('jeteta', slice(-3.0,-2.5))
        fig_title = r'Jet: $-3.0 < \eta < -2.5$'

    # List of JES/JER uncertainties
    variations = [
        '',
        '_jerUp',
        '_jerDown',
        '_jesTotalUp',
        '_jesTotalDown'
    ]

    for year in [2017, 2018]:
        # Get the DY MC
        _h = h.integrate('dataset', re.compile(f'DYJetsToLL.*{year}'))
        # For each variation, calculate the MC efficiency
        fig, ax, rax = fig_ratio()
        
        eff_dict = {}
        centers = _h.axis('jetpt').centers(overflow='over')

        for idx, var in enumerate(variations):
            hnum = _h.integrate('region', f'cr_2m_withEmEF{var}')
            hden = _h.integrate('region', f'cr_2m_noEmEF{var}')

            sumw_num = hnum.values(overflow='over')[()]
            sumw_den = hden.values(overflow='over')[()]

            eff = sumw_num / sumw_den
            err = np.abs(hist.clopper_pearson_interval(sumw_num, sumw_den) - eff)
            ax.errorbar(centers, eff, yerr=err, label=get_legend_label(var), marker='o', color=f'C{idx}')

            # Store the efficiencies and errors in a dictionary
            eff_dict[var] = {
                'eff' : eff,
                'err' : err,
            }

        ax.legend()
        ax.set_ylabel('Efficiency in MC')
        ax.set_xlabel(r'Jet $p_T \ (GeV)$')
        ax.set_ylim(0.90,0.98)
        ax.set_title(fig_title)
        ax.set_xlabel('')

        ax.text(1, 1, year,
            fontsize=12,
            horizontalalignment='right',
            verticalalignment='bottom',
            transform=ax.transAxes
            )

        # Plot the ratio of efficiencies on the ratio pad
        nominal_eff = eff_dict['']['eff']
        for idx, (variation, vals) in enumerate(eff_dict.items()):
            if variation == '':
                continue
            eff, err = vals.values()

            reff = eff / nominal_eff
            rerr = err / nominal_eff

            rax.errorbar(centers, reff, yerr=rerr, marker='o', ls='', color=f'C{idx}')

        rax.grid(True)
        if year == 2017:
            rax.set_ylim(0.96,1.04)
            loc = MultipleLocator(0.02)
        else:
            rax.set_ylim(0.98,1.02)
            loc = MultipleLocator(0.01)
        
        rax.set_xlabel(r'Jet $p_T \ (GeV)$')
        rax.set_ylabel('Ratio to nominal')
        rax.yaxis.set_major_locator(loc)

        # Save figure
        outdir = f'./output/{outtag}'
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        outpath = pjoin(outdir, f'jes_jer_uncs_{etaslice}_endcap_{year}.pdf')
        fig.savefig(outpath)
        plt.close(fig)

        print(f'File saved: {outpath}')

def main():
    inpath = sys.argv[1]
    acc = dir_archive(inpath)
    acc.load('sumw')
    acc.load('sumw2')

    outtag = re.findall('merged_.*', inpath)[0].replace('/', '')

    plot_uncs(acc, outtag, etaslice='pos')
    plot_uncs(acc, outtag, etaslice='neg')

if __name__ == '__main__':
    main()