#!/usr/bin/env python

import os
import sys
import re
from matplotlib import pyplot as plt
from klepto.archives import dir_archive
from bucoffea.plot.util import scale_xs_lumi, merge_extensions, merge_datasets
from coffea import hist
from pprint import pprint

pjoin = os.path.join

# For rebinning
vpt_ax = hist.Bin('vpt', r'$p_T(V) \ (GeV)$', [200, 240, 280, 320, 360, 400, 480, 560, 680, 800, 1000])
mjj_ax = hist.Bin('mjj', r'$M_{jj}$ (GeV)', [200, 400, 600, 900, 1200, 1500, 2000, 2750, 3500, 5000])

def plot_theory_uncs(acc, outtag, dist='gen_v_pt'):
    '''Plot theory uncertainties as a function of given variable (defualt is GEN boson pt)'''
    # Load histograms into memory, scale and pre-process
    for d in [f'{dist}', f'{dist}_unc']:
        acc.load(d)
        
        # Merging + rescaling
        acc[d] = merge_extensions(acc[d], acc, reweight_pu=False)
        scale_xs_lumi(acc[d])
        acc[d] = merge_datasets(acc[d])
        AX_TO_USE = None
        # Rebin 
        if dist == 'gen_v_pt':
            acc[d] = acc[d].rebin('vpt', vpt_ax)
            AX_TO_USE = vpt_ax
        elif dist == 'mjj':
            acc[d] = acc[d].rebin('mjj', mjj_ax)
            AX_TO_USE = mjj_ax

    for year in [2017, 2018]:
        # Get nominal distributions and plot them
        nominal_plots_outdir = f'./output/{outtag}/nominal_plots'
        if not os.path.exists(nominal_plots_outdir):
            os.makedirs(nominal_plots_outdir)

        tag_to_title = {
            f'znunu_{year}' : r'$Z(\nu \nu)$ ' + str(year),
            f'wlnu_{year}'  : r'$W(\ell \nu)$ ' + str(year),
            f'gjets_{year}' : r'$\gamma$ + jets ' + str(year)
        }

        def plot_hist(h, tag):
            fig, ax = plt.subplots()
            hist.plot1d(h, ax=ax)
            ax.set_title(tag_to_title[tag])
            fig.savefig(pjoin(nominal_plots_outdir, f'{tag}_nominal_{dist}.pdf'))
            plt.close()

        h_z_nom = acc[dist][re.compile(f'ZJetsToNuNu.*HT.*{year}')].integrate('region', 'sr_vbf').integrate('dataset')
        plot_hist(h_z_nom, tag=f'znunu_{year}')
        h_w_nom = acc[dist][re.compile(f'WJetsToLNu.*HT.*{year}')].integrate('region', 'sr_vbf').integrate('dataset')
        plot_hist(h_w_nom, tag=f'wlnu_{year}')
        h_ph_nom = acc[dist][re.compile(f'GJets_DR-0p4.*HT.*{year}')].integrate('region', 'cr_g_vbf').integrate('dataset')
        plot_hist(h_ph_nom, tag=f'gjets_{year}')
            
        # Scale + PDF variations for QCD Z 
        ratios_outdir = f'./output/{outtag}/var_over_nom'
        if not os.path.exists(ratios_outdir):
            os.makedirs(ratios_outdir)

        h_z_unc = acc[f'{dist}_unc'][re.compile(f'ZJetsToNuNu.*HT.*{year}')].integrate('region', 'sr_vbf').integrate('dataset')
        z_ratios = {}    
        for unc in map(str, h_z_unc.axis('uncertainty').identifiers()):
            if 'goverz' in unc or 'ewkcorr' in unc:
                continue
            h = h_z_unc.integrate(h_z_unc.axis('uncertainty'), unc)
            # Store ratios of varied and nominal Z for all uncertainties
            z_ratios[unc] = h.values()[()] / h_z_nom.values()[()]

        # Scale + PDF variations for QCD photons
        h_ph_unc = acc[f'{dist}_unc'][re.compile(f'GJets_DR-0p4.*HT.*{year}')].integrate('region', 'cr_g_vbf').integrate('dataset')
        ph_ratios = {}    
        for unc in map(str, h_ph_unc.axis('uncertainty').identifiers()):
            if 'zoverw' in unc or 'ewkcorr' in unc:
                continue
            h = h_ph_unc.integrate(h_ph_unc.axis('uncertainty'), unc)
            # Store ratios of varied and nominal photons for all uncertainties
            # Multiply by 2 due to bug in NanoAOD
            ph_ratios[unc] = h.values()[()] / h_ph_nom.values()[()]
            # if dist == 'gen_v_pt':
                # ph_ratios[unc] *= 2

        # Plotter code for var/nom ratios
        def plot_var_over_nom(ratios, tag):
            fig, ax = plt.subplots()
            for unc, ratio_arr in ratios.items():
                ax.plot(AX_TO_USE.centers(), ratio_arr, label='_'.join(unc.split('_')[-2:]), marker='o')
            ax.legend(ncol=2)
            ax.set_xlabel(AX_TO_USE.label)
            tag_to_ylabel = {
                f'znunu_{year}' : r'$Z(\nu \nu)_{var} \ / \ Z(\nu \nu)_{nom}$ ' + str(year),
                f'wlnu_{year}'  : r'$W(\ell \nu)_{var} \ / \ W(\ell \nu)_{nom}$ ' + str(year),
                f'gjets_{year}' : r'$(\gamma + jets)_{var} \ / \ (\gamma + jets)_{nom}$ ' + str(year)
            }
            ax.set_ylabel(tag_to_ylabel[tag])
            ax.set_ylim(0.8,1.2)
            ax.grid(True)
            
            xlim = ax.get_xlim()
            ax.plot(xlim, [1, 1], 'k--')
            ax.set_xlim(xlim)
        
            outpath = pjoin(ratios_outdir, f'{tag}_varovernom_{dist}.pdf')
            fig.savefig(outpath)

        plot_var_over_nom(ratios=z_ratios, tag=f'znunu_{year}')
        plot_var_over_nom(ratios=ph_ratios, tag=f'gjets_{year}')

        # Variations in Z/W ratio
        z_over_w_double_ratios = {}
        nominal_z_over_w = h_z_nom.values()[()] / h_w_nom.values()[()]
        for unc in map(str, h_z_unc.axis('uncertainty').identifiers()):
            if 'goverz' in unc or 'ewkcorr' in unc:
                continue
            h = h_z_unc.integrate(h_z_unc.axis('uncertainty'), unc)
            varied_z_over_w = h.values()[()] / h_w_nom.values()[()]
            z_over_w_double_ratios[unc] = varied_z_over_w / nominal_z_over_w

        # Variations in photons/Z ratio
        ph_over_z_double_ratios = {}
        nominal_ph_over_z = h_ph_nom.values()[()] / h_z_nom.values()[()]
        for unc in map(str, h_ph_unc.axis('uncertainty').identifiers()):
            if 'zoverw' in unc or 'ewkcorr' in unc:
                continue
            h = h_ph_unc.integrate(h_ph_unc.axis('uncertainty'), unc)
            varied_ph_over_z = h.values()[()] / h_z_nom.values()[()]
            ph_over_z_double_ratios[unc] = varied_ph_over_z / nominal_ph_over_z

        # Plotter code for var/nom double ratios
        def plot_double_ratio(ratios, tag):
            fig, ax = plt.subplots()
            for unc, ratio_arr in ratios.items():
                ax.plot(AX_TO_USE.centers(), ratio_arr, label='_'.join(unc.split('_')[-2:]), marker='o')
            ax.legend(ncol=2)
            ax.set_xlabel(AX_TO_USE.label)
            tag_to_ylabel = {
                f'zoverw_{year}' : r'$Z(\nu \nu)\ / \ W(\ell \nu)$ ' + str(year),
                f'goverz_{year}' : r'$(\gamma + jets) \ / \ Z(\nu \nu)$ ' + str(year)
            }
            ax.set_ylabel(tag_to_ylabel[tag])
            ax.set_ylim(0.8,1.2)
            ax.grid(True)
            
            xlim = ax.get_xlim()
            ax.plot(xlim, [1, 1], 'k--')
            ax.set_xlim(xlim)
        
            outpath = pjoin(ratios_outdir, f'{tag}_varovernom_{dist}.pdf')
            fig.savefig(outpath)

        plot_double_ratio(ratios=z_over_w_double_ratios, tag=f'zoverw_{year}')
        plot_double_ratio(ratios=ph_over_z_double_ratios, tag=f'goverz_{year}')

def main():
    inpath = sys.argv[1]

    acc = dir_archive(
        inpath,
        serialized=True,
        memsize=1e3,
        compression=0
    )

    acc.load('sumw')
    acc.load('sumw2')

    if inpath.endswith('/'):
        outtag = inpath.split('/')[-2]
    else:
        outtag = inpath.split('/')[-1]

    plot_theory_uncs(acc, outtag, dist='gen_v_pt')
    plot_theory_uncs(acc, outtag, dist='mjj')

if __name__ == '__main__':
    main()
