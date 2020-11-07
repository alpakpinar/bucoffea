#!/usr/bin/env python

import os
import sys
import re
from coffea import hist
from bucoffea.plot.util import merge_datasets, merge_extensions, scale_xs_lumi, fig_ratio
from bucoffea.helpers.paths import bucoffea_path
from matplotlib import pyplot as plt
from klepto.archives import dir_archive
from pprint import pprint

pjoin = os.path.join

def get_legend_label(key):
    mapping = {
        'jet_id_all' : 'Jet cut applied everywhere',
        'jet_id_on_endcap' : 'Jet cut only on endcaps'
    }

    return mapping[key]

def get_dataset_regex(dataset_tag, year):
    mapping = {
        'vbf' : f'VBF_HToInvisible.*M125.*{year}',
        'qcd_zvv' : f'ZJetsToNuNu.*{year}',
        'ewk_zvv' : f'EWKZ2Jets_ZToNuNu.*{year}',
    }
    
    return mapping[dataset_tag]

def preprocess(h, acc, dataset_regex, year, variable):
    h = merge_extensions(h, acc, reweight_pu=False)
    scale_xs_lumi(h)
    h = merge_datasets(h)

    # Rebin mjj
    if variable == 'mjj':
        mjj_ax = hist.Bin('mjj', r'$M_{jj}$ (GeV)', [200., 400., 600., 900., 1200., 1500.,2000., 2750., 3500., 5000.])
        h = h.rebin('mjj', mjj_ax)
    
    # Get the relevant dataset
    h = h.integrate('region', 'sr_vbf').integrate('dataset', re.compile(dataset_regex))
    
    return h

def compare_shapes(acc_dict, outtag, year, dataset_tag, variable='mjj'):
    '''
    Compare the shapes for the given dataset as a function of the given variable,
    for the case where the jet ID cut is applied only on endcaps, or it is applied everywhere. 
    '''
    dataset_regex = get_dataset_regex(dataset_tag, year)
    
    h_dict = {}
    for key, acc in acc_dict.items():
        acc.load(variable)
        h_dict[key] = preprocess(acc[variable], acc, dataset_regex, year, variable)
    
    # Now, compare the two histograms!
    fig, ax, rax = fig_ratio()
    legend_labels = []
    for key, h in h_dict.items():
        hist.plot1d(h, ax=ax, clear=False)
        legend_labels.append( get_legend_label(key) )

    ax.legend(labels=legend_labels)
    ax.set_yscale('log')
    ax.set_ylim(1e-3,1e5)
    ax.set_xlabel('')

    data_err_opts = {
        'linestyle':'none',
        'marker': '.',
        'markersize': 10.,
        'color':'k',
    }

    # Plot ratio of the two
    hist.plotratio(
        h_dict['jet_id_all'],
        h_dict['jet_id_on_endcap'],
        ax=rax,
        unc='num',
        error_opts=data_err_opts
    )

    rax.set_xlabel(r'$M_{jj} \ (GeV)$')
    rax.set_ylabel('All / Endcap')
    rax.grid(True)
    rax.set_ylim(0.8,1.2)

    # Save figure
    outdir = f'./output/{outtag}'
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    outpath = pjoin(outdir, f'jet_id_comp_{dataset_tag}_{year}_{variable}.pdf')
    fig.savefig(outpath)
    plt.close(fig)

    print(f'File saved: {outpath}')

def main():
    # Paths containing merged coffea files for the two cases
    inpath_jet_id_all = bucoffea_path('submission/merged_2020-11-06_vbfhinv_03Sep20v7_jetid_all') 
    inpath_jet_id_on_endcap = bucoffea_path('submission/merged_2020-11-06_vbfhinv_03Sep20v7_jetid_on_endcap')

    acc_dict = {
        'jet_id_all' : dir_archive(inpath_jet_id_all),
        'jet_id_on_endcap' : dir_archive(inpath_jet_id_on_endcap)
    }

    for acc in acc_dict.values():
        acc.load('sumw')
        acc.load('sumw2')
    
    # Tag for tracking the inputs 
    outtag = '06Nov20'
    
    compare_shapes(
        acc_dict,
        outtag=outtag,
        year=2017,
        dataset_tag='vbf',
        variable='mjj'
    )

if __name__ == '__main__':
    main()