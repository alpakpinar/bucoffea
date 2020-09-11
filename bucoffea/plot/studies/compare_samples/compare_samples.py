#!/usr/bin/env python

import os
import sys
import re
import warnings
import matplotlib.colors as colors
from coffea import hist
from matplotlib import pyplot as plt
from bucoffea.plot.util import merge_datasets, merge_extensions, scale_xs_lumi
from klepto.archives import dir_archive

pjoin = os.path.join

warnings.filterwarnings('ignore')

def get_pretty_dataset_name(dataset_name):
    year = re.findall(r'201\d', dataset_name)[0]
    pretty_names = {
        f'EWKZ.*' : r'EWK $Z(\nu\nu) \ {}$'.format(year),
        f'ZJetsToNuNu.*' : r'QCD $Z(\nu\nu) \ {}$'.format(year),
        f'GluGlu.*' : r'$ggH(inv) \ {}$'.format(year),
        f'VBF.*' : r'VBF $H(inv) \ {}$'.format(year)
    }

    pretty_name = None
    for regex, label in pretty_names.items():
        if re.match(regex, dataset_name):
            pretty_name = label.format(year)            

    return pretty_name

def compare_samples(acc, outtag, variable, year):
    '''Compare QCD/EWK Z(vv), VBF H(inv) and ggH(inv) distributions.'''
    acc.load(variable)
    h = acc[variable]

    h = merge_extensions(h, acc, reweight_pu=False)
    scale_xs_lumi(h)
    h = merge_datasets(h)

    if variable == 'mjj':
        mjj_ax = hist.Bin('mjj', r'$M_{jj}$ (GeV)', list(range(200,800,300)) + list(range(800,2000,400)) + [2000, 2750, 3500])
        h = h.rebin('mjj', mjj_ax)
    elif variable == 'ak4_pt':
        pt_ax = hist.Bin('jetpt', r'All AK4 jet $p_{T}$ (GeV)',list(range(0,1000,20)) )
        h = h.rebin('jetpt', pt_ax)

    # Get signal region
    h = h.integrate('region', 'sr_vbf')[re.compile(f'.*{year}')]

    fig, ax = plt.subplots()
    hist.plot1d(h, ax=ax, overlay='dataset', overflow='over', density=True)

    ax.set_yscale('log')
    ax.set_ylim(1e-5, 1e2)
    ax.set_ylabel('Counts (Normalized)')

    # Modify legend with short dataset names
    dataset_names = [
        f'EWKZ2Jets_ZToNuNu-mg_{year}',
        f'GluGlu_HToInvisible_M125_HiggspTgt190_pow_pythia8_{year}',
        f'VBF_HToInvisible_M125_pow_pythia8_{year}',
        f'ZJetsToNuNu_HT_{year}'
    ]

    new_labels = []
    for dataset_name in dataset_names:
        pretty_name = get_pretty_dataset_name(dataset_name)
        new_labels.append(pretty_name)

    ax.legend(title='Datasets', labels=new_labels)

    # Save figure
    outdir = f'./output/{outtag}'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    outpath = pjoin(outdir, f'comparison_{variable}_{year}.pdf')
    fig.savefig(outpath)
    print(f'File saved: {outpath}')

def plot_2d(acc, outtag, dataset_regex):
    '''Plot 2D jet eta/phi histogram for given sample.'''
    variable = 'ak4_pt_eta'
    acc.load(variable)
    h = acc[variable]

    h = merge_extensions(h, acc, reweight_pu=False)
    scale_xs_lumi(h)
    h = merge_datasets(h)

    # Rebin jet pt axis
    pt_ax = hist.Bin('jetpt', r'All AK4 jet $p_{T}$ (GeV)',list(range(0,1000,50)) )
    h = h.rebin('jetpt', pt_ax)

    # Get signal region and relevant dataset
    h = h.integrate('region', 'sr_vbf').integrate('dataset', re.compile(dataset_regex))

    fig, ax = plt.subplots()
    patch_opts = {'norm' : colors.LogNorm(vmin=1e-1, vmax=h.values()[()].max())}
    hist.plot2d(h, ax=ax, xaxis='jeteta', patch_opts=patch_opts)

    year = re.findall(r'201\d', dataset_regex)[0]

    regex_to_title = {
        f'ZJets.*{year}' : r'QCD $Z(\nu\nu)$ {}',
        f'EWKZ.*{year}'  : r'EWK $Z(\nu\nu)$ {}',
        f'VBF.*{year}'   : r'VBF $H(inv)$ {}',
        f'Glu.*{year}'   : r'$ggH(inv)$ {}'
    }

    ax.set_title( regex_to_title[dataset_regex].format(year) )

    # Save figure
    outdir = f'./output/{outtag}'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    dataset_tag = dataset_regex.replace('.*', '_')
    outpath = pjoin(outdir, f'2d_{dataset_tag}.pdf')
    fig.savefig(outpath)
    print(f'File saved: {outpath}')

def main():
    inpath = sys.argv[1]
    acc = dir_archive(inpath)
    acc.load('sumw')
    acc.load('sumw2')

    outtag = re.findall('merged_.*', inpath)[0].replace('/', '')

    variables = ['ak4_pt', 'ak4_eta']

    for year in [2017, 2018]:
        for variable in variables:
            compare_samples(acc, outtag, variable=variable, year=year)

    # 2D jet eta/phi plots
    regex_list = ['ZJets.*2017', 'ZJets.*2018', 'EWKZ.*2017', 'EWKZ.*2018', 'VBF.*2017', 'VBF.*2018', 'Glu.*2017', 'Glu.*2018']
    for dataset_regex in regex_list:
        plot_2d(acc, outtag, dataset_regex=dataset_regex)

if __name__ == '__main__':
    main()
