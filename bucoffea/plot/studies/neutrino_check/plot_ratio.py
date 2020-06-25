#!/usr/bin/env python

import os
import sys
import re
import numpy as np
import matplotlib.ticker
from matplotlib import pyplot as plt
from bucoffea.plot.util import merge_datasets, scale_xs_lumi, merge_extensions
from klepto.archives import dir_archive
from coffea import hist

pjoin = os.path.join

tag_to_dataset = {
	'znunu17' : {
		'region' : {'with_nu_cut' : 'sr_vbf_nu', 'without_nu_cut' : 'sr_vbf'},
		'dataset' : re.compile('ZJetsToNuNu.*2017')
	},
	'znunu18' : {
		'region' : {'with_nu_cut' : 'sr_vbf_nu', 'without_nu_cut' : 'sr_vbf'},
		'dataset' : re.compile('ZJetsToNuNu.*2018')
	},
	'wlnu17' : {
		'region' : {'with_nu_cut' : 'sr_vbf', 'without_nu_cut' : 'sr_vbf'},
		'dataset' : re.compile('WJetsToLNu.*2017')
	},
	'wlnu18' : {
		'region' : {'with_nu_cut' : 'sr_vbf', 'without_nu_cut' : 'sr_vbf'},
		'dataset' : re.compile('WJetsToLNu.*2018')
	},
	'zmumu17' : {
		'region' : {'with_nu_cut' : 'cr_2m_vbf_nu', 'without_nu_cut' : 'cr_2m_vbf'},
		'dataset' : re.compile('DYJetsToLL.*2017')
	},
	'zmumu18' : {
		'region' : {'with_nu_cut' : 'cr_2m_vbf_nu', 'without_nu_cut' : 'cr_2m_vbf'},
		'dataset' : re.compile('DYJetsToLL.*2018')
	},
	'zee17' : {
		'region' : {'with_nu_cut' : 'cr_2e_vbf_nu', 'without_nu_cut' : 'cr_2e_vbf'},
		'dataset' : re.compile('DYJetsToLL.*2017')
	},
	'zee18' : {
		'region' : {'with_nu_cut' : 'cr_2e_vbf_nu', 'without_nu_cut' : 'cr_2e_vbf'},
		'dataset' : re.compile('DYJetsToLL.*2018')
	},
	'wmunu17' : {
		'region' : {'with_nu_cut' : 'cr_1m_vbf_nu', 'without_nu_cut' : 'cr_1m_vbf'},
		'dataset' : re.compile('WJetsToLNu.*2017')
	},
	'wmunu18' : {
		'region' : {'with_nu_cut' : 'cr_1m_vbf_nu', 'without_nu_cut' : 'cr_1m_vbf'},
		'dataset' : re.compile('WJetsToLNu.*2018')
	},
	'wenu17' : {
		'region' : {'with_nu_cut' : 'cr_1e_vbf_nu', 'without_nu_cut' : 'cr_1e_vbf'},
		'dataset' : re.compile('WJetsToLNu.*2017')
	},
	'wenu18' : {
		'region' : {'with_nu_cut' : 'cr_1e_vbf_nu', 'without_nu_cut' : 'cr_1e_vbf'},
		'dataset' : re.compile('WJetsToLNu.*2018')
	}
}

data_pairs = {
	'znunu_over_wlnu17' : ('znunu17', 'wlnu17'),
	'znunu_over_wlnu18' : ('znunu18', 'wlnu18'),
	'zmumu_over_wmunu17' : ('zmumu17', 'wmunu17'),
	'zmumu_over_wmunu18' : ('zmumu18', 'wmunu18'),
	'zee_over_wenu17' : ('zee17', 'wenu17'),
	'zee_over_wenu18' : ('zee18', 'wenu18'),
}

ylabels = {
	'znunu_over_wlnu17' : r'$Z\rightarrow \nu \nu$ / $W\rightarrow \ell \nu$',
	'znunu_over_wlnu18' : r'$Z\rightarrow \nu \nu$ / $W\rightarrow \ell \nu$',
	'zmumu_over_wmunu17' : r'$Z\rightarrow \mu \mu$ / $W\rightarrow \mu \nu$',
	'zmumu_over_wmunu18' : r'$Z\rightarrow \mu \mu$ / $W\rightarrow \mu \nu$',
	'zee_over_wenu17' : r'$Z\rightarrow ee$ / $W\rightarrow e \nu$',
	'zee_over_wenu18' : r'$Z\rightarrow ee$ / $W\rightarrow e \nu$',
}

def plot_ratio(acc, tag, out_tag, neutrino_cut='with_nu_cut'):
	'''Plot the ratio of two distributions as a function of mjj.
	   If neutrino_cut is set to True, looks at data where neutrino
	   eta is restricted to be lower than 2.5.'''
	acc.load('mjj')
	h = acc['mjj']

	h = merge_extensions(h, acc, reweight_pu=False)
	scale_xs_lumi(h)
	h = merge_datasets(h)

	# Rebin mjj
	mjjbin = hist.Bin('mjj', r'$M_{jj}\ (GeV)$', list(range(200,800,300)) + list(range(800,2000,400)) + [2000, 3000])
	h = h.rebin('mjj', mjjbin)

	# Pick data for numerator and denominator
	data_pair = data_pairs[tag]
	num_info = tag_to_dataset[data_pair[0]]
	denom_info = tag_to_dataset[data_pair[1]]
	
	num = h.integrate('region', num_info['region'][neutrino_cut]).integrate('dataset', num_info['dataset'])
	denom = h.integrate('region', denom_info['region'][neutrino_cut]).integrate('dataset', denom_info['dataset'])

	def ratio(num, denom):
		'''Calculate ratio and error on ratio.'''
		num_sumw, num_sumw2 = num.values(overflow='over', sumw2=True)[()]
		denom_sumw, denom_sumw2 = denom.values(overflow='over', sumw2=True)[()]
		
		rsumw = num_sumw / denom_sumw
		rsumw_err = np.hypot(
			np.sqrt(num_sumw2) / denom_sumw,
			num_sumw * np.sqrt(denom_sumw2)/ denom_sumw ** 2
		)

		return rsumw, rsumw_err

	rsumw, rsumw_err = ratio(num, denom)

	xcenters = num.axis('mjj').centers(overflow='over')

	# Plot the ratio
	fig, ax = plt.subplots(1,1)
	ax.errorbar(x=xcenters, y=rsumw, yerr=rsumw_err, marker='o', ls='')
	ax.set_xlabel(r'$M_{jj}\ (GeV)$')
	ax.set_ylabel(ylabels[tag])
	ax.grid(True)

	if '17' in tag:
		year = 2017
	elif '18' in tag:
		year = 2018
	
	ax.text(1, 1,
		year,
		fontsize=14,
		horizontalalignment='right',
		verticalalignment='bottom',
		transform=ax.transAxes
	)
	outdir = f'output/{out_tag}'
	if not os.path.exists(outdir):
		os.makedirs(outdir)
	
	outfile = pjoin(outdir, f'{tag}_ratio_{neutrino_cut}.pdf')
	fig.savefig(outfile)
	print(f'File saved: {outfile}')

	return xcenters, rsumw, rsumw_err

def plot_ratio_comparison(r, tag, out_tag):
	'''Plot two given ratios on the same pad.'''
	xcenters1, r1_sumw, r1_sumw_err = r['with_nu_cut']
	xcenters2, r2_sumw, r2_sumw_err = r['without_nu_cut']

	assert (xcenters1 == xcenters2).all()

	# Get ratio of ratios
	rr = r1_sumw / r2_sumw

	fig, (ax, rax) = plt.subplots(2, 1, figsize=(7,7), gridspec_kw={"height_ratios": (3, 1)}, sharex=True)

	ax.errorbar(x=xcenters1, y=r1_sumw, yerr=r1_sumw_err, label=r'With $\nu$ $\eta$ cut', marker='o', ls='')
	ax.errorbar(x=xcenters2, y=r2_sumw, yerr=r2_sumw_err, label=r'Without $\nu$ $\eta$ cut', marker='o', ls='')
	ax.grid(True)
	ax.set_ylabel(ylabels[tag])


	if '17' in tag:
		year = 2017
	elif '18' in tag:
		year = 2018
	
	ax.text(1, 1,
		year,
		fontsize=14,
		horizontalalignment='right',
		verticalalignment='bottom',
		transform=ax.transAxes
	)
	ax.legend()

	ratio_opts = {
		'marker' : '.',
		'linestyle' : 'none',
		'markersize' : 10,
		'color' : 'k'
	}

	rax.plot(xcenters1, rr, **ratio_opts)
	rax.grid(True)
	rax.set_xlabel(r'$M_{jj}\ (GeV)$')
	rax.set_ylabel(r'With / Without $\nu$ Cut')
	rax.fill_between(xcenters1, 1+(r2_sumw_err/r2_sumw), 1-(r2_sumw_err/r2_sumw), color='gray')

	rax.set_xlim(200, 4000)
	rax.set_ylim(0.7, 1.3)
	rax.plot([200, 4000], [1, 1], 'r--')
	
	loc = matplotlib.ticker.MultipleLocator(base=0.1)
	rax.yaxis.set_major_locator(loc)

	outdir = f'output/{out_tag}/comparison'
	if not os.path.exists(outdir):
		os.makedirs(outdir)
	
	outfile = pjoin(outdir, f'{tag}_ratio.pdf')
	fig.savefig(outfile)

	print(f'File saved: {outfile}')

def main():
	inpath = sys.argv[1]

	if inpath.endswith('/'):
		out_tag = inpath.split('/')[-2]
	else:
		out_tag = inpath.split('/')[-1]

	acc = dir_archive(
						inpath,
						memsize=1e3,
						compression=0,
						serialized=True
					)

	acc.load('sumw')
	acc.load('sumw2')

	tags = [
		'znunu_over_wlnu17',
		'znunu_over_wlnu18',
	]

	# Store ratios and ratio errors
	ratio_dict = {}
	for tag in tags:
		ratio_dict[tag] = {}
		ratio_dict[tag]['with_nu_cut'] = plot_ratio(acc, tag=tag, out_tag=out_tag, neutrino_cut='with_nu_cut')
		ratio_dict[tag]['without_nu_cut'] = plot_ratio(acc, tag=tag, out_tag=out_tag, neutrino_cut='without_nu_cut')

		plot_ratio_comparison(ratio_dict[tag], tag=tag, out_tag=out_tag)



if __name__ == '__main__':
	main()


