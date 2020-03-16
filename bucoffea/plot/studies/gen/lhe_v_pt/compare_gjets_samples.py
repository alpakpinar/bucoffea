#!/usr/bin/env python

import os
import sys
import re
import numpy as np
from bucoffea.plot.util import (merge_datasets,
								merge_extensions,
								scale_xs_lumi)

from klepto.archives import dir_archive
from matplotlib import pyplot as plt
from pprint import pprint
from coffea import hist

pjoin = os.path.join

def compare_two_gjets_samples(acc, samples, outtag, distribution='vpt', inclusive=True):
	'''
	Compare the specified distribution of several LO GJets samples.
	List of samples is specified in "samples" argument.
	The distribution to be compared is specified in "distribution" argument.

	If inclusive option is set to True, look at distributions
	without VBF cuts applied, otherwise look at distributions
	after VBF cuts are applied.
	'''
	# Extract dataset years
	extract_year = lambda name: name.split('_')[-1]
	years = map(extract_year, samples)

	# Get the distribution
	# NOTE: For the VBF masked case, both mjj and vpt are stored in "gen_vpt_vbf_stat1" histogram
	dist = f'gen_{distribution}_inclusive_stat1' if inclusive else 'gen_vpt_vbf_stat1'
	
	acc.load(dist)
	h = acc[dist]

	rebin = {
		'vpt' : hist.Bin('vpt', r'$p_T(V)\ (GeV)$', np.arange(200,1500,50)),
		'mjj' : hist.Bin('mjj', r'$M_{jj}\ (GeV)$', list(range(200,800,300)) + list(range(800,2000,400)) + [2000, 2750, 3500])
	}

	# Rebin
	distbin = rebin[distribution]
	h = h.rebin(distribution, distbin)

	edges = h.axis(distribution).edges(overflow='over')
	centers = h.axis(distribution).centers(overflow='over')

	# Merging and scaling
	h = merge_extensions(h, acc, reweight_pu=False)
	scale_xs_lumi(h)
	h = merge_datasets(h)
	# Integrate over jet pt and mjj or vpt axes if looking at
	# distributions after VBF cuts are applied
	if not inclusive:
		integrate_over = 'mjj' if distribution == 'vpt' else 'vpt'
		h = h.integrate('jpt').integrate(integrate_over)

	# Mapping from sample names to regular expressions
	# for the relevant dataset names
	# TODO: Generalize to multiple years
	sample_to_regex = {s : s.replace(f'{year}',f'.*_{year}') for year, s in zip(years, samples)}

	# Store histograms from each dataset in a dictionary
	histos = {s : h[re.compile(sample_to_regex[s])].integrate('dataset') for s in samples}

	# Plot the comparison
	rax = None
	if len(samples) == 2:
		fig, (ax, rax) = plt.subplots(2, 1, figsize=(7,7), gridspec_kw={"height_ratios": (3, 1)}, sharex=True)

	else:
		fig, ax = plt.subplots(1,1)

	for dataset, histo in histos.items():
		ax.step(edges[:-1], histo.values(overflow='over')[()], where='post', label=dataset)
	
	ax.legend()
	ax.set_yscale('log')

	# y-axis limit for each parameter
	ylims = {
		'vpt' : (1e-2,1e8),
		'mjj' : (1e0, 1e10)
	}

	ax.set_ylim( ylims[distribution] )
	# Do not include overflow bin
	ax.set_xlim(edges[0], edges[-2])
	ax.set_ylabel('Counts')

	title = 'Inclusive' if inclusive else 'After VBF Selection'
	ax.set_title(title)

	# If two samples are compared, plot the ratio pad
	if rax:
		# Compute the ratios and uncertainties on them
		sumw_1, sumw2_1 = histos[samples[0]].values(overflow='over', sumw2=True)[()]
		sumw_2, sumw2_2 = histos[samples[1]].values(overflow='over', sumw2=True)[()]

		ratio = sumw_1 / sumw_2
		unc = np.hypot(
			np.sqrt(sumw2_1) / sumw_1,
			np.sqrt(sumw2_2) / sumw_2,
		)

		rax.errorbar(x=centers, y=ratio, yerr=unc, ls='', marker='o', color='k')
		rax.grid(True)
		rax.set_ylim(0.6,1.4)

		xlim = ax.get_xlim()
		rax.plot(xlim, [1., 1.], 'r--')
		rax.set_xlim(xlim)
		rax.set_xlabel(distbin.label)

	else:
		ax.set_xlabel(distbin.label)

	# Save figure
	outdir = f'./output/gjets_comparisons/{outtag}'
	if not os.path.exists(outdir):
		os.makedirs(outdir)

	outpath = pjoin(outdir, f"{'_VS_'.join(samples)}_{'inclusive' if inclusive else 'vbf'}_{distribution}.pdf")
	fig.savefig(outpath)

	print(f'File saved: {outpath}')

def main():
	inpath = sys.argv[1]

	# Get the output tag name for output directory naming
	if inpath.endswith('/'):
		outtag = inpath.split('/')[-2]
	else:
		outtag = inpath.split('/')[-1]

	acc = dir_archive(
		inpath,
		serialized=True,
		memsize=1e3,
		compression=0
	)

	acc.load('sumw')
	acc.load('sumw2')

	# Distributions to compare against
	distributions = ['vpt', 'mjj']

	to_compare = [
		('GJets_HT_2016', 'GJets_HT_2017'),
		('GJets_HT_2016', 'GJets_DR-0p4_HT_2017'),
		('GJets_HT_2017', 'GJets_DR-0p4_HT_2017'),
		('GJets_HT_2016', 'GJets_HT_2017', 'GJets_DR-0p4_HT_2017')
	]

	for samples in to_compare:
		for distribution in distributions:
			compare_two_gjets_samples(acc, samples=samples, inclusive=True, outtag=outtag, distribution=distribution)
			compare_two_gjets_samples(acc, samples=samples, inclusive=False, outtag=outtag, distribution=distribution)

if __name__ == '__main__':
	main()


	

	





