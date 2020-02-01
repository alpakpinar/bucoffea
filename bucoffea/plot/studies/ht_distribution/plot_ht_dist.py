#!/usr/bin/env python

import os
import re
import sys
from bucoffea.plot.util import merge_datasets, merge_extensions, scale_xs_lumi
from klepto.archives import dir_archive
from coffea import hist
from matplotlib import pyplot as plt

def plot_ht_dist(acc, inpath, regex, tag):
	'''Given the accumulator and the dataset regex,
	   plot the HT distribution.'''
	acc.load('lhe_ht')
	h = acc['lhe_ht']

	# Get directory name for output
	# using the inpath provided
	split_path = inpath.split('/')
	if inpath.endswith('/'):
		out_tag = split_path[-2]

	else:
		out_tag = split_path[-1]

	h = merge_extensions(h, acc, reweight_pu=False)
	scale_xs_lumi(h)
	h = merge_datasets(h)

	# Choose the relevant dataset(s)
	h = h[re.compile(regex)]

	new_ht_bins = hist.Bin('ht', r'$H_T \ (GeV)$', 50, 0, 4000)
	h = h.rebin('ht', new_ht_bins)

	# Plot the HT distribution
	fig, ax = plt.subplots(1,1)
	hist.plot1d(h, ax=ax, overflow='all', binwnorm=True, overlay='dataset')
	ax.set_yscale('log')
	ax.set_ylim(1e-3, 1e8)
	if 'gjets' in tag:
		ax.plot([600, 600], [1e-3, 1e8])

	if not os.path.exists(f'./output/{out_tag}'):
		os.makedirs(f'./output/{out_tag}')

	fig.savefig(f'./output/{out_tag}/{tag}_lhe_ht.pdf')

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

	plot_ht_dist(acc, inpath, regex='WJetsToLNu.*(2017|2018)', tag='wjets')
	plot_ht_dist(acc, inpath, regex='DYJets.*(2017|2018)', tag='dy')
	plot_ht_dist(acc, inpath, regex='GJets_HT.*(2017)', tag='gjets_17')
	plot_ht_dist(acc, inpath, regex='GJets_DR-0p4.*(2017)', tag='gjets_dr_17')

if __name__ == '__main__':
	main()





