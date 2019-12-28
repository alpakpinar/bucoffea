#!/usr/bin/env python

import os
import sys
import re
from coffea import hist
from bucoffea.plot.util import (merge_datasets, 
								merge_extensions, 
								scale_xs_lumi)
							
from klepto.archives import dir_archive
from matplotlib import pyplot as plt

REBIN = {
	'mjj' : hist.Bin('mjj', r'$M_{jj}$ (GeV)', list(range(200,800,300)) + list(range(800,2000,400)) + [2000, 2750, 3500]),
	'vpt' : hist.Bin('pt', r'$p_T(V)$ (GeV)', 40, 0, 800) 
}

AX_LABELS = {
	'mjj' : r'$M_{jj}$ (GeV)',
	'vpt' : r'$p_T(V)$ (GeV)'
}

def plot_gen_spectrum(acc, tag='stat1', variable='vpt'):
	'''Plot 1D gen pt or mjj distribution for LO and NLO
	GJets samples on the same canvas.
	===============
	PARAMETERS
	acc : Input accumulator.
	tag : Type of pt (stat1 or dress). 
		  Default is stat1.
	variable : The variable to plot. 
			   Should be specified as "mjj" or "vpt".
			   Defualt is "vpt".
	===============
	'''
	if variable not in ['mjj', 'vpt']:
		raise ValueError(f'{variable}: Not a valid argument for variable. Should be specified as "mjj" or "vpt"')

	# Specify the variable to integrate over
	if variable == 'vpt':
		integrate = 'mjj'
	else:
		integrate = 'vpt'

	dist = f'gen_vpt_vbf_{tag}'
	acc.load(dist)
	histogram = acc[dist]

	# Merge datasets/extensions, 
	# scale the histogram according to x-sec
	histogram = merge_extensions(histogram, acc, reweight_pu=False)
	scale_xs_lumi(histogram)
	histogram = merge_datasets(histogram)

	# Rebin the vpt axis
	new_bin = REBIN['vpt']
#	histogram = histogram.rebin('vpt', new_bin)

	# LO and NLO GJets samples
	dataset = re.compile('G\d?Jet.*_(HT|Pt).*')

	histogram = histogram.integrate('jpt').integrate(f'{integrate}') 

	# Plot the histogram and save the figure
	fig, ax = plt.subplots(1,1, figsize=(7,5))
	hist.plot1d(histogram[dataset], ax=ax, overlay='dataset', binwnorm=True) 

	#ax.set_yscale('log')
	ax.set_ylabel('Events / Bin Width')
	ax.set_xlabel(AX_LABELS[variable])
	ax.set_title(r'LO and NLO GJets Comparison')

	outpath = f'./output/gen_{variable}.pdf'
	fig.savefig(outpath)
	print(f'Saved histogram in {outpath}')

def main():
	inpath = sys.argv[1]
	
	acc = dir_archive(
			inpath,
			serialized=True,
			compression=0,
			memsize=1e3
		)

	acc.load('sumw')
	acc.load('sumw2')

	plot_gen_spectrum(acc, variable='vpt')
	plot_gen_spectrum(acc, variable='mjj')


if __name__ == '__main__':
	main()

