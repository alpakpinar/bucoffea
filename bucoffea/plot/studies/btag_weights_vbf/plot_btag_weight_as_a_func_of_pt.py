#!/usr/bin/env python

import os
import sys
import re
import numpy as np

from matplotlib import pyplot as plt

pjoin = os.path.join

def comb_sf_for_2017(x):
    return 2.22144*((1.+(0.540134*x))/(1.+(1.30246*x)))

def comb_sf_for_2018(x):
    return 0.909339+(0.00354*(np.log(x+19)*(np.log(x+18)*(3-(0.471623*np.log(x+18))))))

def incl_sf_for_2017(x):
    return 0.972902+0.000201811*x+3.96396e-08*x**2+-4.53965e-10*x**3

def incl_sf_for_2018(x):
    return 1.6329+-0.00160255*x+1.9899e-06*x**2+-6.72613e-10*x**3

def plot_btag_as_a_func_of_pt(method='incl'):
    '''From the equations given in the csv source files, plot the central b-weights as a function of jet pt.'''
    outdir = './output/btag_weights_from_src'
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    x = np.linspace(20,1000)

    sf_2017_comb = comb_sf_for_2017(x)
    sf_2018_comb = comb_sf_for_2018(x)

    sf_2017_incl = incl_sf_for_2017(x)
    sf_2018_incl = incl_sf_for_2018(x)

    fig, ax = plt.subplots()

    ax.axhline(1, xmin=0, xmax=1, color='k', ls='--')

    ax.plot(x, sf_2017_comb, label='2017, comb')
    ax.plot(x, sf_2018_comb, label='2018, comb')

    ax.plot(x, sf_2017_incl, label='2017, incl')
    ax.plot(x, sf_2018_incl, label='2018, incl')

    ax.legend(title='Year, Measurement Type')
    ax.set_xlabel(r'Jet $p_T$ (GeV)')
    ax.set_ylabel('b-tag SF')

    outpath = pjoin(outdir, 'bweights_comb.pdf')
    fig.savefig(outpath)
    plt.close(fig)
    print(f'File saved: {outpath}')

def main():
    plot_btag_as_a_func_of_pt()

if __name__ == '__main__':
    main()