#!/usr/bin/env python

import os
import sys
import re
import numpy as np

from matplotlib import pyplot as plt

pjoin = os.path.join

log = np.log

additive_factors_for_comb_eq = { 
    2017: {
        'jetFlav1' : {
            '20_to_30'    : 0.1161959320306778,
            '30_to_50'    : 0.045411378145217896,
            '50_to_70'    : 0.041932329535484314,
            '70_to_100'   : 0.037821229547262192,
            '100_to_140'  : 0.041939254850149155,
            '140_to_200'  : 0.045033644884824753,
            '200_to_300'  : 0.1036531925201416,
            '300_to_600'  : 0.12050666660070419,
            '600_to_1001' : 0.16405443847179413,
        },
        'jetFlav0' : {
            '20_to_30'    : 0.038731977343559265,
            '30_to_50'    : 0.015137125737965107,
            '50_to_70'    : 0.013977443799376488,
            '70_to_100'   : 0.012607076205313206,
            '100_to_140'  : 0.013979751616716385,
            '140_to_200'  : 0.015011214651167393,
            '200_to_300'  : 0.034551065415143967,
            '300_to_600'  : 0.040168888866901398,
            '600_to_1001' : 0.054684814065694809,
        }
    },
    2018: {
        'jetFlav1' : {
            '20_to_30'    : 0.19771461188793182,
            '30_to_50'    : 0.045167062431573868,
            '50_to_70'    : 0.040520280599594116,
            '70_to_100'   : 0.045320175588130951,
            '100_to_140'  : 0.043860536068677902,
            '140_to_200'  : 0.036484666168689728,
            '200_to_300'  : 0.048719070851802826,
            '300_to_600'  : 0.11997123062610626,
            '600_to_1001' : 0.20536302030086517,
        },
        'jetFlav0' : {
            '20_to_30'    : 0.065904870629310608,
            '30_to_50'    : 0.015055687166750431,
            '50_to_70'    : 0.013506759889423847,
            '70_to_100'   : 0.015106724575161934,
            '100_to_140'  : 0.014620178379118443,
            '140_to_200'  : 0.012161554768681526,
            '200_to_300'  : 0.016239689663052559,
            '300_to_600'  : 0.039990410208702087,
            '600_to_1001' : 0.068454340100288391,
        }
    }
}

def comb_sf_for_2017(x, var='central', jetFlav=None, pt_range=None):
    central_eq = 2.22144*((1.+(0.540134*x))/(1.+(1.30246*x)))
    if pt_range is not None:
        pt_range_split = pt_range.split('_')
        pt_lo, pt_hi = int(pt_range_split[0]), int(pt_range_split[-1])

    if var == 'central':
        return central_eq
    elif var == 'up':
        return ((pt_lo <= x) & (x < pt_hi)) * (central_eq + additive_factors_for_comb_eq[2017][jetFlav][pt_range])
    elif var == 'down':
        return ((pt_lo <= x) & (x < pt_hi)) * (central_eq - additive_factors_for_comb_eq[2017][jetFlav][pt_range])

def comb_sf_for_2018(x, var='central', jetFlav=None, pt_range=None):
    central_eq = 0.909339+(0.00354*(log(x+19)*(log(x+18)*(3-(0.471623*log(x+18))))))
    if pt_range is not None:
        pt_range_split = pt_range.split('_')
        pt_lo, pt_hi = int(pt_range_split[0]), int(pt_range_split[-1])

    if var == 'central':
        return central_eq
    elif var == 'up':
        return ((pt_lo <= x) & (x < pt_hi)) * (central_eq + additive_factors_for_comb_eq[2018][jetFlav][pt_range])
    elif var == 'down':
        return ((pt_lo <= x) & (x < pt_hi)) * (central_eq - additive_factors_for_comb_eq[2018][jetFlav][pt_range])

def incl_sf_for_2017(x, var='central'):
    if var == 'central':
        return 0.972902+0.000201811*x+3.96396e-08*x**2+-4.53965e-10*x**3
    elif var == 'down':
        return (0.972902+0.000201811*x+3.96396e-08*x**2+-4.53965e-10*x**3)*(1-(0.101236+0.000212696*x+-1.71672e-07*x**2))
    elif var == 'up':
        return (0.972902+0.000201811*x+3.96396e-08*x**2+-4.53965e-10*x**3)*(1+(0.101236+0.000212696*x+-1.71672e-07*x**2))

def incl_sf_for_2018(x, var='central'):
    if var == 'central':
        return 1.6329+-0.00160255*x+1.9899e-06*x**2+-6.72613e-10*x**3
    elif var == 'down':
        return (1.6329+-0.00160255*x+1.9899e-06*x**2+-6.72613e-10*x**3)*(1-(0.122811+0.000162564*x+-1.66422e-07*x**2))
    elif var == 'up':
        return (1.6329+-0.00160255*x+1.9899e-06*x**2+-6.72613e-10*x**3)*(1+(0.122811+0.000162564*x+-1.66422e-07*x**2))

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

def plot_btag_variations_as_a_func_of_pt(method='incl'):
    '''From the equations given in the csv source files, plot the b-weight variations as a function of jet pt.'''
    outdir = './output/btag_weights_from_src'
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    x = np.linspace(20,1000)
    for year in [2017, 2018]:
        fig, ax = plt.subplots()
        for var in ['central', 'up', 'down']:
            if year == 2017:
                sf = incl_sf_for_2017(x, var=var)
                ax.plot(x,sf,label=var)
            elif year == 2018:
                sf = incl_sf_for_2018(x, var=var)
                ax.plot(x,sf,label=var)

        
        ax.set_xlabel(r'Jet $p_T \ (GeV)$', fontsize=14)
        ax.set_ylabel('b-tag SF', fontsize=14)

        ax.set_title(f'Fake b-tag SF, {year}', fontsize=14)
        ax.legend(title='Variation')

        outpath = pjoin(outdir, f'fake_btag_variations_{year}.pdf')
        fig.savefig(outpath)
        plt.close(fig)
        print(f'File saved: {outpath}')

def plot_btag_variations_for_comb_measurement():
    outdir = './output/btag_weights_from_src'
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    pt_bins = additive_factors_for_comb_eq[2017]['jetFlav1'].keys()

    x = np.linspace(20,1000,99)
    for year in [2017, 2018]:
        fig, ax = plt.subplots()
        if year == 2017:
            central_sf = comb_sf_for_2017(x)
        elif year == 2018:
            central_sf = comb_sf_for_2018(x)
        
        ax.plot(x, central_sf, label='Central SF')

        up_sf = np.zeros_like(x)
        down_sf = np.zeros_like(x)

        for pt_bin in pt_bins:
            if year == 2017:
                sf_up =  comb_sf_for_2017(x,var='up',jetFlav='jetFlav0',pt_range=pt_bin)
                sf_down =  comb_sf_for_2017(x,var='down',jetFlav='jetFlav0',pt_range=pt_bin)
            elif year == 2018:
                sf_up =  comb_sf_for_2018(x,var='up',jetFlav='jetFlav0',pt_range=pt_bin)
                sf_down =  comb_sf_for_2018(x,var='down',jetFlav='jetFlav0',pt_range=pt_bin)

            up_sf += sf_up 
            down_sf += sf_down

        ax.plot(x, up_sf, label='SF up')
        ax.plot(x, down_sf, label='SF down')

        ax.legend()
        ax.set_xlabel(r'Jet $p_T$ (GeV)', fontsize=14)
        ax.set_ylabel('b-tag SF', fontsize=14)

        ax.set_title(f'Real b-tag SF, {year}', fontsize=14)

        outpath = pjoin(outdir, f'real_btag_variations_{year}.pdf')
        fig.savefig(outpath)
        plt.close(fig)
        print(f'File saved: {outpath}')

def main():
    plot_btag_as_a_func_of_pt()
    # Variations of fake SF
    plot_btag_variations_as_a_func_of_pt()
    # Variations of real SF
    plot_btag_variations_for_comb_measurement()

if __name__ == '__main__':
    main()