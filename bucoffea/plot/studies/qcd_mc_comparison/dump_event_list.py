#!/usr/bin/env python

import os
import sys
import uproot
import numpy as np
import pandas as pd

pjoin = os.path.join

# BU input tree
bu_file = './qcd_10Nov20/tree_QCD_HT700to1000-mg_new_pmx_2017.root'
region = 'sr_vbf_qcd_regionA'
events_bu = uproot.open(bu_file)[region].pandas.df()['event']

# Dump event list to a text file
outdir = './output'
if not os.path.exists(outdir):
    os.makedirs(outdir)

outpath = pjoin(outdir, f'bu_event_list_{region}.txt')
events_bu.to_csv(outpath, index=False)