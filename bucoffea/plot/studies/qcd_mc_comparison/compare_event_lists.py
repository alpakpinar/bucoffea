#!/usr/bin/env python

import os
import sys
import uproot
import numpy as np
import pandas as pd

pjoin = os.path.join

# Event list from IC
ic_file = 'MTR.txt'
event_list_ic = np.loadtxt(ic_file, dtype=str)[:,1]

df_ic = pd.read_csv(ic_file, sep=', ', names=['a', 'event', 'b', 'region', 'c'])
events_ic = df_ic[['event', 'region']]

# BU input tree
bu_file = './qcd_10Nov20/tree_QCD_HT700to1000-mg_new_pmx_2017.root'
region = 'sr_vbf_qcd_cr'
events_bu = uproot.open(bu_file)[region].pandas.df()[['event', 'mjj']]

merged_df = pd.merge(
    events_ic,
    events_bu,
    how='left',
    on='event',
    suffixes=['_ic','_bu']
)

# Find out which events are not present on BU side
events_not_present_on_bu = merged_df[np.isnan(merged_df['mjj'])]['event']

# Dump the event list to a csv file
events_not_present_on_bu.to_csv('events_not_present_on_bu.txt', index=False)

