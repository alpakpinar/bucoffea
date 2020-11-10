#!/usr/bin/env python

import os
import sys
import uproot
import argparse
import numpy as np
import pandas as pd

pjoin = os.path.join

def parse_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('bu_file', help='BU input tree.')
    parser.add_argument('ic_file', help='IC input tree.')
    parser.add_argument('--region', help='The region to look at.', default='sr_vbf_qcd_regionA')
    args = parser.parse_args()
    return args


def main():
    args = parse_cli()
    bu_file = args.bu_file
    ic_file = args.ic_file
    region = args.region

    # Read from the IC input file
    df_ic = pd.read_csv(ic_file, sep=', ', header=None)
    events_ic = df_ic.loc[:,1]
    events_ic = events_ic.reset_index(name='event')

    # Read from the BU input file
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

if __name__ == '__main__':
    main()
