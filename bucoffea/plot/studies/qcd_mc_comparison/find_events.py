#!/usr/bin/env python

import os
import sys
import re
import uproot
import pandas as pd
from pprint import pprint
from tqdm import tqdm

pjoin = os.path.join

# Path to QCD HT700to1000 2017 samples
INPATH='/eos/uscms/store/user/aandreas/nanopost/03Sep20v7/QCD_HT700to1000_TuneCP5_13TeV-madgraph-pythia8/QCD_HT700to1000-mg_new_pmx_2017/200925_182623/0000'

files = [f for f in os.listdir(INPATH) if f.endswith('.root')]

def search_for_event(files, event_to_search):
    '''Loop through files and search for a specific event.'''

    for filename in tqdm(files):
        filepath = pjoin(INPATH, filename)
        filepath = re.sub('/eos/uscms', 'root://cmsxrootd.fnal.gov//', filepath)
        # print(f'Working on : {filepath}')
        d = uproot.open(filepath)['Events']
        events = d['event'].array()
    
        event_in_file = event_to_search in events
        if event_in_file:
            print(f'FOUND EVENT {event_to_search}: {filepath}')
            return

    print(f"Couldn't find event: {event_to_search}")

BU_FILE = './output/bu_event_list_sr_vbf_qcd_regionA.txt'

def check_bu_file(event):
    df = pd.read_csv(BU_FILE, sep=', ', names=['event'])
    if event in df:
        print('Also on BU file!')
    else:
        print('Not found in BU file!')

df = pd.read_csv('MTRA.txt', delimiter=', ', names=['HT', 'event', 'mjj', 'met', 'a', 'b', 'c', 'd', 'e'])
events_to_search = df['event']

for event in events_to_search:
    search_for_event(files, event)
    # check_bu_file(event)
