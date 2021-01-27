import copy
import coffea.processor as processor
import re
import numpy as np

from coffea import hist
from coffea.analysis_objects import JaggedCandidateArray

from bucoffea.helpers import object_overlap, weight_shape
from bucoffea.helpers.dataset import extract_year, is_data
from bucoffea.helpers.gen import setup_lhe_cleaned_genjets

Hist = hist.Hist
Bin = hist.Bin
Cat = hist.Cat

def trigger_selection(selection, df):
    '''HLT_PFJet40 requirement.'''
    trigger='HLT_PFJet40'
    selection.add('jet_trig', df[trigger])
    return selection

def jetht_accumulator():
    dataset_ax = Cat("dataset", "Primary dataset")
    region_ax = Cat("region", "Selection region")

    ht_ax = Bin("ht", r"$H_{T}$ (GeV)", 50, 0, 5000)
    htmiss_ax = Bin("htmiss", r"$H_{T}^{miss}$ (GeV)", 500, 0, 1500)

    jet_pt_ax = Bin("jetpt", r"$p_{T}$ (GeV)", 50, 0, 1000)
    jet_eta_ax = Bin("jeteta", r"$\eta$", 50, -5, 5)

    items = {}
    items["htmiss"] = Hist("Counts", dataset_ax, region_ax, htmiss_ax)
    items["ht"] = Hist("Counts", dataset_ax, region_ax, ht_ax)

    # 2D histograms
    items["htmiss_ht"] = Hist("Counts", dataset_ax, region_ax, htmiss_ax, ht_ax)

    return processor.dict_accumulator(items)

def jetht_regions():
    regions = {}
    regions['inclusive'] = ['inclusive']
    regions['trig_pass'] = ['inclusive', 'jet_trig']

    return regions

def setup_jets(df):
    if extract_year(df['dataset']) != 2018:
        # 2016, 2017 data
        jes_suffix = ''
    else:
        # 2018 data
        jes_suffix = '_nom'

    ak4 = JaggedCandidateArray.candidatesfromcounts(
        df['nJet'],
        pt=df[f'Jet_pt{jes_suffix}'],
        eta=df['Jet_eta'],
        abseta=np.abs(df['Jet_eta']),
        phi=df['Jet_phi'],
        mass=np.zeros_like(df['Jet_pt']),
        looseId=(df['Jet_jetId']&2) == 2, # bitmask: 1 = loose, 2 = tight, 3 = tight + lep veto
        tightId=(df['Jet_jetId']&2) == 2, # bitmask: 1 = loose, 2 = tight, 3 = tight + lep veto
    )

    return ak4

class jethtProcessor(processor.ProcessorABC):
    def __init__(self, blind=False):
        self._accumulator = jetht_accumulator()

    @property
    def accumulator(self):
        return self._accumulator

    def process(self, df):
        if not df.size:
            return self.accumulator.identity()
        dataset = df['dataset']
        df['is_data'] = is_data(dataset)
        
        ak4 = setup_jets(df)

        # Calculate HT and HTmiss
        htmiss = ak4[ak4.pt>30].p4.sum().pt
        ht = ak4[ak4.pt>30].pt.sum()
        
        selection = processor.PackedSelection()
        pass_all = np.zeros(df.size) == 0
        selection.add('inclusive', pass_all)

        selection = trigger_selection(selection, df)

        # Fill histograms
        output = self.accumulator.identity()

        regions = jetht_regions()
        for region, cuts in regions.items():
            mask = selection.all(*cuts)

            def ezfill(name, **kwargs):
                """Helper function to make filling easier."""
                output[name].fill(
                                  dataset=dataset,
                                  region=region,
                                  **kwargs
                                  )

            ezfill('htmiss', htmiss=htmiss[mask])
            ezfill('ht', ht=ht[mask])

            ezfill('htmiss_ht', htmiss=htmiss[mask], ht=ht[mask])

        return output

    def postprocess(self, accumulator):
        return accumulator

